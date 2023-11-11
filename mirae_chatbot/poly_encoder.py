import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
from encoder import PolyEncoder
from transform import SelectionJoinTransform, SelectionSequentialTransform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PolyEncoderModel(nn.Module):
    def __init__(self, bert_name, model_path, poly_m=16):
        super(PolyEncoderModel, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained BERT model and tokenizer
        bert_config = BertConfig.from_pretrained(bert_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.tokenizer.add_tokens(['\n'], special_tokens=True)
        self.bert = BertModel.from_pretrained(bert_name, config=bert_config)

        # Initialize PolyEncoder model
        self.model = PolyEncoder(bert_config, bert=self.bert, poly_m=poly_m)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)

        # Initialize transformation functions
        self.context_transform = SelectionJoinTransform(tokenizer=self.tokenizer, max_len=256)
        self.response_transform = SelectionSequentialTransform(tokenizer=self.tokenizer, max_len=128)

    def get_top_similar_candidates(self, context, data, top_n=10):
        # TF-IDF 벡터화를 위한 객체를 생성합니다.
        tfidf_vectorizer = TfidfVectorizer()
        print(len(data['answer']))
        lst = []
        for candidate in data['answer']:
            # 두 문장을 TF-IDF 벡터로 변환합니다.
            tfidf_matrix = tfidf_vectorizer.fit_transform([context, candidate])
            # 코사인 유사도를 계산합니다.
            cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
            lst.append(cosine_sim)
        print(lst)
        print(len(lst))

        # lst 내부의 numpy 배열을 일차원으로 평탄화하고 정렬
        flattened_lst = [item.flatten()[0] for item in lst]
        sorted_indices = np.argsort(flattened_lst)[::-1][:10]

        # # Get indices of the top similar candidates
        for i in range(10):
            print('top 10 smi:',lst[sorted_indices[i]])
            print('top 10 answer:', data['answer'][sorted_indices[i]])

        # Retrieve the top similar candidates
        top_candidates = [data['answer'][index] for index in sorted_indices]
        
        return top_candidates


    def context_input(self, context):
        context_input_ids, context_input_masks = self.context_transform(context)
        contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]

        long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]

        contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)

        return contexts_token_ids_list_batch, contexts_input_masks_list_batch

    def response_input(self, candidates):
        responses_token_ids_list, responses_input_masks_list = self.response_transform(candidates)
        responses_token_ids_list_batch, responses_input_masks_list_batch = [responses_token_ids_list], [responses_input_masks_list]

        long_tensors = [responses_token_ids_list_batch, responses_input_masks_list_batch]

        responses_token_ids_list_batch, responses_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)

        return responses_token_ids_list_batch, responses_input_masks_list_batch

    def embs_gen(self, contexts_token_ids_list_batch, contexts_input_masks_list_batch):
        with torch.no_grad():
            self.model.eval()
            ctx_out = self.model.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
            poly_code_ids = torch.arange(self.model.poly_m, dtype=torch.long).to(contexts_token_ids_list_batch.device)
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, self.model.poly_m)
            poly_codes = self.model.poly_code_embeddings(poly_code_ids)  # [bs, poly_m, dim]
            embs = self.model.dot_attention(poly_codes, ctx_out, ctx_out)  # [bs, poly_m, dim]
        return embs

    def cand_emb_gen(self, responses_token_ids_list_batch, responses_input_masks_list_batch):
        with torch.no_grad():
            self.model.eval()
            batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape
            responses_token_ids_list_batch = responses_token_ids_list_batch.view(-1, seq_length)
            responses_input_masks_list_batch = responses_input_masks_list_batch.view(-1, seq_length)
            cand_emb = self.model.bert(responses_token_ids_list_batch, responses_input_masks_list_batch)[0][:,0,:] # [bs, dim]
            cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]
        return cand_emb

    def loss(self, embs, cand_emb, contexts_token_ids_list_batch, responses_token_ids_list_batch):
        batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape
        ctx_emb = self.model.dot_attention(cand_emb, embs, embs) # [bs, bs, dim]
        ctx_emb = ctx_emb.squeeze()
        dot_product = (ctx_emb * cand_emb) # [bs, bs]
        dot_product = dot_product.sum(-1)
        mask = torch.eye(batch_size).to(contexts_token_ids_list_batch.device) # [bs, bs]
        loss = F.log_softmax(dot_product, dim=-1)
        loss = loss * mask
        loss = (-loss.sum(dim=1))
        loss = loss.mean()
        return loss

    def score(self, embs, cand_emb):
        with torch.no_grad():
            self.model.eval()
            ctx_emb = self.model.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb * cand_emb).sum(-1)
        return dot_product

    def get_top_answer(self, context, candidates):
        contexts_token_ids_list_batch, contexts_input_masks_list_batch = self.context_input(context)
        responses_token_ids_list_batch, responses_input_masks_list_batch = self.response_input(candidates)

        embs = self.embs_gen(contexts_token_ids_list_batch, contexts_input_masks_list_batch)
        cand_emb = self.cand_emb_gen(responses_token_ids_list_batch, responses_input_masks_list_batch)
        
        # Calculate scores
        score_ = self.score(embs, cand_emb)

        # Find the index of the top-scoring answer
        max_value, max_index = torch.max(score_, dim=1, keepdim=True)
        top_answer = candidates[max_index.item()]

        return top_answer