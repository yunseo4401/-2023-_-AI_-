import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

# BiEncoder 클래스는 BERT를 사용하여 텍스트 인코딩 및 유사성을 계산하는 모델입니다.
class BiEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']

    def forward(self, context_input_ids, context_input_masks,
                            responses_input_ids, responses_input_masks, labels=None):
        # 레이블이 주어진 경우, 첫 번째 응답만 선택합니다.
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)

        # 문맥 벡터 계산: BERT 모델을 사용하여 문맥 문장을 인코딩하고 첫 번째 토큰의 임베딩을 선택합니다.
        context_vec = self.bert(context_input_ids, context_input_masks)[0][:,0,:]  # [bs,dim]

        # 응답 벡터 계산: BERT 모델을 사용하여 응답 문장을 인코딩하고 첫 번째 토큰의 임베딩을 선택합니다.
        batch_size, res_cnt, seq_length = responses_input_ids.shape
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        responses_vec = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:]  # [bs,dim]
        responses_vec = responses_vec.view(batch_size, res_cnt, -1)

        if labels is not None:
            # 문맥 벡터와 응답 벡터 간의 내적을 계산하고 대각선을 제외한 부분을 마스킹하여 손실을 계산합니다.
            responses_vec = responses_vec.squeeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.t())  # [bs, bs]
            mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            # 문맥 벡터와 응답 벡터 간의 내적을 계산하여 유사도를 반환합니다.
            context_vec = context_vec.unsqueeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.permute(0, 2, 1)).squeeze()
            return dot_product

# PolyEncoder 클래스는 다중 인코딩을 사용하여 텍스트 간의 상호 작용을 모델링하는 모델입니다.
class PolyEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.poly_m = kwargs['poly_m']
        self.poly_code_embeddings = nn.Embedding(self.poly_m, config.hidden_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, config.hidden_size ** -0.5)

    def dot_attention(self, q, k, v):
        # 닷 어텐션 메커니즘을 사용하여 쿼리, 키, 밸류 간의 관련성을 계산합니다.
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

    def forward(self, context_input_ids, context_input_masks,
                            responses_input_ids, responses_input_masks, labels=None):
        # 레이블이 주어진 경우, 첫 번째 응답만 선택합니다.
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
        batch_size, res_cnt, seq_length = responses_input_ids.shape # res_cnt is 1 during training

        # 문맥 인코더
        ctx_out = self.bert(context_input_ids, context_input_masks)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]

        # 응답 인코더
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        cand_emb = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        # 병합
        if labels is not None:
            # 훈련 중에는 응답을 재활용하여 빠른 훈련을 위해 응답을 반복하여 사용합니다.
            # 모든 문맥이 배치 크기만큼의 응답과 짝을 이루도록 합니다.
            cand_emb = cand_emb.permute(1, 0, 2) # [1, bs, dim]
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2]) # [bs, bs, dim]
            ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze() # [bs, bs, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1) # [bs, bs]
            mask = torch.eye(batch_size).to(context_input_ids.device) # [bs, bs]
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            ctx_emb = self.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1)
            return dot_product