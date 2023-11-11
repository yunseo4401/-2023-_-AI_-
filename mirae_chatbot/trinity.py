from transformers import AutoModelWithLMHead, AutoTokenizer, PreTrainedTokenizerFast
import torch
import re
from typing import List
from konlpy.tag import Okt
from textrankr import TextRank

class KoGPT_Trinity:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5",
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')
        self.model = AutoModelWithLMHead.from_pretrained(model_path)
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_response(self, prompt, leverage):
        받침 = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        if leverage=='No':
          prompt=prompt
        else:
          last_char = prompt[-1]
          if '가' <= last_char <= '힣':
            char_code = ord(last_char)
            is_final_consonant = (char_code - ord('가')) % 28 != 0
            if is_final_consonant:
              prompt= prompt + '이란'
            else:
              prompt= prompt + '란'
          else:
            prompt=prompt
        prompt_ids = self.tokenizer.encode(prompt)
        inp = torch.tensor(prompt_ids)[None].to(self.__device)
        self.model = self.model.to(self.__device)
        print('답변생성시작')
        preds = self.model.generate(inp,
                                    max_length=256,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    bos_token_id=self.tokenizer.bos_token_id,
                                    repetition_penalty=2.0,
                                    use_cache=True
                                    )
        text = self.tokenizer.decode(preds[0].cpu().numpy())
        pattern = r'(니다)([\s\n]+)'
        replacement = r'\1.\2'
        result = re.sub(pattern, replacement, text)
        return result

   
    def get_first_sentence(self,result):
        match = re.search(r'[^.!?]*[.!?]', result)
        if match:
          first_sentence = match.group()
         
        
        return first_sentence.strip()  # 양 끝의 공백을 제거합니다.
    @staticmethod
    def summarize_text(text):
        class OktTokenizer:
            okt: Okt = Okt()

            def __call__(self, text: str) -> List[str]:
                tokens: List[str] = self.okt.phrases(text)
                return tokens

        mytokenizer: OktTokenizer = OktTokenizer()
        textrank: TextRank = TextRank(mytokenizer)
        summary = textrank.summarize(text, 1)
        return summary
     