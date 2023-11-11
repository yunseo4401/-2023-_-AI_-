import re
from typing import List
import pandas as pd
from konlpy.tag import Okt
from textrankr import TextRank

class PreProcessor:
    def __init__(self):
        self.okt = Okt()
        self.tokenizer = self.OktTokenizer()
        self.textrank = TextRank(self.tokenizer)

    class OktTokenizer:
        def __init__(self):
            self.okt = Okt()

        def __call__(self, text: str) -> List[str]:
            tokens: List[str] = self.okt.phrases(text)
            return tokens

    def clean_text(self, text: str) -> str:
        text = re.sub(' +', ' ', text)
        text = re.sub('\n', '', text)
        text = re.sub('\u200b', '', text)
        text = re.sub('\xa0', '', text)
        text = re.sub('[-=+,#/\?:^$@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text)
        return text

    def calculate_sent_count(self, text: str) -> int:
        sentences = re.split(r'[.!?]', text)
        non_empty_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return len(non_empty_sentences)
    
    def apply_textrank(self, articles_df, summary_length=20):
        summaries = []

        for index, row in articles_df.iterrows():
            text = row['content']
            cleaned_text = self.clean_text(text)
            sent_count = self.calculate_sent_count(cleaned_text)

            if sent_count > summary_length:
                tr = self.textrank.summarize(cleaned_text, summary_length)
                summaries.append(tr)
            else:
                summaries.append(cleaned_text)

            articles_df.at[index, 'sent_count'] = sent_count
            articles_df.at[index, 'TextRank'] = summaries[index]
