from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from Crawler import Crawler
from PreProcessor import PreProcessor
from PostProcessor import PostProcessor

class Summarizer:
    def __init__(self, stock, model_checkpoint, model_weights_path=None):
        self.stock = stock
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__result_df = None

    def summarize(self, text, max_length=1000):
        # 모델에 입력하고 요약하는 코드
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0).to(self.__device)
        generated = self.model.generate(input_ids, max_length=max_length)
        summary = self.tokenizer.decode(generated[0])
        return summary
    
    # 크롤링 + 요약 결과 데이터프레임 생성
    def process_news(self):
        crawler = Crawler(self.stock)
        self.__result_df = crawler.crawling()
        print(len(self.__result_df))
        for index, row in self.__result_df.iterrows():
            title, content = crawler.extract_title_content(row)
            print(content)
            self.__result_df.at[index, 'title'] = title
            self.__result_df.at[index, 'content'] = content

        pp = PreProcessor()
        pp.apply_textrank(self.__result_df)

        for index, row in self.__result_df.iterrows():
            content_to_summarize = row['TextRank']
            summary = self.summarize(content_to_summarize)
            self.__result_df.at[index, 'summary'] = summary
            print(summary)

    def post_process(self):
        # 위의 요약 결과 데이터프레임 후처리하기
        post = PostProcessor(self.stock, self.__result_df)
        return post.post_process_summary()
    
    def get_summary_df(self):
        return self.__result_df