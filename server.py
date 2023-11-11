# -*- coding: utf-8 -*-
from flask import Flask,render_template, request
from trinity import KoGPT_Trinity #kogpt3 모듈
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from poly_encoder import PolyEncoderModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
kogpt_path = "C:\\Users\diaky\\OneDrive\\Documents\\바탕 화면\\model\kogpt_finetuning"#kogpt3가중치파일 경로 
trinity = KoGPT_Trinity(kogpt_path) #kogpt3-trinity 객체 설정

df = pd.read_csv("C:\\생성형AI_SKP-서비스구현\\mirae_chatbot\\poly-encoder답변모음.csv") #poly-encoder에서 retrieve할 답변 db 
df = df.drop(columns=['Unnamed: 0'])
data = df.iloc[:-4758]
data = data[data['answer'].notna() & (data['answer'] != ' ')]

poly_path = "C:\\Users\\diaky\\OneDrive\\Documents\\바탕 화면\\model\\poly-encoder-model.bin"#poly-encdoer fine-tuning파일 경로. 경로에 한글이 포함되어있으면 인코딩 오류가 날 수도 있어서 경로에 영어만 포함하도록 짬. 
bert_name = 'klue/bert-base'
poly_encoder = PolyEncoderModel(bert_name, poly_path) #poly-encoder 객체설정




app=Flask(__name__)
@app.route("/",methods=['GET','POST'])
def index():
  if request.method == 'GET':
    return render_template('index.html')
  elif request.method == 'POST':
    question = str(request.form['name'])
  poly_encoder_answer=''
  poly_encoder_answer = poly_encoder.get_top_similar_candidates(question, data)
  poly_encoder_answer = poly_encoder.get_top_answer(question, poly_encoder_answer) 
  return render_template('index.html', poly_encoder_answer=poly_encoder_answer)

 

@app.route("/mean", methods=['GET','POST'])
def means(): 
  if request.method == 'GET':
    return render_template('index.html')
  elif request.method == 'POST':
    q_vocab = str(request.form['new_name'])
  kogpt_anwer=''
  kogpt_answer= trinity.generate_response(q_vocab,'YES')
  kogpt_answer = trinity.get_first_sentence(kogpt_answer)
  return render_template('index.html', kogpt_answer=kogpt_answer)


  
if __name__ == '__main__':
   app.run(debug = True)


  