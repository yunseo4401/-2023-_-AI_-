# -*- coding: utf-8 -*-
from Summarizer import Summarizer
from flask import Flask,render_template, request
save_path=r"C:\Users\diaky\OneDrive\Documents\model\model_GPU_EDA완.pth"
model_checkpoint = "eenzeenee/t5-base-korean-summarization"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
app=Flask(__name__)
@app.route("/",methods=['GET','POST'])
def index():
  if request.method == 'GET':
    return render_template('index.html')
  if request.method == 'POST':
    name= str(request.form['name'])
  summary=' '
  news_processor = Summarizer(name, model_checkpoint, save_path)
  news_processor.process_news()
  summary=news_processor.post_process()
  summary = summary.replace('\n', '<br>')
  
  return render_template('index.html',summary=summary)
  
if __name__ == '__main__':
   app.run(debug = True)