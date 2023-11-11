import Summarizer

class PostProcessor:
    def __init__(self, stock, result):
        self.stock = stock
        self.result = result
        self.res=''

    def post_process_summary(self):
      # 요약 후처리 코드
      res=''
      

      # result dataframe 가져오기
      df = self.result
      print(df)

      # 그 중 summary 결측치 제거 + 공모주명 없는 거 제외
      dd2=df[df['summary']!='<pad>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       '][['date','summary','link']]
      dd2 = dd2[dd2.apply(lambda row: self.stock in row['summary'], axis=1)]
      dd2.reset_index(drop=True, inplace=True)
      print(dd2)
      if len(dd2)==0:
        return '준비중...'

      
      # summary 최신순으로 3개만 가져오기
      for i, d in enumerate(dd2['summary']):
        d = d.replace('<pad>','').replace('</s>','')
        res += str(i+1)+' : '+d+'\n'

        if i==2:
          break

      res+='<br>출처: '

      for i, d in enumerate(dd2['link']):
        res+='\n'+str(i+1)+': '+d

        if i==2:
          break
       
      self.res = res
      return res
    
    def get_res(self):
       return self.res

      
