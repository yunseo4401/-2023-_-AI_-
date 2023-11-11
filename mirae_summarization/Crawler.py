from selenium import webdriver
from selenium.webdriver.common.by import By
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import warnings
from datetime import datetime, timedelta

# Chrome WebDriver 설치
import subprocess
'''
# Run the 'apt-get update' command
try:
    subprocess.run(['apt-get', 'update'], check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

# Run the 'apt install' command
try:
    subprocess.run(['apt', 'install', '-y', 'chromium-chromedriver'], check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

# Define the shell command
command = "cp /usr/lib/chromium-browser/chromedriver /usr/bin"

# Run the command
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

import os
# Set the locale to UTF-8
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# Chrome WebDriver 설치
os.system('apt-get update')
os.system('apt install -y chromium-chromedriver')
os.system('cp /usr/lib/chromium-browser/chromedriver /usr/bin')

sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
'''
from selenium import webdriver
from selenium.webdriver.common.by import By

# Chrome WebDriver 실행
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 화면 출력 없이 실행하려면 이 옵션을 사용
options.add_argument('--no-sandbox')  # Colab 환경에서 실행할 때 필요한 옵션
options.add_argument('--disable-dev-shm-usage')  # Colab 환경에서 실행할 때 필요한 옵션
driver = webdriver.Chrome(options=options)  # 'chromedriver'는 PATH에 있는 경우 생략 가능

# 버전 확인
print("Chrome WebDriver Version:", driver.capabilities['chrome']['chromedriverVersion'])

# 혹시나 driver.get(base_url) 안 뺐다면..
base_url = "https://www.google.co.kr/"


# WebDriver 종료
driver.quit()

class Crawler:
  def __init__(self, stock):
    self.stock = stock
    self.keyword = f'"{stock}"%2B"공모주"'
    
  def convert_relative_date(self, relative_date):
      # 상대 날짜를 절대 날짜로 변환하는 코드
      if isinstance(relative_date, str):
          if '일 전' in relative_date:
              days_ago = int(relative_date.split('일 전')[0])
              current_date = datetime.today()
              modified_date = current_date - timedelta(days=days_ago)
              return modified_date.strftime('%Y.%m.%d.')  # Updated date format
          elif '시간 전' in relative_date:
              hours_ago = int(relative_date.split('시간 전')[0])
              current_date = datetime.today()
              modified_date = current_date - timedelta(hours=hours_ago)
              return modified_date.strftime('%Y.%m.%d.')  # Updated date format
          elif '분 전' in relative_date:
              minutes_ago = int(relative_date.split('분 전')[0])
              current_date = datetime.today()
              modified_date = current_date - timedelta(minutes=minutes_ago)
              return modified_date.strftime('%Y.%m.%d.')  # Updated date format
      return relative_date

  def naver(self, link):
      # 네이버 뉴스 크롤링 코드
      response = requests.get(link)
      response.encoding = 'utf-8'
      soup = BeautifulSoup(response.text, 'html.parser')

      article_link = []
      title_list = []
      article_list = []
      date_list = []

      news_links = soup.find_all('a', attrs={'class': 'news_tit'})
      for i in range(min(10, len(news_links))):
          article_link.append(news_links[i].get('href'))
          title_list.append(news_links[i].get('title'))
          art_temp = soup.find_all('a', attrs={'class': 'info press'})[i]
          if art_temp.find('i') is not None:
              art_temp.find('i').decompose()
          article_list.append(art_temp.text)

      date = soup.find_all('span', attrs={'class': 'info'})
      for i in range(min(20, len(date))):
          date_temp = soup.find_all('span', attrs={'class': 'info'})[i]
          word_to_remove = '면'
          if word_to_remove in date_temp.text:
              continue
          date_list.append(date_temp.text)

      return article_link, article_list, title_list, date_list

  def crawling(self):
      warnings.simplefilter(action='ignore', category=FutureWarning)
      max_page = 100
      df = pd.DataFrame(columns=['link', 'article', 'name', 'title', 'date'])

      print(f"{self.stock} 크롤링 시작 !@!")

      # Calculate the date one month ago from the current date
      one_month_ago = (datetime.now() - timedelta(hours=24)).date()

      for j in range(max_page):
          try:
              if j == 0:
                  url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + self.keyword + "&sort=1&photo=0&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&office_category=0&service_area=0&nso=so:dd,p:all,a:all&start=1"
              else:
                  url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + self.keyword + f"&sort=1&photo=0&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&office_category=0&service_area=0&nso=so:dd,p:all,a:all&start={j}1"

              naver_data = self.naver(url)

              if len(naver_data[0]) == 0 or len(naver_data[1]) == 0 or len(naver_data[2]) == 0 or len(naver_data[3]) == 0:
                  print(f"{self.stock} 크롤링 다시 해봐. 기사 0개야.")

              if len(naver_data[0]) < 10 or len(naver_data[1]) < 10 or len(naver_data[2]) < 10 or len(naver_data[3]) < 10:
                  for i in range(len(naver_data[0])):
                      article_date_str = naver_data[3][i]
                      article_date = self.convert_relative_date(article_date_str)
                      article_date = datetime.strptime(article_date, "%Y.%m.%d.").date()
                      if article_date < one_month_ago:
                          # If the article date is older than one month ago, break the loop
                          return df
                      df = df.append({
                          'link': naver_data[0][i],
                          'article': naver_data[1][i],
                          'name': self.stock,
                          'title': naver_data[2][i],
                          'date': article_date
                      }, ignore_index=True)
                  break

              for i in range(10):
                  article_date_str = naver_data[3][i]
                  article_date = self.convert_relative_date(article_date_str)
                  article_date = datetime.strptime(article_date, "%Y.%m.%d.").date()
                  if article_date < one_month_ago:
                      # If the article date is older than one month ago, break the loop
                      return df
                  df = df.append({
                      'link': naver_data[0][i],
                      'article': naver_data[1][i],
                      'name': self.stock,
                      'title': naver_data[2][i],
                      'date': article_date
                  }, ignore_index=True)

              time.sleep(3)

          except Exception as e:
              print(f"An error occurred on page {j + 1}: {e}")
              print("Retrying after 5 seconds...")
              time.sleep(5)

      print(f"{self.stock} 크롤링 완료 !@! 기사 개수 (중복 제거 전):", len(df))
      # Remove duplicates based on the 'link' column
      df.drop_duplicates(subset='link', keep='first', inplace=True)
      print(f"{self.stock} 크롤링 완료 !@! 기사 개수 (중복 제거 후):", len(df))
      return df

  # 신문사별 크롤링 코드 mapping
  def extract_title_content(self, row):
      if row['article'] == '머니투데이':
          title, content = self.mt(row)
      elif row['article'] == '매일경제':
          title, content = self.mk(row)
          if content == '요청하신 페이지를 찾을 수 없습니다.':
            title, content = row['title'], ''
      elif row['article']=='서울경제신문' or row['article'] == '서울경제':
        title, content = self.sedaily(row)
      elif row['article']=='더벨(thebell)' or row['article'] == '더벨':
        row['link'] = row['link'] + '&svccode=00&page=1&sort=thebell_check_time'
        row['link'] = row['link'].replace('/front/free/contents/news/article_view.asp', '/free/content/ArticleView.asp')
        title, content = self.thebell(row)
      elif row['article']=='한국경제':
        title, content = self.hankyung(row)
      elif row['article']=='이투데이':
        title,content = self.etoday(row)
      elif row['article']=='머니s' or row['article'] == '머니S':
        title, content = self.moneys(row)
      elif row['article']=='아시아경제':
        title, content = self.asiae(row)
      elif row['article']=='서울파이낸스':
        title, content = self.seoulfn(row)
      elif row['article']=='딜사이트':
        title, content = self.dealsite(row)
      elif row['article']=='비즈워치':
        title,content = self.bizwatch(row)
      elif row['article']=='이코노미스트':
        title, content = self.economist(row)
      elif row['article']=='라이센스뉴스':
        title, content = self.license_news(row)
      elif row['article']=='시사저널e':
        title, content = self.sisa(row)
      elif row['article']=='충청신문':
        title,content = self.chungcheong(row)
      elif row['article']=='조선비즈':
        title, content = self.chosun(row)
      elif row['article']=='전국매일신문':
        title, content = self.jkmail(row)
      elif row['article']=='Investing.com':
        title, content = self.inv(row)
      elif row['article']=='영남일보':
        if row['link'].startswith('//'):
          row['link'] = 'https:' + row  # Add https: at the beginning
        title,content = self.yn(row)
      elif row['article']=='CBC뉴스':
        title, content = self.cbc(row)
      elif row['article']=='블로터':
        title, content = self.blt(row)
      elif row['article']=='더팩트':
        title, content = self.thefact(row)
      elif row['article']=='아시아에이':
        title, content = self.asiaa(row)
      elif row['article']=='문화뉴스':
        title,content = self.culture(row)
      elif row['article']=='한겨레':
        title, content = self.hangyere(row)
      elif row['article']=='뉴스핌':
        title, content = self.pim(row)
      elif row['article']=='이데일리':
        title, content = self.e_daily(row)
      elif row['article']=='비즈니스포스트':
        title, content = self.biz_post(row)
      elif row['article']=='시사저널이코노미':
        title, content = self.sisa_e(row)
      elif row['article']=='파이낸셜뉴스':
        title, content = self.fn_news(row)
      elif row['article']=='국제뉴스':
        title, content = self.international_news(row)
      elif row['article']=='데일리안':
        title, content = self.daily_ahn(row)
      elif row['article']=='MTN':
        title, content = self.mtn(row)
      elif row['article']=='아주경제':
        title, content = self.aju_e(row)
      elif row['article']=='뉴스토마토':
        title, content = self.news_tomato(row)
      elif row['article']=='뉴스1':
        title, content = self.news1(row)
      elif row['article']=='한국경제TV':
        title, content = self.hk_tv(row)
        if content == '해당기사가 삭제되었거나 보유기간이 종료되었습니다. ':
          title, content = row['title'], ''
      else:
          title, content = row['title'], ''
      return title, content



  # 이 아래서부터는 신문사별 크롤링 메소드
  def mt(self, row):
      response = requests.get(row['link'])
      soup = BeautifulSoup(response.text, 'html.parser')
      title = soup.find('h1', {'class': 'subject'})
      title = title.text if title else ''

      text = soup.find('div', {'id': 'textBody'})
      text = text.text if text else ''

      return title, text

  def mk(self, row):
      response = requests.get(row['link'])
      response.encoding = 'utf-8'
      soup = BeautifulSoup(response.text, 'html.parser')
      title = soup.find('h2', {'class': 'news_ttl'})
      if not title:
          title_div = soup.find('div', {'class': 'news_title_text'})
          if title_div:
              title = title_div.find('h1', {'class': 'top_title'})
              title = title.text if title else ''
          else:
              title = ''

      content = soup.find('div', {'class': 'news_cnt_detail_wrap'})
      if not content:
          content = soup.find('div', {'itemprop': 'articleBody'})
          if not content:
              content = soup.find('h1', {'class': 'page_ttl f12_stit'})
              if content:
                  content = content.text
              else:
                  content = ''
          else:
              content = content.text
      else:
          content = content.text

      return title, content

  def sedaily(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('h1',{'class':'art_tit'})
    if title is None:
      title=soup.find('div',{'class':'sub_view'}).find('div').find('h2')
      if title is None:
        title = ''
      else:
        title = title.text
    else:
      title=title.text
    content=soup.find('div',{'class':'article_view'})
    if content is None:
      content=soup.find('div',{'itemprop':'articleBody'})
      if content is None:
        content = ''
      else:
        content = content.text
    else:
      content=content.text
    return title, content

  def thebell(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('p',{'class':'tit'})
    if title is None:
      title=''
    else:
      title=title.text
    content=soup.find('div',{'class':'viewSection'})
    if content is None:
      content=''
    else:
      content=content.text[60:]
    return title, content

  def hankyung(self, row):
    driver = webdriver.Chrome(options=options)
    driver.get(row['link'])
    driver.implicitly_wait(3)
    try:
      title = driver.find_element('class name', 'article-contents')
      if title is None:
        title=''
      else:
        title=title.find_element('class name', 'headline').text
    except:
      title = driver.find_element('class name', 'article-tit').text

    text = driver.find_element('class name', 'article-body').text
    return title, text

  def etoday(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('h1',{'class':'main_title'})
    if title is None:
      title=''
    else:
      title=title.text
    text=soup.find('div',{'class':'articleView'})
    content_list=[]
    if text is None:
      text=''
    else:
      text=text.find_all('p')
      for i in range(len(text)):
        content_list.append(text[i].text)
      text=" ".join(content_list) #본문
    return title, text

  def moneys(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('div',{'id':'article'})
    if title is None:
      title=soup.find('h1',{'class':'mgt37'}).text
    else:
      title=title.find('h1').text
    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content

  def asiae(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('h1',{'class':'mainTitle'})
    if title is None:
      # 예외
      title=soup.find('div', {'class':'area_title'})
      if title is None:
        title = ''
      else:
        title = title.find('h1').text
      # 예외 끝
    else:
      title=title.text
    text=soup.find('div',{'class':'va_cont'})
    content_list=[]
    if text is None:
      # 예외
      text=soup.find('div',{'itemprop':'articleBody'})
      if text is None:
        text=''
      else:
        text=text.find_all('p')
        content_list=[]
        for i in range(len(text)):
          content_list.append(text[i].text)
        text=" ".join(content_list) #본문
      # 예외 끝
    else:
      text=text.find_all('p')
      content_list=[]

      for i in range(len(text)):
        content_list.append(text[i].text)
      text=" ".join(content_list) #본문
    return title, text

  def seoulfn(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('div',{'class':'article-head-title'})
    if title is None:
      title=''
    else:
      title=title.text
    text=soup.find('div',{'id':'article-view-content-div'})
    content_list=[]
    if text is None:
      text=''
    else:
      text=text.find_all('p')

      for i in range(len(text)):
        content_list.append(text[i].text)
      text=" ".join(content_list) #본문
    return title,  text

  def dealsite(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('div',{'class':'read-news-title'})
    if title is None:
      title=''
    else:
      title=title.text
    text=soup.find('div',{'class':'rnmc-right rnmc-right1 content-area'})
    content_list=[]
    if text is None:
      text=''
    else:
      text=text.find_all('p')

      for i in range(len(text)):
        content_list.append(text[i].text)
      text=" ".join(content_list) #본문
    return title, text

  def bizwatch(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('div',{'class':'top_content'}).find('h1')
    if title is None:
      title=''
    else:
      title=title.text
    text=soup.find('div',{'itemprop':'articleBody'})
    content_list=[]
    if text is None:
      text=''
    else:
      text=text.find_all('p')
      for i in range(len(text)):
        content_list.append(text[i].text)
      text=" ".join(content_list) #본문
    return title, text

  def economist(self, row):
    driver = webdriver.Chrome(options=options)
    driver.get(row['link'])
    driver.implicitly_wait(3)
    title = driver.find_element('id', 'article_body')
    if title is None:
      title=''
    else:
      title=title.find_element('class name', 'view-article-title').text
    text = driver.find_element('class name', 'content').text
    return title, text

  def license_news(self, row):
    driver = webdriver.Chrome(options=options)
    driver.get(row['link'])
    driver.implicitly_wait(3)
    title = driver.find_element('class name', 'article-view-header')
    if title is None:
      title=''
    else:
      title=title.find_element('class name', 'heading').text
    text = driver.find_element('id', 'article-view-content-div').text
    return title, text

  def sisa(self, row):
    driver = webdriver.Chrome(options=options)
    driver.get(row['link'])
    driver.implicitly_wait(3)
    title = driver.find_element('class name', 'article-header-wrap')
    if title is None:
      title=''
    else:
      title=title.find_element('class name', 'article-head-title').text
    text = driver.find_element('id', 'article-view-content-div').text
    return title, text

  def chungcheong(self, row):
    driver = webdriver.Chrome(options=options)
    driver.get(row['link'])
    driver.implicitly_wait(3)
    title = driver.find_element('class name', 'article-view-header')
    if title is None:
      title=''
    else:
      title=title.find_element('class name', 'heading').text
    text = driver.find_element('id', 'article-view-content-div').text
    return title, text

  def chosun(self, row):
    driver = webdriver.Chrome(options=options)
    driver.get(row['link'])
    driver.implicitly_wait(3)
    title = driver.find_element('class name', 'article-header__headline')
    if title is None:
      title=''
    else:
      title=title.text
    text = driver.find_element('class name', 'article-body').text
    return title, text

  def jkmail(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('div',{'class':'article-header-wrap'})
    if title is None:
      title=''
    else:
      title=title.find('div', {'class':'article-head-title'}).text
    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def inv(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('h1',{'class':'articleHeader'})
    if title is None:
      title=''
    else:
      title=title.text
    content=soup.find('div',{'class':'WYSIWYG articlePage'})
    if content is None:
      content=''
    else:
      content=content.find_all('p')
      content_list=[]
      for i in range(len(content)):
        content_list.append(content[i].text)
      content=" ".join(content_list) #본문
    return title, content # title, content 완료

  def yn(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('p', {'class':'article-top-title'})
    if title is None:
      # 예외
      title=soup.find('h1', {'class':'article-top-title'})
      if title is None:
        title = ''
      else:
        title = title.text
    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def cbc(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('div',{'class':'article-header-wrap'})
    if title is None:
      title=''
    else:
      title=title.find('div', {'class':'article-head-title'}).text
    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.find_all('p')
      content_list=[]
      for i in range(len(content)):
        content_list.append(content[i].text)
      content=" ".join(content_list) #본문
    return title, content # title, content 완료

  def blt(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('header',{'class':'article-view-header'})
    if title is None:
      title=''
    else:
      title=title.find('h1', {'class':'heading'}).text
    content = soup.find('article',{'itemprop' : 'articleBody'})
    if content is None:
      content = ''
    else:
      content=content.text
    return title, content # title, content 완료

  def thefact(self, row):
    response = requests.get(row['link'])
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('h2',{'class':'articleTitle'})
    if title is None:
      title=soup.find('div',{'class':'articleTitle'})
      if title is None:
        title = ''
      else:
        title = title.text
    else:
      title=title.text
    content = soup.find('div',{'class' : 'mArticle'})
    if content is None:
      content = soup.find('div',{'itemprop' : 'articleBody'})
      if content is None:
        content = ''
      else:
        content = content.text
    else:
      content=content.find_all('p')
      content_list=[]
      for i in range(len(content)):
        content_list.append(content[i].text)
      content=" ".join(content_list) #본문
    return title, content # title, content 완료

  def asiaa(self, row):
    response = requests.get(row['link'])  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('h3',{'class':'heading'})
    if title is None:
      title=''
    else:
      title=title.text
    content = soup.find('article',{'id' : 'article-view-content-div'})
    if content is None:
      content=''
    else:
      content=content.find_all('p')
      content_list=[]
      for i in range(len(content)):
        content_list.append(content[i].text)
      content=" ".join(content_list) #본문
    return title, content # title, content 완료

  def culture(self, row):
    response = requests.get(row['link'])  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('h3',{'class':'heading'})
    if title is None:
      title=''
    else:
      title=title.text
    content = soup.find('div',{'style' : 'text-align:center'})
    if content is None:
      content = soup.find('div',{'class' : 'article-body'})
      if content is None:
        content = ''
      else:
        content = content.text
    else:
      content=content.find_all('p')
      content_list=[]
      for i in range(len(content)):
        content_list.append(content[i].text)
      content=" ".join(content_list) #본문
    return title, content # title, content 완료

  def hangyere(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('span',{'class':'title'})
    if title is None:
      title=''
    else:
      title=title.text

    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content

  def pim(self, row):
    response = requests.get(row['link'])  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')
    title=soup.find('h2',{'id':'main-title'})
    if title is None:
      title=''
    else:
      title=title.text
    content = soup.find('div',{'id':'news-contents'})
    if content is None:
      content=''
    else:
      content=content.find_all('p')
      content_list=[]
      for i in range(len(content)):
        content_list.append(content[i].text)
      content=" ".join(content_list) #본문
    return title, content # title, content 완료

  def e_daily(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('div',{'class':'news_titles'})
    if title is None:
      title=''
    else:
      title=title.find('h1').text

    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def biz_post(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('div',{'class':'detail_title'})
    if title is None:
      title=''
    else:
      title=title.find('h2').text

    content=soup.find('div',{'class':'detail_editor'})
    if content is None:
      content=soup.find('dd',{'class':'clearfix'})
      if content is None:
        content = ''
      else:
        content = content.text
    else:
      content=content.text
    return title, content # title, content 완료

  def sisa_e(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('div',{'class':'article-head-title'})
    if title is None:
      title=''
    else:
      title=title.find('strong').text

    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def fn_news(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('h1',{'class':'tit_view'})
    if title is None:
      title=''
    else:
      title=title.text

    content=soup.find('div',{'id':'article_content'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def international_news(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('h3',{'class':'heading'})
    if title is None:
      title=''
    else:
      title=title.text

    content=soup.find('article',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def daily_ahn(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('h1',{'class':'title'})
    if title is None:
      title=''
    else:
      title=title.text

    content_list = []
    content=soup.find('div',{'class':'article'})
    if content is None:
      content=''
    else:
      content=content.find_all('p')
      for i in range(len(content)):
        content_list.append(content[i].text)
      content=" ".join(content_list) #본문
    return title, content # title, content 완료

  def mtn(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('div',{'class':'news-header'})
    if title is None:
      title=''
    else:
      title=title.find('h1').text

    content_list = []
    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def aju_e(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('div',{'class':'inner'})
    if title is None:
      title=''
    else:
      title=title.find('li', {'class':'tit'}).text

    content_list = []
    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def news_tomato(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('div',{'class':'rn_stitle'})
    if title is None:
      title=''
    else:
      title=title.text

    content_list = []
    content=soup.find('div',{'class':'rns_text'})
    if content is None:
      content=''
    else:
      content=content.text
    return title, content # title, content 완료

  def news1(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('div',{'class':'title'})
    if title is None:
      title=soup.find('div',{'class':'photo_article'})
      if title is None:
        title = ''
      else:
        title = title.find('h3').text
    else:
      title=title.find('h2').text

    content=soup.find('div',{'itemprop':'articleBody'})
    if content is None:
      content=soup.find('p',{'itemprop':'articleBody'})
      if content is None:
        content = ''
      else:
        content=content.text
    else:
      content=content.text
    return title, content # title, content 완료

  def hk_tv(self, row):
    response = requests.get(row['link'])
    response.encoding = 'utf-8'  # 지정된 인코딩으로 설정
    soup=BeautifulSoup(response.text,'html.parser')

    title=soup.find('h1',{'class':'title-news'})
    if title is None:
      title=''
    else:
      title=title.text

    content=soup.find('div',{'class':'box-news-body'})
    if content is None:
      content = soup.find('p', {'class':'title-news'})
      if content is None:
        content = ''
      else:
        content = content.text
    else:
      content=content.text
    return title, content # title, content 완료