# IPO_HELPER
2023 미래에셋 빅데이터 페스티벌 생성형 ai부분 최우수 팀 프로젝트 깃헙입니다. 

저희는 공모주 청약을 도울 수 있는 공모주 뉴스 요약과 챗봇의 모델링과 실제 서비스 구현을 진행하였습니다. 

## 공모주 뉴스 요약 서비스 

원하는 공모주의 이름을 입력하면 해당 입력날짜로부터 한달 사이안의 기사를 실시간으로 크롤링하여 요약 기사를 반환합니다.

요약 모델로는 T5을 finetuning하여 구축하였습니다. 

최종적인 후처리를 거쳐 입력된 공모주에 대한 최신 3개의 요약본을 반환합니다.

모델링 개요는 다음과 같습니다. 
![](https://github.com/yunseo4401/IPO-HELPER/blob/main/image1.png)


## 공모주 챗봇 서비스
공모주 챗봇 서비스는 이용 목적 및 성능을 고려하여 두 가지 챗봇을 구성하였습니다. 
1. 단어 사전 챗봇

   모르는 단어를 명사형태로 입력하면 이에 대한 정의를 반환하여 줍니다.
   
   KoGPT- trinity를 finetuning하여 모델을 구축하였습니다.
   
   ![](https://github.com/yunseo4401/IPO-HELPER/blob/main/image3.png)
   
3. 질의 응답 챗봇

   공모주 청약 중 발생 할 수 있는 질문을 자유로운 형태로 입력하면 이에 대한 알맞은 답변을 반환하여 줍니다.
   
   encoder로 Bert를 사용한 poly-encoder을 fintuning하여 모델을 구축하였습니다.

   ![](https://github.com/yunseo4401/IPO-HELPER/blob/main/image2.png)


## 데이터

450개의 공모주에 대하여 네이버와 구글의 신문기사를 직접 크롤링하였습니다. 
기사는 청약일 기준 한달 내의 기사만 사용하였습니다. 
2023년 7월기준 최신 순으로 450개를 선정하였습니다. 
챗봇데이터는 지식인과 아하사이트에서 "공모주"키워드 검색 결과 질문과 답변 데이터를 직접 크롤링하였습니다. 
데이터 증강 후 최종 개수는 다음과 같습니다. 
|데이터 종류|개수|
|:-----:|:-----:|
|공모주 기사|[53,640]|
|공모주 질문데이터|[15,661]|
|공모주 답변데이터|[31,905]|


## 성능
요약모델과 챗봇모델의 rouge 스코어와 perplexity는 다음과 같습니다. 
|요약모델|score|
|:-----:|:-----:|
|rouge 1 mid|[0.6432]|
|rouge 2 mid|[0.5803]|
|rouge L mid|[0.6336]|
|rouge Lsum mid|[0.6418]|

|챗봇-poly encoder|score(epoch10)|
|:-----:|:-----:|
|R1|[0.8294]|
|R2|0.9183]|
|R5|[0.9899]|
|R10|[1.0]|

|챗봇-Kogpt|score(epoch4)|
|:-----:|:-----:|
|train_loss|[0.1197]|
|valid loss|0.8293]|
|perplexity|[2.2917]|

## 서비스 이용하기 

```
1. git clone --
2. pip install -r requirements.txt
3. cd C:/ -- (레포를 git clone 한 공간으로)
4. python server.py
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
