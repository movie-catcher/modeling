#### 모델링

- [👀 Model Selection](#-model-selection)
- [🧩 Modeling](#-modeling)
  - [Model 1](#model-1)
    - [변수 설정](#변수-설정)
    - [모델링](#모델링-1)
  - [Model 2](#model-2)
    - [변수 설정](#변수-설정-1)
    - [모델링 진행](#모델링-진행)
- [💡 Netflix 컨텐츠](#-netflix-컨텐츠)
  - [Viral Index 선정 방법](#viral-index-선정-방법)

* * * 

## ⚒ Stacks
![Python](https://img.shields.io/badge/-Python-306998?logo=python&logoColor=ffd43b&style=for-the-badge)
![Jupyter Notebook](https://img.shields.io/badge/-jupyter%20notebook-727272?logo=jupyter&logoColor=eb7633&style=for-the-badge)
![numpy](https://img.shields.io/badge/-numpy-ffd43b?logo=numpy&logoColor=306998&style=for-the-badge)
![pandas](https://img.shields.io/badge/-pandas-150454?logo=pandas&logoColor=ffffff&style=for-the-badge)
![seaborn](https://img.shields.io/badge/-seaborn-454571?logo=seaborn&logoColor=ffffff&style=for-the-badge)
![matplotlib](https://img.shields.io/badge/-matplotlib-125277?logo=matplotlib&logoColor=ffffff&style=for-the-badge)
![sklearn](https://img.shields.io/badge/-sklearn-125277?logo=sklearn&logoColor=ffffff&style=for-the-badge)
![xgboost](https://img.shields.io/badge/-xgboost-125277?logo=xgboost&logoColor=ffffff&style=for-the-badge)
![lightgbm](https://img.shields.io/badge/-lightgbm-125277?logo=lightgbm&logoColor=ffffff&style=for-the-badge)


# 👀 Model Selection

|name|desc|MSE|RMSE|
|---|---|---:|---:|
|Elastic|Ridge, Lasso 합친 모델|0.43|0.66|
|✅ **RandomForest**|트리모형|0.42|0.65|
|LightGBM|트리모형|0.47|0.69|
|xgboost|트리모형|0.45|0.68|


<br/>

# 🧩 Modeling

> Model1 -> error -> Model2 -> Viral index

* 최대한 **입소문**에만 영향을 줄 수 있는 피쳐들로 Viral Index를 구상하기 위해서 **잔차**를 이용
* 첫 번째 모델의 잔차를 사용하여 두 번째 모델링을 진행하였기 때문에,   
    영화와 직접적으로 관련된 피쳐들의 영향력은 제거된 후 두 번째 모델에서 예측을 진행
## Model 1

1. movie feature로 모델링
2. 예측값(`y_hat`) 산출

### 변수 설정
- 종속변수 : 관람객 수
- 독립변수 : 국가, 배급사, 장르, 상영시간, 연령제한

### 모델링
1. 사용X 피쳐 제거
2. one-hat encoding으로 카테고리형 데이터 수정
3. 수치형 자료의 scale 안정화를 위한 정규화
4. 모델 정확도를 위해 Trainset과 testset을 8:2 로 나누기
4. Randomforest 모델의 성능을 위해 random cross validation으로 파라미터 수정
5. MSE와 RMSE 확인   

      |MSE|RMSE|
      |---|---|
      |0.84|0.92|
6. 모델 fit, prediction을 할때 train, test를 나누지 않고 추정 후 잔차 구하기



## Model 2
1. Model1에서 구한 예측값으로 잔차(`y`-`y_hat`) 계산
2. 잔차를 종속변수로 사용하여 viral feature로 모델링
3. 산출한 값을 scaling하여 Viral Index 산출

### 변수 설정
- 종속변수 : y_hat = 관객수 – model 1의 예측값
- 독립변수 : Log 조회수 평균, Sqrt 댓글 수, Sqrt 감성분석 점수, Sqrt배우 영향력


### 모델링 진행
1. Transformation 피쳐 사용(Log/Sqrt 변환 데이터)
2. 수치형 데이터 단위 차이를 줄이기 위한 정규화
3. Model 2의 MSE와 RMSE

      |MSE|RMSE|
      |---|---|
      |0.37|0.61|
4. Randomforest 모델의 성능을 위해 random cross validation으로 파라미터 수정

<br/>

# 💡 Netflix 컨텐츠 
2021년 가장 흥행했던 '오징어 게임'의 바이럴 지수를 100점 기준으로 삼아 다른 OTT 컨텐츠의 점수를 산정

## Viral Index 선정 방법
1. 최소가 0점, 최대가 100점이 되도록 스케일링하는 방법
2. 평균이 0 분산이 1되도록해서 정규화를 진행하고 100을 곱하는 방법
3. ✅ **오징어게임의 예측값이 매우 크므로 오징어 게임을 100점 기준으로 잡고 스케일링을 하는 방법**


> Conclusion


|순위|OTT 컨텐츠 이름|Viral Index|
|:---:|:---:|---:|
|1|오징어 게임|100.00|
|2|보건교사 안은영|50.32|
|3|킹덤|50.22|
|4|승리호|37.93|
|5|옥자|11.94|

