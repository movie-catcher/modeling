#### ๋ชจ๋ธ๋ง

- [๐ Model Selection](#-model-selection)
- [๐งฉ Modeling](#-modeling)
  - [Model 1](#model-1)
    - [๋ณ์ ์ค์ ](#๋ณ์-์ค์ )
    - [๋ชจ๋ธ๋ง](#๋ชจ๋ธ๋ง-1)
  - [Model 2](#model-2)
    - [๋ณ์ ์ค์ ](#๋ณ์-์ค์ -1)
    - [๋ชจ๋ธ๋ง ์งํ](#๋ชจ๋ธ๋ง-์งํ)
- [๐ก Netflix ์ปจํ์ธ ](#-netflix-์ปจํ์ธ )
  - [Viral Index ์ ์  ๋ฐฉ๋ฒ](#viral-index-์ ์ -๋ฐฉ๋ฒ)

* * * 

## โ Stacks
![Python](https://img.shields.io/badge/-Python-306998?logo=python&logoColor=ffd43b&style=for-the-badge)
![Jupyter Notebook](https://img.shields.io/badge/-jupyter%20notebook-727272?logo=jupyter&logoColor=eb7633&style=for-the-badge)
![numpy](https://img.shields.io/badge/-numpy-ffd43b?logo=numpy&logoColor=306998&style=for-the-badge)
![pandas](https://img.shields.io/badge/-pandas-150454?logo=pandas&logoColor=ffffff&style=for-the-badge)
![seaborn](https://img.shields.io/badge/-seaborn-454571?logo=seaborn&logoColor=ffffff&style=for-the-badge)
![matplotlib](https://img.shields.io/badge/-matplotlib-125277?logo=matplotlib&logoColor=ffffff&style=for-the-badge)
![sklearn](https://img.shields.io/badge/-sklearn-125277?logo=sklearn&logoColor=ffffff&style=for-the-badge)
![xgboost](https://img.shields.io/badge/-xgboost-125277?logo=xgboost&logoColor=ffffff&style=for-the-badge)
![lightgbm](https://img.shields.io/badge/-lightgbm-125277?logo=lightgbm&logoColor=ffffff&style=for-the-badge)


# ๐ Model Selection

|name|desc|MSE|RMSE|
|---|---|---:|---:|
|Elastic|Ridge, Lasso ํฉ์น ๋ชจ๋ธ|0.43|0.66|
|โ **RandomForest**|ํธ๋ฆฌ๋ชจํ|0.42|0.65|
|LightGBM|ํธ๋ฆฌ๋ชจํ|0.47|0.69|
|xgboost|ํธ๋ฆฌ๋ชจํ|0.45|0.68|


<br/>

# ๐งฉ Modeling

> Model1 -> error -> Model2 -> Viral index

* ์ต๋ํ **์์๋ฌธ**์๋ง ์ํฅ์ ์ค ์ ์๋ ํผ์ณ๋ค๋ก Viral Index๋ฅผ ๊ตฌ์ํ๊ธฐ ์ํด์ **์์ฐจ**๋ฅผ ์ด์ฉ
* ์ฒซ ๋ฒ์งธ ๋ชจ๋ธ์ ์์ฐจ๋ฅผ ์ฌ์ฉํ์ฌ ๋ ๋ฒ์งธ ๋ชจ๋ธ๋ง์ ์งํํ์๊ธฐ ๋๋ฌธ์,   
    ์ํ์ ์ง์ ์ ์ผ๋ก ๊ด๋ จ๋ ํผ์ณ๋ค์ ์ํฅ๋ ฅ์ ์ ๊ฑฐ๋ ํ ๋ ๋ฒ์งธ ๋ชจ๋ธ์์ ์์ธก์ ์งํ
## Model 1

1. movie feature๋ก ๋ชจ๋ธ๋ง
2. ์์ธก๊ฐ(`y_hat`) ์ฐ์ถ

### ๋ณ์ ์ค์ 
- ์ข์๋ณ์ : ๊ด๋๊ฐ ์
- ๋๋ฆฝ๋ณ์ : ๊ตญ๊ฐ, ๋ฐฐ๊ธ์ฌ, ์ฅ๋ฅด, ์์์๊ฐ, ์ฐ๋ น์ ํ

### ๋ชจ๋ธ๋ง
1. ์ฌ์ฉX ํผ์ณ ์ ๊ฑฐ
2. one-hat encoding์ผ๋ก ์นดํ๊ณ ๋ฆฌํ ๋ฐ์ดํฐ ์์ 
3. ์์นํ ์๋ฃ์ scale ์์ ํ๋ฅผ ์ํ ์ ๊ทํ
4. ๋ชจ๋ธ ์ ํ๋๋ฅผ ์ํด Trainset๊ณผ testset์ 8:2 ๋ก ๋๋๊ธฐ
4. Randomforest ๋ชจ๋ธ์ ์ฑ๋ฅ์ ์ํด random cross validation์ผ๋ก ํ๋ผ๋ฏธํฐ ์์ 
5. MSE์ RMSE ํ์ธ   

      |MSE|RMSE|
      |---|---|
      |0.84|0.92|
6. ๋ชจ๋ธ fit, prediction์ ํ ๋ train, test๋ฅผ ๋๋์ง ์๊ณ  ์ถ์  ํ ์์ฐจ ๊ตฌํ๊ธฐ



## Model 2
1. Model1์์ ๊ตฌํ ์์ธก๊ฐ์ผ๋ก ์์ฐจ(`y`-`y_hat`) ๊ณ์ฐ
2. ์์ฐจ๋ฅผ ์ข์๋ณ์๋ก ์ฌ์ฉํ์ฌ viral feature๋ก ๋ชจ๋ธ๋ง
3. ์ฐ์ถํ ๊ฐ์ scalingํ์ฌ Viral Index ์ฐ์ถ

### ๋ณ์ ์ค์ 
- ์ข์๋ณ์ : y_hat = ๊ด๊ฐ์ โ model 1์ ์์ธก๊ฐ
- ๋๋ฆฝ๋ณ์ : Log ์กฐํ์ ํ๊ท , Sqrt ๋๊ธ ์, Sqrt ๊ฐ์ฑ๋ถ์ ์ ์, Sqrt๋ฐฐ์ฐ ์ํฅ๋ ฅ


### ๋ชจ๋ธ๋ง ์งํ
1. Transformation ํผ์ณ ์ฌ์ฉ(Log/Sqrt ๋ณํ ๋ฐ์ดํฐ)
2. ์์นํ ๋ฐ์ดํฐ ๋จ์ ์ฐจ์ด๋ฅผ ์ค์ด๊ธฐ ์ํ ์ ๊ทํ
3. Model 2์ MSE์ RMSE

      |MSE|RMSE|
      |---|---|
      |0.37|0.61|
4. Randomforest ๋ชจ๋ธ์ ์ฑ๋ฅ์ ์ํด random cross validation์ผ๋ก ํ๋ผ๋ฏธํฐ ์์ 

<br/>

# ๐ก Netflix ์ปจํ์ธ  
2021๋ ๊ฐ์ฅ ํฅํํ๋ '์ค์ง์ด ๊ฒ์'์ ๋ฐ์ด๋ด ์ง์๋ฅผ 100์  ๊ธฐ์ค์ผ๋ก ์ผ์ ๋ค๋ฅธ OTT ์ปจํ์ธ ์ ์ ์๋ฅผ ์ฐ์ 

## Viral Index ์ ์  ๋ฐฉ๋ฒ
1. ์ต์๊ฐ 0์ , ์ต๋๊ฐ 100์ ์ด ๋๋๋ก ์ค์ผ์ผ๋งํ๋ ๋ฐฉ๋ฒ
2. ํ๊ท ์ด 0 ๋ถ์ฐ์ด 1๋๋๋กํด์ ์ ๊ทํ๋ฅผ ์งํํ๊ณ  100์ ๊ณฑํ๋ ๋ฐฉ๋ฒ
3. โ **์ค์ง์ด๊ฒ์์ ์์ธก๊ฐ์ด ๋งค์ฐ ํฌ๋ฏ๋ก ์ค์ง์ด ๊ฒ์์ 100์  ๊ธฐ์ค์ผ๋ก ์ก๊ณ  ์ค์ผ์ผ๋ง์ ํ๋ ๋ฐฉ๋ฒ**


> Conclusion


|์์|OTT ์ปจํ์ธ  ์ด๋ฆ|Viral Index|
|:---:|:---:|---:|
|1|์ค์ง์ด ๊ฒ์|100.00|
|2|๋ณด๊ฑด๊ต์ฌ ์์์|50.32|
|3|ํน๋ค|50.22|
|4|์น๋ฆฌํธ|37.93|
|5|์ฅ์|11.94|

