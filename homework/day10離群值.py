#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:54:26 2021

@author: zhengyusong
"""
#%%
df_train=pd.read_csv("train.csv")
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
#%%
train_Y=np.log1p(df_train["SalePrice"])#為了讓資料更服從高斯分佈，有點類似把資料標準化
df=df_train.drop(["Id","SalePrice"],axis=1)
#%%只取int64 \ float64兩種數值型欄位，存於num_features中
num_features=[] #用list存
for dtypes,feature in zip(df.dtypes,df.columns): #dtypes是用來看資料的種類
    if dtypes == 'float64' or dtypes=='int64':
        num_features.append(feature)#append表示要追加的對象
#%%消減文字型欄位，只剩數值型
df= df[num_features]
df=df.fillna(-1)#用-1取代空值
MMEncoder=MinMaxScaler()#MMEncoder是為了把數據依比例調整到一個固定的範圍中,在這裡他把他設定在
                         #最大最小值中
#%%顯示 GrLivArea 與目標值的散佈圖
sns.regplot(x=df['GrLivArea'],y=train_Y)
#做線性迴歸觀察分數
train_X=MMEncoder.fit_transform(df)
estimator=LinearRegression()
cross_val_score(estimator,train_X,train_Y,cv=5).mean()#交叉驗證5次取平均分數越高越準確
#%%將 GrLivArea 限制在 800 到 2500 以內, 調整離群值
df['GrLivArea']=df['GrLivArea'].clip(800,2500)
sns.regplot(x=df['GrLivArea'],y=train_Y)
#做線性迴歸觀察分數
train_X=MMEncoder.fit_transform(df)
estimator=LinearRegression()
cross_val_score(estimator,train_X,train_Y,cv=5).mean()
#%%將 GrLivArea 限制在 800 到 2500 以內, 捨棄離群值
keep_indexs = (df['GrLivArea']> 800) & (df['GrLivArea']< 2500)
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]
sns.regplot(x = df['GrLivArea'], y=train_Y)
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()












