#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:17:18 2021

@author: zhengyusong
"""
#%%
import numpy as np
import pandas as pd
#%%
data_train=pd.read_csv("titanic_train.csv")
data_test=pd.read_csv("titanic_test.csv")
#%%
data_train.dtypes.value_counts()
data_train.select_dtypes(include="object").apply(pd.Series.nunique,axis=0)
#%%將資料組成為訓練/預測用
train_Y=data_train["Survived"]
ids=data_test["PassengerId"]
df_train=data_train.drop(["PassengerId","Survived"],axis=1)
df_test=data_test.drop(["PassengerId"],axis=1)
data=pd.concat([df_train,df_test])
#%%秀出資料欄位的類型與數量
dtype_df=data.dtypes.reset_index()
dtype_df.columns=["Count","Column Type"]
dtype_df=dtype_df.groupby("Column Type").aggregate("count").reset_index()
#%%確定只有三種類型後，分別將欄位名稱存在三個list中
int_features=[]
float_features=[]
object_features=[]
for dtype,feature in zip(data.dtypes,data.columns):
    if dtype == "float64":
        float_features.append(feature)
    elif dtype == "int64":
        int_features.append(feature)
    else:
        object_features.append(feature)
    
#%%列出三種特徵類型的三種方法
data[int_features].mean()
data[float_features].mean()
data[int_features].max()
data[float_features].max()
data[int_features].nunique()
data[float_features].nunique()
