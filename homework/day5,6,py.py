#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:08:24 2021

@author: zhengyusong
"""
#%%套件
import os
import numpy as np
import pandas as pd
#%%套件
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#%%讀取資料
data_test=pd.read_csv("application_test.csv")
data_train=pd.read_csv("application_train.csv")
#%%找資料切割資料
data_test.head()
data_test.info()
data_test.shape[1]
data_test.shape[0]
data_test.iloc[1:5]
data_test["SK_ID_CURR"]
#%%檢視個欄位數量（可以知道類別型數量有哪些）
data_train.dtypes.value_counts()
#%%檢視資料欄位中各自類別型數量（可以知道類別型資料中有幾個資料（不重複））
data_train.select_dtypes(include=["object"]).apply(pd.Series.nunique,axis=0)
#%%label encoder
le=LabelEncoder()
le_count=0

for col in data_train:
    if data_train[col].dtype=="object":
        if len(list(data_train[col].unique()))<=2:
            le.fit(data_train[col])
            data_train[col]=le.transform(data_train[col])
            le_count=le_count+1
#%%onehot encoder
data_train=pd.get_dummies(data_train)
