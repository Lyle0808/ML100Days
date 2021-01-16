#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:46:59 2021

@author: zhengyusong
"""

請將 app_train 中的 CNT_CHILDREN 依照下列規則分為四組，並將其結果在原本的 dataframe 命名為 CNT_CHILDREN_GROUP

0 個小孩
有 1 - 2 個小孩
有 3 - 5 個小孩
有超過 5 個小孩
請根據 CNT_CHILDREN_GROUP 以及 TARGET，列出各組的平均 AMT_INCOME_TOTAL，並繪製 baxplot

請根據 CNT_CHILDREN_GROUP 以及 TARGET，對 AMT_INCOME_TOTAL 計算 Z 轉換 後的分數
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
app_train=pd.read_csv("application_train.csv")
#%%
app_train['CNT_CHILDREN'].hist()
#%%條件
a=app_train[app_train["CNT_CHILDREN"]==0]
b=app_train[app_train["CNT_CHILDREN"]==1]
c=app_train[app_train["CNT_CHILDREN"]==2]
d=app_train[app_train["CNT_CHILDREN"]==3]
e=app_train[app_train["CNT_CHILDREN"]==4]
f=app_train[app_train["CNT_CHILDREN"]==5]
g=app_train[app_train["CNT_CHILDREN"]>5] 
#%%在第七個column插入新的特徵
a.insert(7,"CNT_CHILDREN_GROUP",0)
b.insert(7,"CNT_CHILDREN_GROUP",1)
c.insert(7,"CNT_CHILDREN_GROUP",1)
d.insert(7,"CNT_CHILDREN_GROUP",2)
e.insert(7,"CNT_CHILDREN_GROUP",2)
f.insert(7,"CNT_CHILDREN_GROUP",2)
g.insert(7,"CNT_CHILDREN_GROUP",3)

#%%把資料及由上而下合併
app_train = pd.concat([a, b, c,d,e,f,g])
#%%在'NAME_CONTRACT_TYPE'的各個條件'TARGET'的平均
app_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].mean()
app_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].hist()
#%%各個群組AMT_INCOME_TOTAL的平均
app_train.groupby(['CNT_CHILDREN_GROUP'])['AMT_INCOME_TOTAL'].mean()
app_train.groupby(['CNT_CHILDREN_GROUP'])['AMT_INCOME_TOTAL'].hist()
#%%列出個目標中AMT_INCOME_TOTAL的平均及畫圖
app_train.groupby(['TARGET'])['AMT_INCOME_TOTAL'].mean()
app_train.groupby(['TARGET'])['AMT_INCOME_TOTAL'].hist()
#%%把資料進行z轉換
value=app_train["AMT_INCOME_TOTAL"].values
app_train['AMT_INCOME_TOTAL_zscore']=(value-app_train[app_train["TARGET"]==1]["AMT_INCOME_TOTAL"].mean()/np.std(value))
app_train['AMT_INCOME_TOTAL_zscore'].hist(bins=30)
#%%
# 取前 10000 筆作範例: 分別將 AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY 除以根據 NAME_CONTRACT_TYPE 分組後的平均數，
app_train.loc[0:10000, ['NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']].groupby(['NAME_CONTRACT_TYPE']).apply(lambda x: x / x.mean())






















