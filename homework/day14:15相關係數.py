#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:38:25 2021

@author: zhengyusong
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
app_train=pd.read_csv("application_train.csv")
#%% 隨機生成兩組 1000 個介於 0~50 的數的整數 x, y, 看看相關矩陣如何
x=np.random.randint(0,50,1000)
y=np.random.randint(0,50,1000)
#%% 呼叫 numpy 裡的相關矩陣函數 (corrcoef)
np.corrcoef(x, y)
#%%畫出散佈圖
plt.scatter(x, y)
#%%
# 隨機生成 1000 個介於 0~50 的數 x
x = np.random.randint(0, 50, 1000)
# 這次讓 y 與 x 正相關，再增加一些雜訊
y = x + np.random.normal(0, 10, 1000)
# 再次用 numpy 裡的函數來計算相關係數
np.corrcoef(x, y)
#%%畫出散佈圖
plt.scatter(x, y)
#%%
x = np.random.randint(0, 50, 1000)
# 這次讓 y 與 x 負相關，再增加一些雜訊
y =  np.random.normal(0, 10, 1000) - x
# 再次用 numpy 裡的函數來計算相關係數
np.corrcoef(x, y)
#%%
plt.scatter(x, y)
#%%
app_train["DAYS_EMPLOYED"]
#%% 由於其他天數都是負值, 且聘僱日數不太可能是 365243 (大約 1000年), 算是異常數字 
# 因此我們推斷這份資料中, DAYS_EMPLOYED 的欄位如果是 365243, 應該是對應到空缺值, 繪圖時應該予以忽略
sub_df = app_train[app_train['DAYS_EMPLOYED'] != 365243]
#%%如果直接畫散布圖 - 看不出任何趨勢或形態
plt.plot(sub_df['DAYS_EMPLOYED'] / (-365), sub_df['AMT_INCOME_TOTAL'], '.')
plt.xlabel('Days of employed (year)')
plt.ylabel('AMT_INCOME_TOTAL (raw)')
corr = np.corrcoef(sub_df['DAYS_EMPLOYED'] / (-365), sub_df['AMT_INCOME_TOTAL'])
#%%# 通常可以對數值範圍較大的取 log: 發現雖然沒有相關，但是受雇越久的人，AMT_INCOME_TOTAL 的 variance 越小
plt.plot(sub_df['DAYS_EMPLOYED'] / (-365), np.log10(sub_df['AMT_INCOME_TOTAL'] ), '.')
plt.xlabel('Days of employed (year)')
plt.ylabel('AMT_INCOME_TOTAL (log-scale)')
corr_log = np.corrcoef(sub_df['DAYS_EMPLOYED'] / (-365), np.log10(sub_df['AMT_INCOME_TOTAL']))
#%%
# 將只有兩種值的類別型欄位, 做 Label Encoder, 計算相關係數時讓這些欄位可以被包含在內
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# 檢查每一個 column
for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder, 以加入相關係數檢查
            app_train[col] = le.fit_transform(app_train[col])            
print(app_train.shape)
app_train.head()
#%%
# 受雇日數為異常值的資料, 另外設一個欄位記錄, 並將異常的日數轉成空值 (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# 出生日數 (DAYS_BIRTH) 取絕對值 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
#%% 觀察相關係數(觀察所有資料與TARGET之間的關係)
corr_all=app_train.corr()['TARGET']
#%%畫出"EXT_SOURCE_3"與TARGET之間的關係（發現無相關）用散步圖不好看出
plt.scatter(app_train["EXT_SOURCE_3"], app_train["TARGET"])
#%%改用廂型圖就較為顯而易見
sns.boxplot(app_train["TARGET"],app_train["EXT_SOURCE_3"])
#%%由小排到大
corr_all=sorted(corr_all, reverse=False)

















