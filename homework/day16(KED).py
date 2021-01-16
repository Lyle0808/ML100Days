#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:12:55 2021

@author: zhengyusong
"""
自 20 到 70 歲，切 11 個點，進行分群比較 (KDE plot)
以年齡區間為 x, target 為 y 繪製 barplot
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
#%%
app_train=pd.read_csv("application_train.csv")
#%%看資料中各類型資料有幾筆
int_features=[]
float_features=[]
object_feature=[]
for dtypes,feature in zip(app_train.dtypes,app_train.columns):
    if dtypes=="float64":
        float_features.append(feature)
    elif dtypes == "int64":
        int_features.append(feature)
    else:
        object_feature.append(feature)
#%%資料整理 ( 'DAYS_BIRTH'全部取絕對值 )
app_train["DAYS_BIRTH"]=abs(app_train["DAYS_BIRTH"])
#%%# 以生存年數繪製分布圖
plt.hist(app_train["DAYS_BIRTH"]/365,edgecolor="W",bins=25)
# 改變繪圖樣式 (style)
plt.style.use('ggplot') 
# 改變樣式後再繪圖一次, 比較效果
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'W', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
plt.show()
#%% 設定繪圖區域的長與寬
plt.figure(figsize = (10, 8))
# Kenel Density Estimation (KDE) plot: 會準時還貸者 (下圖紅線)
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')
# KDE plot: 不會準時還貸者 (下圖藍線)
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
# 設定標題與 X, y 軸的說明
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
#%%# KDE, 比較不同的 kernel function
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Gaussian esti.', kernel='gau')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Cosine esti.', kernel='cos')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Triangular esti.', kernel='tri')
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
#%%完整分布圖 (distplot) : 將 bar 與 Kde 同時呈現
sns.distplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
plt.legend()#右上角的圖示
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
#%%






















