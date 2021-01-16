#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:46:01 2021

@author: zhengyusong
"""
新增一個欄位 customized_age_grp，把 age 分為 (0, 10], (10, 20], (20, 30], (30, 50], (50, 100] 這五組， 
'(' 表示不包含, ']' 表示包含
Hints: 執行 ??pd.cut()，了解提供其中 bins 這個參數的使用方式
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% 初始設定 Ages 的資料
ages=pd.DataFrame({"age":[18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})
#%%等寬劃分(按照相同寬度將資料分成幾分，但受異常值影響很大)
# 新增欄位 "equal_width_age", 對年齡做等寬劃分
ages["equal_width_age"] = pd.cut(ages["age"], 4)
# 觀察等寬劃分下, 每個種組距各出現幾次
ages["equal_width_age"].value_counts() # 每個 bin 的值的範圍大小都是一樣的
#%%等頻劃分（每份資料的個數相同）
# 新增欄位 "equal_freq_age", 對年齡做等頻劃分
ages["equal_freq_age"] = pd.qcut(ages["age"], 4)
# 觀察等頻劃分下, 每個種組距各出現幾次
ages["equal_freq_age"].value_counts() # 每個 bin 的資料筆數是一樣的
#%%新增一個欄位 customized_age_grp，把 age 分為 (0, 10], (10, 20], (20, 30], (30, 50], (50, 100] 這五組， '(' 表示不包含, ']' 表示包含
bins=[0,10,20,30,50,100]
ages["customized_age_grp"] = pd.cut(ages["age"], bins)