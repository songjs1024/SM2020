# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:28:36 2020

@author: junes
"""


import numpy as np
import pandas as pd
from tensorflow import keras

data_path = "EvaluationInfo.txt"
info = pd.read_csv(data_path, header = None)
raw_data = pd.read_csv(info.at[0,0].replace("\\","/"), header = None)
raw_data.columns = ['value']
cols = int(info.at[1,0])#x칼럼
rows = int(info.at[2,0])#x로우
#x 데이터 셋 구성
tempt_list = []
for i in range(rows):
    tempt_list.append(list(raw_data['value'][i*cols:(i+1)*cols]))
    
x = np.array(tempt_list)

model = keras.models.load_model(info.at[4,0])

y_predict = model.predict(x)

result = pd.DataFrame(y_predict)

result.to_csv(info.at[3,0],sep = '\n', index=False, header = False)