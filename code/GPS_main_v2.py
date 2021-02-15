#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code")
sys.path

import os

from data_manipulation_MH import *
from GPS_learning_v2 import *
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
import pandas as pd

def dataconvert(filelist, datatype):
    data = []
    for i in range(len(filelist)):
        data_temp = csvread(filelist[i], datatype[i])
        for j in range(len(data_temp)):
            data.append(data_temp[j])
    data = chartofloatarray(data)
    return(data)

path = "D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\\train_test_data\\train_data"
os.chdir(path)
print(os.listdir())

train_filename = (['GPS'])
# test_filename = (['GPS'])

input_type = 'sensor'
output_type = 'GPS'

# training data
input_train = dataconvert([train_filename], ['training_'+input_type])
output_train = dataconvert([train_filename], ['training_'+output_type])

# test data: Sequential gesture
# input_test = dataconvert([test_filename], ['test_'+input_type])

# parsing with certain number
check_num = 10000.0
# for training data
input_trainset = parsingdata(input_train, check_num)
output_trainset = parsingdata(output_train, check_num)
output_trainset = listtoarray2(output_trainset) # model.fit의 입력으로 list 타입이 아닌 np.array 타입으로 넣어야 함.

# for test data
# input_testset = parsingdata(input_test, check_num) # fixed window size로 이미 구분되어 있기 때문에 zero padding 불필요
# aux_input_testset = auxiliary_input(input_testset)
# aux_input_testset = listtoarray2(aux_input_testset)
# input_testset = listtoarray3(input_testset) # parsingdata의 출력은 list형태이나 model의 입력은 array여야 하므로 변환해줌.

# Zero-padding
input_trainset = sequence.pad_sequences(input_trainset, dtype='float32', padding='pre')

# sequence.pad_sequences는 list 타입의 입력 데이터를 array 타입으로 바꿔서 출력해줌.
# 학습 데이터는 길이가 다양한 시퀀스로 구성됨.
# model.fit의 입력인 inputs는 array 타입의 데이터를 허용하는데, 다양한 길이의 시퀀스로는 3차원 array를 만들 수 없음.
# 따라서 다양한 길이의 input sequence를 사용할 때 zero-padding은 필수임.

# padding 전의 input_trainset의 데이터의 형태를 표현하면 아래의 예시와 같음.
# [[sensor1.t0 sensor2.t0 ... sensor10.t0;
#  sensor1.t1 sensor2.t1 ... sensor10.t1;
#  ...
#  sensor1.t30 sensor2.t30 ... sensor10.t30]_sample1

# [sensor1.t0 sensor2.t0 ... sensor10.t0;
#  sensor1.t1 sensor2.t1 ... sensor10.t1;
#  ...
#  sensor1.t17 sensor2.t17 ... sensor10.t17]_sample2
#
# ...
# [sensor1.t0 sensor2.t0 ... sensor10.t0;
#  sensor1.t1 sensor2.t1 ... sensor10.t1;
#  ...
#  sensor1.t25 sensor2.t25 ... sensor10.t25]_sample26400]

GPS_model = GPSestimation(10) # model의 입력은 사용된 센서 개수를 의미함.
GPS_model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])
GPS_model.summary()

# model compile 후 실제로 학습되는 과정: model.fit
GPS_model.fit(input_trainset, output_trainset, epochs=10, batch_size=2640)

# 학습이 끝난 model에 test 데이터를 입력하여 출력값을 저장
# predictions = GPS_model.predict({'main_input': input_testset, 'last_step_input': aux_input_testset})

# model 저장, h5 파일
# os.chdir("D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\model")
# GPS_model.save('GPS_model.h5')

# os.chdir("D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code")
# pd.DataFrame(predictions).to_csv('predictions.csv', header=False, index=False)

# Export weight & bias
os.chdir("D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\model_reconstruction\weights_from_python\GPS_model")
weightexport_Conv1D(GPS_model, [1])
weightexport_Dense(GPS_model, [4,5,6,7,8,9,10])


# In[2]:


prediction = GPS_model.predict(input_trainset)


# In[24]:


prediction[29]


# In[9]:


import keras

layer_name = 'flatten'
intermediate_layer_model = keras.Model(inputs=GPS_model.input, outputs=GPS_model.get_layer(layer_name).output)
inter_output = intermediate_layer_model.predict(input_trainset)


# In[10]:


inter_output[0]


# In[11]:


inter_output[0].shape


# In[ ]:




