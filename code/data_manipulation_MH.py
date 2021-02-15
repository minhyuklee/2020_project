#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xlsxwriter
import xlrd
import csv
import numpy as np
import pandas as pd
import os
import math

array_data_size = 32

def csvread(filelist, filetype):
    data = []
    filename_history = []
    for i in range(0, len(filelist)):
        filename = filelist[i]
        filename_history.append(filename)
        data_parsing = read_csv(filename+filetype+'.csv')
        data.append(data_parsing)
    return(data)

def chartofloatarray(data):
    data_out = []
    for i in range(0, len(data)):
        data_temp = data[i]
        data_out_temp = np.zeros((len(data_temp), len(data_temp[0])))
        data_out_temp[:, :] = data_temp[:, :]
        data_out.append(data_out_temp)
    return(data_out)

def read_csv(filename):
    with open(filename, newline='', encoding = 'utf-8') as csvfile:
        data_read = list(csv.reader(csvfile))

    data_temp = listtoarray(data_read)
    data_out = data_temp

    return(data_out)

def listtoarray(list_data): # csv 파일의 list 타입을 array로 바꿀 때 사용 (입력과 출력 모두 2D 행렬의 형태)
    row_size = len(list_data)
    col_size = 0
    for line in list_data:
        if len(line) > col_size:
            col_size = len(line)

    array_data = np.chararray((row_size, col_size), itemsize = array_data_size, unicode = True)

    for row_count, line in enumerate(list_data):
        for col_count, data in enumerate(line):
            array_data[row_count, col_count] = data

    # print(row_size, col_size)
    return(array_data)

def listtoarray2(data): # list의 각 element의 행이 1일 때 사용. Vertical 방향으로 stack된 형태의 array 출력됨.
    if len(data) != 1:
        for count in range(len(data)):
            if count == 0:
                data_out = data[count]
            else:
                data_out = np.vstack((data_out, data[count]))
    else:
        data_out = data[0]
    return(data_out)

def listtoarray3(data): # list 내 각 element가 (timestep,features) 형태의 sequence인 경우 사용. 출력은 (sample,timestep,feature) 형태 가짐.
    timestep = data[0].shape[0]
    feature = data[0].shape[1]
    data_output = np.zeros((len(data), timestep, feature))
    for i in range(len(data)):
        tmp = data[i]
        for j in range(timestep):
            data_output[i,j,:] = tmp[j,:]
    
    return(data_output)
    

def parsingdata(data, check_num):
    temp_raw = np.array(data)
    data_out = []
    temp = np.zeros((1,temp_raw[0,0,:].shape[0]))
    for i in range(temp_raw.shape[1]):
        if np.float64(temp_raw[0,i,0]).item() == check_num:
            temp_comp = np.delete(temp, 0, 0)
            data_out.append(temp_comp)
            temp = np.zeros((1,temp_raw[0,0,:].shape[0]))
            continue
        else:
            temp2 = np.reshape(temp_raw[0,i,:],(1,-1))
            temp = np.append(temp,temp2,axis=0)
            
    return(data_out)

def auxiliary_input(list_trainset):
    output = [] # list of last step input
    for i in range(len(list_trainset)):
        temp = list_trainset[i]
        output.append(temp[-1,:])
        
    return(output)


# In[ ]:




