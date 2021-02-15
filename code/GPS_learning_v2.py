#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.initializers import glorot_normal
from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate, LSTM, Bidirectional, Masking, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.optimizers import Nadam, Adamax, Adam

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import time
import math
import os
import csv

def GPSestimation(columns_size):
    with tf.device('/gpu:0'):
        sensor_input = Input(shape=(20,columns_size), name='main_input')
        input_mask = Masking(mask_value=0)(sensor_input)
        
#        hidden1 = Conv1D(filters=128, kernel_size=21, strides=1, padding='valid', activation='relu', input_shape=(36,columns_size), name='conv1')(sensor_input)
#        hidden1 = MaxPooling1D(2)(hidden1)
#        hidden1 = Flatten(name='flatten')(hidden1)
        hidden1 = Bidirectional(LSTM(64), merge_mode= 'concat')(input_mask)
        
        y1 = Dense(512, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(hidden1)
        y1 = Dropout(0.5)(y1)
        y2 = Dense(512, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y1)
        y2 = Dropout(0.5)(y2)
        y3 = Dense(512, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y2)
        y3 = Dropout(0.5)(y3)
        y4 = Dense(512, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y3)
        y4 = Dropout(0.5)(y4)
        y5 = Dense(512, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y4)
        y5 = Dropout(0.5)(y5)
        y6 = Dense(512, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y5)
        y6 = Dropout(0.5)(y6)

        GPS_output = Dense(1, activation = 'sigmoid', kernel_initializer=glorot_normal(seed=1))(y6)

        GPS_model = Model(inputs = [sensor_input], outputs = [GPS_output])
        
    return(GPS_model)

def weightexport_LSTM(model, layer_ind):
    for i in range(len(layer_ind)):
        LSTM_weight_x = model.layers[layer_ind[i]].get_weights()[0]
        LSTM_weight_h = model.layers[layer_ind[i]].get_weights()[1]
        LSTM_bias = model.layers[layer_ind[i]].get_weights()[2]
        pd.DataFrame(LSTM_weight_x).to_csv('LSTM_weight_x'+str(layer_ind[i])+'.csv', header = False, index = False)
        pd.DataFrame(LSTM_weight_h).to_csv('LSTM_weight_h'+str(layer_ind[i])+'.csv', header = False, index = False)
        pd.DataFrame(LSTM_bias).to_csv('LSTM_bias'+str(layer_ind[i])+'.csv', header = False, index = False)

def weightexport_BiLSTM(model, layer_ind):
    for i in range(len(layer_ind)):
        LSTM_weight_x_forward = model.layers[layer_ind[i]].get_weights()[0]
        LSTM_weight_h_forward = model.layers[layer_ind[i]].get_weights()[1]
        LSTM_bias_forward = model.layers[layer_ind[i]].get_weights()[2]
        pd.DataFrame(LSTM_weight_x_forward).to_csv('LSTM_weight_x'+str(layer_ind[i])+'_forward'+'.csv', header = False, index = False)
        pd.DataFrame(LSTM_weight_h_forward).to_csv('LSTM_weight_h'+str(layer_ind[i])+'_forward'+'.csv', header = False, index = False)
        pd.DataFrame(LSTM_bias_forward).to_csv('LSTM_bias'+str(layer_ind[i])+'_forward'+'.csv', header = False, index = False)
        
        LSTM_weight_x_backward = model.layers[layer_ind[i]].get_weights()[3]
        LSTM_weight_h_backward = model.layers[layer_ind[i]].get_weights()[4]
        LSTM_bias_backward = model.layers[layer_ind[i]].get_weights()[5]
        pd.DataFrame(LSTM_weight_x_backward).to_csv('LSTM_weight_x'+str(layer_ind[i])+'_backward'+'.csv', header = False, index = False)
        pd.DataFrame(LSTM_weight_h_backward).to_csv('LSTM_weight_h'+str(layer_ind[i])+'_backward'+'.csv', header = False, index = False)
        pd.DataFrame(LSTM_bias_backward).to_csv('LSTM_bias'+str(layer_ind[i])+'_backward'+'.csv', header = False, index = False)
        
def weightexport_Dense(model, layer_ind):
    for i in range(len(layer_ind)):
        Dense_weight = model.layers[layer_ind[i]].get_weights()[0]
        Dense_bias = model.layers[layer_ind[i]].get_weights()[1]
        pd.DataFrame(Dense_weight).to_csv('Dense_weight'+str(layer_ind[i])+'.csv', header = False, index = False)
        pd.DataFrame(Dense_bias).to_csv('Dense_bias'+str(layer_ind[i])+'.csv', header = False, index = False)
        
def weightexport_Conv1D(model, layer_ind):
    for i in range(len(layer_ind)):
        Conv1D_weight = []
        iter_num = model.layers[layer_ind[i]].get_weights()[0].shape[2]
        for j in range(iter_num):
            if j==0:
                Conv1D_weight = model.layers[layer_ind[i]].get_weights()[0][:,:,0]
            else:
                Conv1D_weight = np.vstack((Conv1D_weight, model.layers[layer_ind[i]].get_weights()[0][:,:,j]))
            
        Conv1D_bias = model.layers[layer_ind[i]].get_weights()[1]
        pd.DataFrame(Conv1D_weight).to_csv('Conv1D_weight'+str(layer_ind[i])+'.csv', header = False, index = False)
        pd.DataFrame(Conv1D_bias).to_csv('Conv1D_bias'+str(layer_ind[i])+'.csv', header = False, index = False)        

