#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.initializers import glorot_normal
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, concatenate, Multiply, Add, LSTM, Lambda
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
        sensor_input = Input(shape=(None,columns_size), name='main_input')
        hidden1 = LSTM(128, return_sequences=True)(sensor_input)
        hidden2 = LSTM(128)(hidden1)
        
        auxiliary_sensor_input = Input(shape=(columns_size,), name='last_step_input')
        
        r = concatenate([hidden2, auxiliary_sensor_input])
        
        y1 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(r)
        y2 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y1)
        y3 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y2)
        y4 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y3)
        y5 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y4)
        y6 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y5)

        GPS_output = Dense(1, activation = 'sigmoid', kernel_initializer=glorot_normal(seed=1))(y6)

        GPS_model = Model(inputs = [sensor_input, auxiliary_sensor_input], outputs = [GPS_output])
        
    return(GPS_model)

def weightexport_LSTM(model, layer_ind):
    for i in range(len(layer_ind)):
        LSTM_weight_x = model.layers[layer_ind[i]].get_weights()[0]
        LSTM_weight_h = model.layers[layer_ind[i]].get_weights()[1]
        LSTM_bias = model.layers[layer_ind[i]].get_weights()[2]
        pd.DataFrame(LSTM_weight_x).to_csv('LSTM_weight_x'+str(layer_ind[i])+'.csv', header = False, index = False)
        pd.DataFrame(LSTM_weight_h).to_csv('LSTM_weight_h'+str(layer_ind[i])+'.csv', header = False, index = False)
        pd.DataFrame(LSTM_bias).to_csv('LSTM_bias'+str(layer_ind[i])+'.csv', header = False, index = False)
        
def weightexport_Dense(model, layer_ind):
    for i in range(len(layer_ind)):
        Dense_weight = model.layers[layer_ind[i]].get_weights()[0]
        Dense_bias = model.layers[layer_ind[i]].get_weights()[1]
        pd.DataFrame(Dense_weight).to_csv('Dense_weight'+str(layer_ind[i])+'.csv', header = False, index = False)
        pd.DataFrame(Dense_bias).to_csv('Dense_bias'+str(layer_ind[i])+'.csv', header = False, index = False)

