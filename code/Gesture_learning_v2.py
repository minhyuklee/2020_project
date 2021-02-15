#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.initializers import glorot_normal
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, concatenate, Multiply, Add, LSTM, Lambda, Dropout
from keras.optimizers import Nadam, Adamax, Adam

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import time
import math
import os
import csv

def Gesture_classification(columns_size):
    with tf.device('/gpu:0'):
        sensor_input = Input(shape=(None,columns_size))
        
        hidden1 = LSTM(128)(sensor_input)
        
#        hidden1 = LSTM(64, return_sequences=True)(sensor_input)
#        hidden2 = LSTM(64)(hidden1)
        
        y1 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(hidden1)
#        y1 = Dropout(0.3)(y1)
        y2 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y1)
#        y2 = Dropout(0.3)(y2)
        y3 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y2)

        Gesture_output = Dense(17, activation = 'softmax', kernel_initializer=glorot_normal(seed=1))(y3)

        Gesture_model = Model(inputs = sensor_input, outputs = Gesture_output)
        
    return(Gesture_model)

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

