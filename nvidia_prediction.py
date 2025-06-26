# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:21:46 2025

@author: pranj
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

loaded_model = pickle.load(open('C:\Users\pranj\machine learning/nvidia_model.sav','rb'))
input_data=[0.04477530714919751,0.0355814614532041,0.0401187785903406,2714688000]

input_data_as_array= np.asarray(input_data)
input_data_reshape= input_data_as_array.reshape(1,-1)
std_data= scaler.transform(input_data_reshape)

prediction= loaded_model.predict(std_data)

print(prediction)