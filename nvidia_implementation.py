# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:27:30 2025

@author: pranj
"""

import numpy as np
import pickle
import streamlit as st

loaded_model= pickle.load(open('C:\Users\pranj\machine learning/nvidia_model.sav','rb'))


def nvidia_share_prediction(input_data):
    
    input_data= [0.04477530714919751,0.0355814614532041,0.0401187785903406,2714688000]
    
    input_data_as_array= np.asarray(input_data)
    input_data_reshape= input_data_as_array.reshape(1,-1)
    prediction= loaded_model.predict(input_data_reshape)
    print(prediction)
    
def main():
    
    st.title('NVidia share price prediction web app')
    
    High= st.text_input('highest price of the share')
    Low= st.text_input('Lowest price of the share')
    Open= st.text_input('opening price of the share')
    Volume= st.text_input('no of shares purchased')
    
    close_price=''
    
    if st.button('closing price prediction result'):
        close_price= nvidia_share_prediction([High,Low,Open,Volume])
        st.success(close_price)
        
if __name__=='__main__':
    main()
       
    