# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:21:53 2023

@author: shuai
"""

import numpy as np
import streamlit as st
import pickle

# loading the model
loaded_model = pickle.load(open('C:/vscode programs/machineLearning/parkinson_model/trained_parkinson_model.sav', 'rb'))
scaler = pickle.load(open('C:/vscode programs/machineLearning/parkinson_model/scaler_model.sav', 'rb'))


def parkinson_prediction(input_data):
    numpy_array = np.asarray(input_data)

    input_data_reshaped = numpy_array.reshape(1, -1)

    std_data = scaler.transform(input_data_reshaped)
    
    prediction = loaded_model.predict(std_data)
    
    if (prediction[0] == 1):
        return "The person has Parkinson Disease"
    else:
        return "The person does not have Parkinson Disease"

def main():
    st.title('Parkinson Detection Web app')
    
    average_frequency = st.text_input('Average vocal fundamental frequency')
    max_frequency = st.text_input('Maximum vocal fundamental frequency')
    min_frequency = st.text_input('Minimum vocal fundamental frequency')
    jitter_per = st.text_input('Jitter percentage')
    jitter_abs = st.text_input('Jitter_absolute')
    shimmer = st.text_input('Shimmer')
    shimmer_db = st.text_input('shimmer db')
    nhr = st.text_input('noise/harmonic ratio')
    hnr = st.text_input('harmonic/noise ratio')
    rpde = st.text_input('RPDE')
    dfa = st.text_input('Signal fractal scaling exponent')
    spread1 = st.text_input('Spread 1')
    spread2 = st.text_input('spread_2')
    d2 = st.text_input('D2')
    ppe = st.text_input('PPE')
    
    diagnosis = ''
    
    if st.button('Parkinson Detect'):
        diagnosis = parkinson_prediction([average_frequency, max_frequency, min_frequency, jitter_per, jitter_abs, 0.003306, 0.003446, 0.009920, shimmer, shimmer_db, 0.00563, 0.00680, 0.00802, 0.01689, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()