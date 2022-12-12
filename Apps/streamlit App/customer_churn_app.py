# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:40:18 2022

@author: Ebenezer Edusei
"""

import numpy as np
import pickle
import streamlit as st

# load the saved model
loaded_model = pickle.load(open('C:/Users/Ebenezer Edusei/anaconda3/envs/PythonEnv/Edusei_Demo/Classification Assignment/trained_model.sav', 'rb'))


#Creating a function for prediction
def churn_prediction(input_data):
     
     
    #change the input data into a numpy array
    input_data_nparray = np.asarray(input_data)

    #reshape the array given we are predicting for one instance
    input_data_reshape = input_data_nparray.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0]==0):
        return('customer will not churn')
    else:
        return('customer will churn')
    
    
    
    
def main():
    
    #Titling our web app
    st.title("Customer Churn Web App")
    
    
    #Getting user input data
    #gender = st.text_input("Please indicate your gender") 
    SeniorCitizen = st.selectbox("Indicate if customer is a senior citizen or not(Yes=1,No=0)", [1,0])
    tenure   = st.slider("Indicate user subscription duration", 1,25,50)
    PhoneService  = st.selectbox("Indicate customer preference for phone service(Yes=1,No=0)", [1,0])
    TotalCharges = st.slider("Indicate user's total subscription charges", 1,1000)
    InternetService_DSL = st.selectbox("Indicate customer preference of DSL yes=1,or no=0",[1,0])
    InternetService_Fiberoptic = st.selectbox("Indicate customer preference of Fibre optics yes=1,or no=0",[1,0]) 
    OnlineSecurity_Yes = st.selectbox("Indicate customer preference for online security yes=1,or no=0",[1,0])
    TechSupport_Yes = st.selectbox("Indicate customer access to Techn support, yes=1,or no=0",[1,0])  
    StreamingMovies_Yes = st.selectbox("Indicate customer preference of streaming movies yes=1,or no=0",[1,0])  
    Contract_Month = st.selectbox("Indicate customer preference of monthly subscription yes=1,or no=0",[1,0])  
    Contract_Oneyear = st.selectbox("Indicate customer preference of yearly subscription yes=1,or no=0",[1,0]) 
    PaymentMethod_Electroniccheck = st.selectbox("Indicate customer preference of electronic payement yes=1,or no=0",[1,0]) 
    #PaymentMethod_Mailedcheck = st.text_input("Indicate customer preference for mailed check payment")
        
    
    # code for prediction
    customer_churn = " "
    
    #Creating a button for predition
    
    if st.button("Predict"):
        customer_churn = churn_prediction([[SeniorCitizen, tenure, PhoneService,TotalCharges, InternetService_DSL, InternetService_Fiberoptic, OnlineSecurity_Yes, TechSupport_Yes, StreamingMovies_Yes, Contract_Month, Contract_Oneyear, PaymentMethod_Electroniccheck]])
    
        st.success(customer_churn)
    
    
    
    
if __name__=='__main__':
    main()
    
    
    
    
