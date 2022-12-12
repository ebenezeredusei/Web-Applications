A Beginner's Guide to Building and Deploying a ML Model Part Three - Web App Deployment

In the Part One of this subject, we discussed how to follow theBusiness Understanding, Data Understanding and Data Preparation stages of the CRISP-DM framework when it comes to training/building machine learning models. In Part Two, we looked at the Modeling and Evaluation Stages of the CRISP-DM framework. That leaves us with the last stage, which is DEPLOYMENT.
The assumption with developing predictive models is, it is not useful if the intended users are not able to access and utilize it to meet their needs. In the case of the data set used for this article, the goal is to deploy our model unto a web app which the organization can use to determine if its prospective customers will churn or not. We will utilize both Streamlit and Gradio for this phase.
Building a Web Application with Streamlit
This is an open source framework for developing web applications for users to use. In this section, we will look at a block of code that can be used to create the web application. Firstly, the model would have to be wrapped into a function and variable which will be inserted into our app. First, we will import our pickle library. Next we will create a variable called "filename" which will house our trained model in ".sav" format. Next we will use the 'pickle.dump' function to save the model as demonstrated below:
#import the pickle library
import pickle

#Create a file containing our trained model.
filename = "trained_model.sav"
pickle.dump(lr_model2, open(filename, "wb"))
Please not that the above block of code entails passing a file to pickle function, for it to be used to generate the pickle file. Also, understand that the open() function of pickle requires "wb" (write and binary) and "rb" (read and binary) since pickle utilizes binary protocol. "wb" is used when the model is being saved as a pickle file whereas "rb" is used when the model is being loaded. Below is a code block for loading the model:

# load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#create a function with our loaded model
input_data = (1, 189, 1, 1277, 1, 1, 1, 0, 1, 1, 1, 1)

#change the input data into a numpy array
input_data_nparray = np.asarray(input_data)

#reshape the array given we are predicting for one instance
input_data_reshape = input_data_nparray.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0]==0):
    print('customer will not churn')
else:
    print('customer will churn')
with our model generated and saved on our PC, we can now move to creating the components of our Streamlit App. Understand that our app will have four main parts:

1. The part where the necessary libraries are loaded

import numpy as np
import pickle
import streamlit as st

2. The part where the Model is loaded
# load the saved model
loaded_model = pickle.load(open('C:/Users/Ebenezer Edusei/anaconda3/envs/PythonEnv/Edusei_Demo/Classification Assignment/trained_model.sav', 'rb'))

3. The part where the Predictive function with our loaded model is specified:
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

4. The part where the various features of our web app are specified in line with the input features of our model.
 
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
Above, we use the "st.text_input" function of the streamlit library to allow for inputing numbers and eliminating the risk of our app/model reading the inputs as strings/objects. Also, we use the "st.selectbox" to create drop down menus which the user can comfortably chose an option for an entry. The "st.slider" allows for users to use a scrolling bar to indicate numeric values pertaining to an entry.

5. And the code for prediction and the creation of the prediction button.
# code for prediction
    customer_churn = " "
    
    #Creating a button for predition
    
    if st.button("Predict"):
        customer_churn = churn_prediction([[SeniorCitizen, tenure, PhoneService,TotalCharges, InternetService_DSL, InternetService_Fiberoptic, OnlineSecurity_Yes, TechSupport_Yes, StreamingMovies_Yes, Contract_Month, Contract_Oneyear, PaymentMethod_Electroniccheck]])
    
        st.success(customer_churn)
    
    
        
if __name__=='__main__':
    main()



Now, lets go through the documentation for building a simple web application using Gradio
Building a Web Application with Gradio
Gradio provides a fast and easy way for beginners to demonstrate their machine learning models using a user friendly web interface. Like the case of the Streamlit application, our model would have to be firstly wrapped and exported into a gradio application building code block.
The Gradio application can also be built using the following stages:
First, you import the important libraries needed for creating the app:
import gradio as gr
import pandas as pd
import pickle

Next, you load the saved predictive model that was built in the modeling and evaluation stage of the CRISP-DM Framework
with open('C:/Users/Ebenezer Edusei/anaconda3/envs/PythonEnv/Edusei_Demo/Classification Assignment/trained_model.sav', 'rb') as f:
    model = pickle.load(f)
    
    loaded_model = pickle.load()
Next, you indicate the function for prediction before you move on to define the features of the application:
def predict_churn(SeniorCitizen, tenure, PhoneService, TotalCharges, InternetService_DSL, InternetService_Fiberoptic, OnlineSecurity_Yes, TechSupport_Yes, StreamingMovies_Yes, Contract_monthly, Contract_Yearly, PaymentMethod_Electronic):
    with open('C:/Users/Ebenezer Edusei/anaconda3/envs/PythonEnv/Edusei_Demo/Classification Assignment/trained_model.sav', 'rb')
        model = pickle.load(f)
        prediction = model.predict([[SeniorCitizen, tenure, PhoneService, TotalCharges, InternetService_DSL, InternetService_Fiberoptic, OnlineSecurity_Yes, TechSupport_Yes, StreamingMovies_Yes, Contract_monthly, Contract_Yearly, PaymentMethod_Electronic]])
    if prediction == 1:
            return "Customer will churn"
    else:
            return "Customer will not churn"


Having done that, each feature of the app should be designed in such a way that they can accept numeric data and allow the model to use that data to make a prediction of customer churn decision. In this case, the "gr.Number" function was used to allow for the user to make such inputs. Also, the "gr.inputs.Slider" function allows users to scroll and select numeric figures based on the indicated figures of customers. These should assist with obtaining input values in the right format for predicitng customer churn decision.

SeniorCitizen = gr.Number(label = "Indicate the senior citizen status of the customer {Yes=1, No=0}")
#tenure = gr.Number(label = "indicate customer intended tenure with the service provider")
tenure = ["select the preferred tenure of the customer", gr.inputs.Slider(0, 100)]
PhoneService =  gr.Number(label = "Indicate customer access to phone service {Yes=1, No=0}")
#TotalCharges = gr.Number(label = "Indicate the total charge of the customer")
TotalCharges = ["select the total charge of the customer", gr.inputs.Slider(0, 1000)]
InternetService_DSL =  gr.Number(label = "Indicate customer access to DSL internet service {Yes=1, No=0}")
InternetService_Fiberoptic =  gr.Number(label = "Indicate the customer access to fiberoptics {Yes=1, No=0}")
OnlineSecurity_Yes = gr.Number(label = "Indicate customer access to Online security {Yes=1, No=0}")
TechSupport_Yes = gr.Number(label = "Indicate customer access to Tech Support {Yes=1, No=0}")
StreamingMovies_Yes =  gr.Number(label = "Indicate customer access to movie movie streaming (Yes=1, No=0)")
Contract_monthly = gr.Number(label = "Indicate customer preference for a monthly contract (Yes=1, No=0)")
Contract_Yearly = gr.Number(label = "Indicate customer preferrance for long term contract (Yes=1, No=0)")
PaymentMethod_Electronic = gr.Number(label = "Indicate customer preferrence for electronic payment (Yes=1, No=0)")
Lastly, a variable called app would be grated . In this variable, the Gradio interface for our customer churn prediciton machine learning model to be housed. The variable is "app" is then launched to allow for the gradio interface to open just like in the code below. 
# We create the output
output = gr.Textbox()

app = gr.Interface(fn = predict_churn, inputs=[SeniorCitizen, tenure, PhoneService, TotalCharges, InternetService_DSL, InternetService_Fiberoptic, OnlineSecurity_Yes, TechSupport_Yes, StreamingMovies_Yes,Contract_monthly, Contract_Yearly, PaymentMethod_Electronic], outputs= output)
app.launch()


Conclusion
As a begginer data scientist like myself, it is essential that our code with its multiple components are serialized. The python pickle library allows us to do that and also import our machine learning models into python codes used in the creation of apps. Also, it is essential for us to follow the CRISP-DM framework to make usre that our apporach to solving basically any data-related problem is systematic and allows for us to continually improve upon our analysis methods based on new knowledge.
Your thoughts on this articles and the others would be very much appreciated.