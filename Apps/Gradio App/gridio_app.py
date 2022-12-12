import gradio as gr
import pandas as pd
#from utils import *
import pickle

with open("filename.pkl", "rb") as f:
    model = pickle.load(f)



def predict_churn(SeniorCitizen, tenure, PhoneService, TotalCharges, InternetService_DSL, InternetService_Fiberoptic, OnlineSecurity_Yes, TechSupport_Yes, StreamingMovies_Yes, Contract_monthly, Contract_Yearly, PaymentMethod_Electronic):
    with open("filename.pkl", "rb") as f:
        model = pickle.load(f)
        prediction = model.predict([[SeniorCitizen, tenure, PhoneService, TotalCharges, InternetService_DSL, InternetService_Fiberoptic, OnlineSecurity_Yes, TechSupport_Yes, StreamingMovies_Yes, Contract_monthly, Contract_Yearly, PaymentMethod_Electronic]])
    if prediction == 1:
            return "Customer will churn"
    else:
            return "Customer will not churn"



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

# We create the output
output = gr.Textbox()


app = gr.Interface(fn = predict_churn, inputs=[SeniorCitizen, tenure, PhoneService, TotalCharges, InternetService_DSL, InternetService_Fiberoptic, OnlineSecurity_Yes, TechSupport_Yes, StreamingMovies_Yes,Contract_monthly, Contract_Yearly, PaymentMethod_Electronic], outputs= output)
app.launch()