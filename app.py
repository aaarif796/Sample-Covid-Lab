# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 19:15:27 2023

@author: aaari
"""
import streamlit as st
st.set_page_config(
    page_title="Covid Clinic App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    
)


import pickle
from datetime import datetime
# from datetime import datetime
st.markdown('<link href="styles.css" rel="stylesheet">', unsafe_allow_html=True)


def main():
    centered_title_style = """
        <style>
        .centered_title {
            display: flex;
            justify-content: center;
        }
        body{
            background-color:#f2f2f2;
            }
        
        </style>
    """
    
    # Render the custom CSS
    st.markdown(centered_title_style, unsafe_allow_html=True)
    # Set app title and description
    st.title('Sample Lab Data')
    
    st.markdown('<h3 style="color: red;">Covid Datasets Forcasting Using TimeSeries.</h3>',unsafe_allow_html=True)

    # Add content to the app
    st.write('<div style="color:blue;font-style: italic;">Prediction of Total Covid Sample Received, Sum of Payment received, Denial received and Denial Percentage.</div>',unsafe_allow_html=True)
    

    # Load Arima_D model
    with open('Arima_D.pkl','rb') as f:
        model_D=pickle.load(f)
        
    
    def predictmodD(select_date):
        data=model_D.predict(start=select_date,
                             end=select_date)
        data=data[0]
        return data
    
    
    
    select_date=st.date_input('Select Date',min_value=datetime(2022,8,31),max_value=datetime(2024,6,30))
    select_date=select_date


    # Load Arima_DR model
    with open('Arima_DR.pkl', 'rb') as f:
        model_DR = pickle.load(f)
    
        
    def predictmodDR(select_date):
        data=model_DR.predict(start=select_date,
                      end=select_date,
                      params=model_DR.params)
        data=data[0]    
        return data
        
    # Load Arima_SPR model
    with open('Arima_SPR.pkl','rb') as f:
        model_SPR=pickle.load(f)
    
    def predictmodSPR(select_date):
        data=model_SPR.predict(start=select_date,
                      end=select_date)
        data=data[0]    
        return data
    
    # Load Arima_CSR model
    with open('Arima_CSR.pkl','rb') as f:
        model_CSR=pickle.load(f)
    
    def predictmodCSR(select_date):
        data=model_CSR.predict(start=select_date,
                       end=select_date)
        data=data[0]
        return data
    
    
    submit_button=st.button('Submit')
    if submit_button:
        data_DR=predictmodDR(select_date)
        data_D=predictmodD(select_date)
        data_SPR=predictmodSPR(select_date)
        data_CSR=predictmodCSR(select_date)
        st.write('Total Covid Sample Received:',data_CSR)
        st.write('Sum of Payment Received:',data_SPR)
        st.write("Denail Received is:",data_DR) 
        st.write('Denail percentage is:',data_D)
        
    
    

if __name__ == '__main__':
    main()
