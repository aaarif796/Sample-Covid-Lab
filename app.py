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

import pandas as pd
# import numpy as np
import pickle
from datetime import datetime
# import plotly.express as px
import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
# from datetime import datetime
st.markdown('<link href="styles.css" rel="stylesheet">', unsafe_allow_html=True)

df=pd.read_csv('sample_clean.csv')
df.set_index('Date',inplace=True)



    



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
    st.write('<h5>Prediction of Total Covid Sample Received, Sum of Payment received, Denial received and Denial Percentage.</h5>',unsafe_allow_html=True)
    
    
    
    # plot the diagram
    st.markdown('<h4> Plots of Data </h4>',unsafe_allow_html=True)
    
    # fig,ax=plt.subplots(nrows=4,ncols=1,figsize=(10, 20))
    # date_format = mdates.DateFormatter("%b %Y")
    # ax[0].plot(df['Covid_Samples_Received'])
    # ax[0].xaxis.set_tick_params(rotation=45)
    # ax[0].xaxis.set_major_formatter(date_format)
    # ax[0].set_ylabel('Covid_Samples Received')
    
    # ax[1].plot(df['Sum_of_Payment_Received'])
    # ax[1].xaxis.set_tick_params(rotation=45)
    # ax[1].xaxis.set_major_formatter(date_format)
    # ax[1].set_ylabel('Sum_of_Payment_Received')
    
    # ax[2].plot(df['Denials_Received'])
    # ax[2].xaxis.set_major_formatter(date_format)
    # ax[2].xaxis.set_tick_params(rotation=45)
    # ax[2].set_ylabel('Denials_Received')
    
    # ax[3].plot(df['Denial_%'])
    # ax[3].xaxis.set_major_formatter(date_format)
    # ax[3].xaxis.set_tick_params(rotation=45)
    # ax[3].set_ylabel('Denial_%')
    
    # plt.show(block=False)
    # st.pyplot(fig)
    
    fig=make_subplots(rows=2,cols=2)
    fig.add_trace(go.Scatter(x=df.index,y=df['Covid_Samples_Received'],name='Covid_Samples_Received'),
                  row=1,col=1)
    
    fig.add_trace(go.Scatter(x=df.index,y=df['Sum_of_Payment_Received'],name='Sum_of_Payment_Received'),
                  row=1,col=2)
    
    fig.add_trace(go.Scatter(x=df.index,y=df['Denials_Received'],name='Denials_Received'),
                  row=2,col=1)
    
    fig.add_trace(go.Scatter(x=df.index,y=df['Denial_%'],name='Denial_%'),
                  row=2,col=2)
    st.plotly_chart(fig)
    
    explain="""
    As we see the line plot above, we can clearly assumed that data are non stationary.
    
    So, I perform Augmented Dickey-Fuller (ADF) test to check wheather the data is staionary or not. If it's not stationary then again I perform differentiate
    between the data value to make data statinary. 
    
    After that, I tried different algorithm like AR,MA, ARMA, ARIMA, SARIMA and Prophet. But, I got better accuracy using ARIMA model.
    
    """
    
    st.write(explain)
    
    
    # Load Arima_D model
    with open('Arima_D.pkl','rb') as f:
        model_D=pickle.load(f)
        
    
    def predictmodD(select_date):
        data=model_D.predict(start=datetime(2023,4,30),
                             end=select_date)
        return pd.DataFrame(data)
    
    
    st.write()
    
    select_date=st.date_input('Select the date for forcast',min_value=datetime(2023,4,30),max_value=datetime(2024,5,31))
    select_date=select_date


    # Load Arima_DR model
    with open('Arima_DR.pkl', 'rb') as f:
        model_DR = pickle.load(f)
    
        
    def predictmodDR(select_date):
        data=model_DR.predict(start=datetime(2023,4,30),
                      end=select_date,
                      params=model_DR.params)
        return pd.DataFrame(data)
        
    # Load Arima_SPR model
    with open('Arima_SPR.pkl','rb') as f:
        model_SPR=pickle.load(f)
    
    def predictmodSPR(select_date):
        data=model_SPR.predict(start=datetime(2023,4,30),
                      end=select_date)
        # data=data[0]    
        return pd.DataFrame(data)
    
    # Load Arima_CSR model
    with open('Arima_CSR.pkl','rb') as f:
        model_CSR=pickle.load(f)
    
    def predictmodCSR(select_date):
        data=model_CSR.predict(start=datetime(2023,4,30),
                       end=select_date)
        # data=data[0]
        return pd.DataFrame(data)
    
    
    submit_button=st.button('Submit')
    if submit_button:
        data_DR=predictmodDR(select_date)
        # df = df.rename(columns={'Name': 'Full Name', 'Age': 'Age Group'})
        data_DR=data_DR.rename(columns={'predicted_mean':'Denials_Received'})
        data_D=predictmodD(select_date)
        data_D=data_D.rename(columns={'predicted_mean':'Denial_%'})
        data_SPR=predictmodSPR(select_date)
        data_SPR=data_SPR.rename(columns={'predicted_mean':'Sum_of_Payment_Received'})
        data_CSR=predictmodCSR(select_date)
        data_CSR=data_CSR.rename(columns={'predicted_mean':'Covid_Samples_Received'})        
        # data={'Total Covid Sample Received':[data_CSR],
        #       'Sum of Payment Received':[data_SPR],
        #       "Denail Received":[data_DR],
        #       "Denail percentage":[data_D]
        #     }
        data=pd.concat([data_CSR,data_SPR,data_DR,data_D],axis=1)
        data.index = data.index.to_series().dt.date
        fig=make_subplots(rows=2,cols=2)
        fig.add_trace(go.Scatter(x=df.index,y=data['Covid_Samples_Received'],name='Covid_Samples_Received'),
                      row=1,col=1)
        
        fig.add_trace(go.Scatter(x=df.index,y=data['Sum_of_Payment_Received'],name='Sum_of_Payment_Received'),
                      row=1,col=2)
        
        fig.add_trace(go.Scatter(x=df.index,y=data['Denials_Received'],name='Denials_Received'),
                      row=2,col=1)
        
        fig.add_trace(go.Scatter(x=df.index,y=data['Denial_%'],name='Denial_%'),
                      row=2,col=2)
        st.plotly_chart(fig)
        st.table(data)
    
    st.write()
    st.write()
    
    st.markdown("<b>Connect On LinkedIn:</b>",unsafe_allow_html=True)
    st.markdown('<a href="https://www.linkedin.com/in/mohammad-aarif-ansari-a6b869b6/">https://www.linkedin.com/in/mohammad-aarif-ansari-a6b869b6/</a>',
                    unsafe_allow_html=True)
        
    
    

if __name__ == '__main__':
    main()
