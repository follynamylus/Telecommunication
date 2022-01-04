#Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as ex
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle


st.header('TELCO ANALYSIS') # <-- Create Web app header
df = pd.read_csv('telco.csv') # <-- Read in data

# Sort columns to categorical and numerical.
num_cols = []
categorical_cols = []
def sort_cols(data):


    for i in data.columns:
        if data[i].dtypes == "object":
            categorical_cols.append(i)
        else :
            num_cols.append(i)
sort_cols(df)

## Fix missing values for categorical and numerical columns
def fix_cat_cols(data,cat_columns):
    for i in cat_columns:
        data[i] = data[i].fillna(data[i].mode()[0])
fix_cat_cols(df,categorical_cols)
def fix_num_cols(data,num_columns):
    for i in num_columns:
        data[i] = data[i].fillna(data[i].mean())
fix_num_cols(df,num_cols)

## Scale numerical columns
df[num_cols] = StandardScaler().fit_transform(df[num_cols])

## Generate top 3 most used application
df['Social Media data'] = df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
df['Google data'] = df['Google DL (Bytes)'] + df['Google UL (Bytes)']
df['Email data'] = df['Email DL (Bytes)'] + df['Email UL (Bytes)']
df['Youtube data'] = df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
df['Netflix data'] = df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
df['Gaming data'] = df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']
df['Other data'] = df['Other DL (Bytes)'] + df['Other UL (Bytes)']
top_3_apps = ["Gaming data","Other data","Youtube data"]



df['Total_volume (Bytes)'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
relevant_cols = ['Bearer Id','Dur. (ms)','Other data','Gaming data','Youtube data','Netflix data',
                'Email data','Google data','Social Media data','Total UL (Bytes)','Total DL (Bytes)']

## Univariate EDA for Numerical
for i in top_3_apps :
    st.write(ex.histogram(df,i,color = i,marginal = 'box' ))

##plots for scaled applications against total data volume

##for i in relevant_cols :
    ##st.write(ex.scatter(df, x='Total_volume (Bytes)',y=i , color= i, title= f'Scaled {i} against Total_volume (Bytes)'))


## plots for top 3 most used applications
for i in top_3_apps:
    st.write(ex.scatter(df, x='MSISDN/Number',y=i , color= i, title= f'Scaled {i}'))

## Load model file
model = pickle.load(open('model_pkl_1','rb'))

## Create widgets to input features
session_frequency = float(st.text_input('input session frequency',3))
Avg_RTT_UL = float(st.text_input('Avg_RTT_UL (ms)',20))
Avg_RTT_DL = float(st.text_input('Avg_RTT_DL (ms)', 50))
Avg_BearerTP_UL = float(st.text_input('Avg_BearerTP_UL (kbps)', 200))
Avg_BearerTP_DL = float(st.text_input('Avg_BearerTP_DL (kbps)', 1200))
TCP_Retrans_vol_UL = float(st.text_input('TCP_Retrans_vol_UL (Bytes)',150000 ))
TCP_Retrans_vol_DL = float(st.text_input('TCP_Retrans_vol_DL (Bytes)',500000))
Total_UL = float(st.text_input('Total_UL (Bytes)', 2000000))
Total_DL = float(st.text_input('Total_DL (Bytes)', 5000000))
Dur_ms = float(st.text_input('Dur. (ms)',150000))

## Generate Features from widget inputs
Avg_RTT = Avg_RTT_UL + Avg_RTT_DL
Avg_BearerTP = Avg_BearerTP_UL + Avg_BearerTP_DL
TCP_Retrans_vol = TCP_Retrans_vol_UL + TCP_Retrans_vol_DL
Total_Data = Total_UL + Total_DL

## Store Standard scaler as scaler variable
scaler = StandardScaler()

## Convert Scaled features to dataframe
scaled = [[session_frequency,Avg_RTT,Avg_BearerTP,TCP_Retrans_vol,Total_Data,Dur_ms]]

## Scale Dataframe
X = scaler.fit_transform(scaled)


## Make predictions
st.write(f'The predicted satisfactory score is : {model.predict(X)}')