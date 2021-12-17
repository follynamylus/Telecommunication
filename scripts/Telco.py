import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as ex
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle


st.header('TELCO ANALYSIS')
df = pd.read_csv('telco.csv')
print(df)
num_cols = []
categorical_cols = []
def sort_cols(data):


    for i in data.columns:
        if data[i].dtypes == "object":
            categorical_cols.append(i)
        else :
            num_cols.append(i)
sort_cols(df)
def fix_cat_cols(data,cat_columns):
    for i in cat_columns:
        data[i] = data[i].fillna(data[i].mode()[0])
fix_cat_cols(df,categorical_cols)
def fix_num_cols(data,num_columns):
    for i in num_columns:
        data[i] = data[i].fillna(data[i].mean())
fix_num_cols(df,num_cols)
df['Total_volume (Bytes)'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
df['Session frequency'] = df.groupby(['Bearer Id'])['Dur. (ms)'].transform('count')
X = df[['Session frequency','Dur. (ms)','Total_volume (Bytes)']]
scaler = StandardScaler()
col_names = ["Session frequency","Dur. (ms)",'Total_volume (Bytes)']
X[col_names] = scaler.fit_transform(X[col_names])
k = 3
centroids = (X.sample(n=k))
'''
#st.write(ex.scatter(X, x='Dur. (ms)', y='Total_volume (Bytes)', color='Session frequency' ),
ex.scatter(centroids,x='Dur. (ms)',y='Total_volume (Bytes)',color='Session frequency'))
'''

kmeans = KMeans(
    init= "random",
    n_clusters = 3,
    n_init= 10,
    max_iter= 300,
    random_state= 42
)
kmeans.fit(X)
X['labels'] = kmeans.labels_
df['Social Media data'] = df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
df['Google data'] = df['Google DL (Bytes)'] + df['Google UL (Bytes)']
df['Email data'] = df['Email DL (Bytes)'] + df['Email UL (Bytes)']
df['Youtube data'] = df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
df['Netflix data'] = df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
df['Gaming data'] = df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']
df['Other data'] = df['Other DL (Bytes)'] + df['Other UL (Bytes)']
top_3_apps = ["Gaming data","Other data","Youtube data"]
for i in top_3_apps:
    st.write(ex.scatter(df, x='MSISDN/Number',y=i))

model = pickle.load(open('model_pkl_1','rb'))


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

Avg_RTT = Avg_RTT_UL + Avg_RTT_DL
Avg_BearerTP = Avg_BearerTP_UL + Avg_BearerTP_DL
TCP_Retrans_vol = TCP_Retrans_vol_UL + TCP_Retrans_vol_DL
Total_Data = Total_UL + Total_DL

scaler = StandardScaler()

scaled = [[session_frequency,Avg_RTT,Avg_BearerTP,TCP_Retrans_vol,Total_Data,Dur_ms]]

X = scaler.fit_transform(scaled)

st.write(f'The predicted satisfactory score is : {model.predict(X)}')