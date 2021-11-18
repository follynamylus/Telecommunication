## Import Libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

#%matplotlib inline


##TASK 1

df = pd.read_csv('telco.csv')

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

Df = df

from sklearn.preprocessing import MinMaxScaler
col_names = ["IMSI", "MSISDN/Number","IMEI","Bearer Id","Dur. (ms)"]

def scale(data):
    scaler = MinMaxScaler(feature_range=(5, 10))
    data[col_names] = scaler.fit_transform(data[col_names])

def bytes_scale(data,substring,conv,replace_str):
    conv = [j for j in data.columns if substring in j]
    for i in  conv :
        data[i.replace(replace_str,substring)] = data[i]/1000000
        data.drop(i, axis = 1, inplace = True)

## TASK 1.1

def group_count(data,x,y):
    x+'_per_'+y = data.groupby([x]).agg({y:'count'})

def group_sum(data,x,y):
    x+'_per_'+y = data.groupby([x]).agg({y:'sum'})

def group_double_sum(data,x,y,z):
    x+y+'_per_'+z = data.groupby([x,y]).agg({z:'count'})

## TASK 1.2


def non_grahical_EDA(data,relevant_num,relevant_cat):
    for cols in relevant_num:
        print(data[cols].describe())
        print(f"Column name is {cols}")
        print(f'skewness for this column is {data[cols].skew()}')
        print(f'kurtosis for this column is {data[cols].kurtosis()}')
        Q3,Q1 = np.percentile(data[cols], [75,25])
        IQR = Q3 - Q1
        print(f'The IQR is {IQR}')
        print(f'The number of Unique value of column {cols} is : {data[cols].nunique()}')
        print('____________________________________________________________________')
    for cols in relevant_cat:
        print(df[cols].describe(include=['O']))


def univariate_graph_num_EDA(data,relevant_num):
    for cols in relevant_num:
        sns.histplot(data=data, x= cols )
        sns.boxplot(data=data, x= cols )
        sns.kdeplot(data=data, x= cols )
        plt.show()

def bivariate_graph_num_EDA(data,relevant_app,x):
    for i in relevant_app:
        sns.scatterplot(data=data,x=x,y=i,alpha=0.5)
        plt.title(f'graph of {i} against {x}')
        plt.xlabel(x)
        plt.ylabel(i)
        plt.show()



def variable_transformation(data,x,y,newl):
    df[y] = pd.qcut(data[x], 10,labels=False,duplicates= 'drop')
    New_df = pd.DataFrame()
    New_df['total_data_UL+DL'] = data['Total_volume (MB)']
    New_df['MSISDN/Number'] = data['MSISDN/Number']
    New_df['top_5_decile_Dur. (MS)'] = data['top_5_decile_Dur. (MS)']

    new_df = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==3,:]
    new_df1 = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==2,:]
    new_df2 = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==0,:]
    new_df3 = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==8,:]
    new_df4 = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==7,:]

    new_df = pd.DataFrame(new_df.reset_index())
    new_df1 = pd.DataFrame(new_df1.reset_index())
    new_df2 = pd.DataFrame(new_df2.reset_index())
    new_df3 = pd.DataFrame(new_df3.reset_index())
    new_df4 = pd.DataFrame(new_df4.reset_index())



    newl.append(new_df)
    newl.append(new_df1)
    newl.append(new_df2)
    newl.append(new_df3)
    newl.append(new_df4)

    top_5s = pd.concat(newl,axis=0)

    top_5s.drop("index",axis=1,inplace=True)

def new_corr(data):
    df_data = pd.DataFrame()

    df_data['Social Media data'] = data['Social Media DL (MB)'] + data['Social Media UL (MB)']
    df_data['Google data'] = data['Google DL (MB)'] + data['Google UL (MB)']
    df_data['Email data'] = data['Email DL (MB)'] + df['Email UL (MB)']
    df_data['Youtube data'] = data['Youtube DL (MB)'] + data['Youtube UL (MB)']
    df_data['Netflix data'] = data['Netflix DL (MB)'] + data['Netflix UL (MB)']
    df_data['Gaming data'] = data['Gaming DL (MB)'] + data['Gaming UL (MB)']
    df_data['Other data'] = data['Other DL (MB)'] + data['Other UL (MB)']

    df_data.corr()

def standard_scale(data):
    from sklearn.preprocessing import StandardScaler
    data.drop(categorical_cols, axis=1, inplace=True)
    data = StandardScaler().fit_transform(data)


def PCA(data,principalDf):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data= principalComponents,columns = ['principal component 1', 'principal component 2'])

    



