## Import Libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler




##TASK 1

df = pd.read_csv('telco.csv')

num_cols = []
categorical_cols = []

def sort_columns(data,num_cols,categorical_cols):


    for i in data.columns:
        if data[i].dtypes == "object":
            categorical_cols.append(i)
        else :
            num_cols.append(i)
    return num_cols, categorical_cols


def fix_missing_values(data,cat_columns,num_cols):
    for i in cat_columns:
        data[i] = data[i].fillna(data[i].mode()[0])
    for i in num_columns:
        data[i] = data[i].fillna(data[i].mean())
    return df.isnull().sum()

def min_max_scale(data,col_names,x,y):
    scaler = MinMaxScaler(feature_range=(x, y))
    data[col_names] = scaler.fit_transform(data[col_names])
    return df.head(5)
def standard_scale(data,col_names):
    scaler = StandardScaler()
    data[col_names] = scaler.fit_transform(data[col_names])
    return df.head(5)

def scale_manual(data,substring,conv,replace_str,div_value):
    conv = [j for j in data.columns if substring in j]
    for i in  conv :
        data[i.replace(replace_str,substring)] = data[i]/div_value
        data.drop(i, axis = 1, inplace = True)
    return df.head()

## TASK 1.1

def single_cols_groupby(data,x,y,z):
    z = data.groupby([x]).agg({y:'count'})
    return z

def group_sum(data,x,y,z):
    z = data.groupby([x]).agg({y:'sum'})
    return z

def group_double_sum(data,x,y,z):
    z = data.groupby([x,y]).agg({z:'count'})
    return z

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


def univariate_graphical(data,relevant_num):
    for cols in relevant_num:
        sns.histplot(data=data, x= cols )
        sns.boxplot(data=data, x= cols )
        sns.kdeplot(data=data, x= cols )
        plt.show()

def bivariate_graphical(data,relevant_app,x):
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

def new_corr(data,X,Y):
    df_data = pd.DataFrame()

    df_data['Social Media data'] = data['Social Media DL (MB)'] + data['Social Media UL (MB)']
    df_data['Google data'] = data['Google DL (MB)'] + data['Google UL (MB)']
    df_data['Email data'] = data['Email DL (MB)'] + df['Email UL (MB)']
    df_data['Youtube data'] = data['Youtube DL (MB)'] + data['Youtube UL (MB)']
    df_data['Netflix data'] = data['Netflix DL (MB)'] + data['Netflix UL (MB)']
    df_data['Gaming data'] = data['Gaming DL (MB)'] + data['Gaming UL (MB)']
    df_data['Other data'] = data['Other DL (MB)'] + data['Other UL (MB)']

    df_data.corr()
def PCA(data,principalDf,principal_1,principal_2):
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data= principalComponents,columns = [principal_1, principal_2])
    return principalDf
    

    



