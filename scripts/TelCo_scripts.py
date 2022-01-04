## Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set_style("darkgrid")



def sort_cols(data):


    for i in data.columns:
        if data[i].dtypes == "object":
            categorical_cols.append(i)
        else :
            num_cols.append(i)



def fix_cat_cols(data,cat_columns):
    for i in cat_columns:
        data[i] = data[i].fillna(data[i].mode()[0])

def fix_num_cols(data,num_columns):
    for i in num_columns:
        data[i] = data[i].fillna(data[i].mean())




def scale(data, scale, col_names):
    scaler = scale()
    data[col_names] = scaler.fit_transform(data[col_names])

def manual_scale(data,substring,conv,replace_str,columns,n):
    conv = [j for j in data.columns if substring in j]
    for i in  conv :
        data[i.replace(replace_str,substring)] = data[i]/n
        data.drop(i, axis = 1, inplace = True)

## TASK 1.1

def group_count(data,x,y,z):
    z = data.groupby([x]).agg({y:'count'})

def group_sum(data,x,y,z):
    z = data.groupby([x]).agg({y:'sum'})

def group_double_sum(data,x,y,z,m):
    m = data.groupby([x,y]).agg({z:'count'})

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



def decile_transformation(data,x,y,n):
    df[y] = pd.qcut(data[x], n,labels=False,duplicates= 'drop')

def get_corr(data):
    

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

    



