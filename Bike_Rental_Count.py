#!/usr/bin/env python
# coding: utf-8

# ## Importing Required libraries

# In[271]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[272]:


# Loading the dataset
data = pd.read_csv("day.csv")


# In[273]:


#Checking the dimensions of dataset
data.shape


# In[274]:


data.head()


# In[275]:


#Inital insight of datset
data.describe()


# In[276]:


data.dtypes


# In[277]:


## Removing unnessary variables from dataset
# instant - It is basically index number
# dteday - All the values from dteday are present in datset under differnet variables
# casual and registered - cnt is basically the sum of casual amd registerd variables
data = data.drop(['instant','dteday','casual','registered'],axis=1)


# In[278]:


#Creating a copy of original dataset
data_vis = data.copy()
data.head()


# In[279]:


##Converting the interger values into proper naming
data_vis['season'] = data_vis['season'].replace([1,2,3,4],['Springer','summer','fall','winter'])
data_vis['yr'] = data_vis['yr'].replace([0,1],[2011,2012])
data_vis['weathersit'] = data_vis['weathersit'].replace([1,2,3,4],[' Clear+Few clouds+Partly cloudy','Mist + Cloudy, Mist + Broken clouds, ',' Light Snow, Light Rain + Thunderstorm ','Heavy Rain + Ice Pallets '])
data_vis['holiday'] = data_vis['holiday'].replace([0,1],['working Day','Holiday'])
data_vis.head()


# In[280]:


print(data.dtypes)
print(data.head())


# ## Univarient Analysis

# In[281]:


## Bar Graph for Categorical data

sns.set_style("whitegrid")
sns.factorplot(data=data_vis,x='season',kind='count',size=4,aspect=2)
sns.factorplot(data=data_vis,x='yr',kind='count',size=4,aspect=2)
sns.factorplot(data=data_vis,x='mnth',kind='count',size=4,aspect=2)
sns.factorplot(data=data_vis,x='holiday',kind='count',size=4,aspect=2)
sns.factorplot(data=data_vis,x='workingday',kind='count',size=4,aspect=2)
sns.factorplot(data=data_vis,x='weathersit',kind='count',size=4,aspect=2)


# In[282]:


plt.hist(data_vis['temp'],bins=30)
plt.xlabel('temp')
plt.ylabel('Frequency')
plt.show()


# In[283]:


plt.hist(data_vis['atemp'],bins=30)
plt.xlabel('atemp')
plt.ylabel('Frequency')
plt.show()


# In[284]:


plt.hist(data_vis['hum'],bins=30)
plt.xlabel('humidity')
plt.ylabel('Frequency')
plt.show()


# In[285]:


plt.hist(data_vis['windspeed'],bins=30)
plt.xlabel('WindSpeed')
plt.ylabel('Frequency')
plt.show()


# In[286]:


#Dimensions of dataset after removing outliers
data.shape


# ## Bivariant Analysis

# In[287]:


##  Using Scatter Plot
# Index(['instant', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
#        'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual',
#        'registered', 'cnt'],
#       dtype='object')


# In[288]:


fig,x = plt.subplots(nrows= 2,ncols=2)
fig.set_size_inches(12,15)
sns.scatterplot(x="temp",y = "cnt",data = data_vis,palette="Set3",ax=x[0][0])
sns.scatterplot(x="atemp",y = "cnt",data = data_vis,palette="Set3",ax=x[0][1])
sns.scatterplot(x="hum",y = "cnt",data = data_vis,palette="Set3",ax=x[1][0])
sns.scatterplot(x="windspeed",y = "cnt",data = data_vis,palette="Set3",ax=x[1][1])


# ## Outlier Analysis

# In[289]:


## Checking the presence of outlier in continous variables
sns.boxplot(data = data[['temp','atemp','windspeed','hum']])
fig  = plt.gcf()
fig.set_size_inches(8,8)


# In[290]:


## Removing outlier and checking correlation between target variable and independent continous variables

print(data.shape)
print(data['hum'].corr(data['cnt']))
print(data['windspeed'].corr(data['cnt']))

q75, q25 = np.percentile(data.loc[:,'hum'],[75,25])
iqr = q75 - q25

min = q25-(iqr*1.5)
max = q75+(iqr*1.5)

 
print(min)
print(max)

data = data.drop(data[data.loc[:,'hum']<min].index)
data = data.drop(data[data.loc[:,'hum']>max].index)


q75, q25 = np.percentile(data.loc[:,'windspeed'],[75,25])
iqr = q75 - q25

min = q25-(iqr*1.5)
max = q75+(iqr*1.5)

 
print(min)
print(max)

data = data.drop(data[data.loc[:,'windspeed']<min].index)
data = data.drop(data[data.loc[:,'windspeed']>max].index)


# ## Missing Value Analysis

# In[291]:


total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)

# There are no missing vlaues present after outlier analysis


# ## Feature Selection

# In[292]:


def Correlation(df):
    df_corr = df.loc[:,df.columns]
    corr = df_corr.corr()
    sns.set()
    plt.figure(figsize=(10,10))
    sns.heatmap(corr,annot=True,fmt=".3f",square=True,linewidths=0.5)

    
Correlation(data)


# In[293]:


## There is high correlation between temp and atemp variable
## there is very weak relation between holiday, weekday and working day variables
## So we will drop those variables
data_fs = data.drop(['atemp','holiday','weekday','workingday'],axis=1)
data_fs.head()


# In[294]:


# Splitting Dataset into train and test dataset

train,test = train_test_split(data_fs,test_size=0.2,random_state=121)


# ## Feature Scaling 

# In[295]:


## Data is normalized no need to do feature scaling
train.head()


# ## Error Metrics 

# In[296]:


## Defining Performance Metrics

def MAPE(y_true, y_pred):
    MAE = np.mean(np.abs((y_true - y_pred)))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    print("MAE is:", MAE)
    print("MAPE is:", mape)
    return mape

def RMSE(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    print("MSE: ",mse)
    print("RMSE: ",rmse)
    return rmse
    


# ## Linear Regression 

# In[297]:


LR_model = sm.OLS(train.iloc[:,7],train.iloc[:,0:6]).fit()

#Summary
print(LR_model.summary())

#Predict
LR_Model_predict = LR_model.predict(test.iloc[:,0:6])


# In[298]:


MAPE(test.iloc[:,7],LR_Model_predict)
RMSE(test.iloc[:,7],LR_Model_predict)


# In[299]:


result = pd.DataFrame({'Actual Value':test.iloc[:,7],'Linear Regression':LR_Model_predict})
result.head()


# ## Desicion Tree

# In[300]:


DT_model = DecisionTreeRegressor(random_state=100).fit(train.iloc[:,0:6],train.iloc[:,7])

#prediction
DT_model_predict = DT_model.predict(test.iloc[:,0:6],DT_model)


# In[ ]:





# In[301]:


MAPE(test.iloc[:,7],DT_model_predict)
RMSE(test.iloc[:,7],DT_model_predict)


# In[302]:


result['Desicion Tree'] = DT_model_predict
result.head()


# ## Random Forest

# In[303]:


RF_model = RandomForestRegressor(random_state=123)
np.random.seed(10)

arg_dict = {'max_depth':[2,4,6,8,10],
           'bootstrap':[True,False],
           'max_features':['auto','sqrt','log2',None],
           'n_estimators':[100,200,300,400,500]}

gs_randomForest = RandomizedSearchCV(RF_model,cv=10,param_distributions=arg_dict,
                                    n_iter=10)

gs_randomForest.fit(train.iloc[:,0:6],train.iloc[:,7])
print("Best Parameters using random Search",
     gs_randomForest.best_params_)


# In[304]:


RF_model.set_params(n_estimators = 500,
                   max_features='sqrt',
                   max_depth=8,
                   bootstrap=True)
RF_model.fit(train.iloc[:,0:6],train.iloc[:,7])
RF_model_predict = RF_model.predict(test.iloc[:,0:6])


# In[305]:


MAPE(test.iloc[:,7],RF_model_predict)
RMSE(test.iloc[:,7],RF_model_predict)


# In[306]:


result['Random Forest'] = RF_model_predict
result.head()


# From above models Random forest is performing well according to RMSE values

# In[307]:


#Saving the result of test data onto local machine
result.to_csv("Test_Result_python.csv",index=False)


# In[ ]:




