#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# Ignore harmless warnings 
import warnings
warnings.filterwarnings("ignore")


# Set to display all the columns in dataset
pd.set_option("display.max_columns",None)


# Import psql to run queries
import pandasql as psql


# In[2]:


#Load  the dataset
BMdata_set = pd.read_csv(r"C:\Users\Forha\Downloads\bank-direct-marketing-campaigns.csv",header=0)

# Copy to back-up files
BMdata_set_bk = BMdata_set.copy()

#Display 5 record
BMdata_set.head()


# In[3]:


#Display dataset information
BMdata_set.info()


# In[4]:


#display the unique values of the all variables
BMdata_set.nunique()


# In[5]:


#use LabelEnabler to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

BMdata_set['job'] = LE.fit_transform(BMdata_set[['job']])

BMdata_set['marital'] = LE.fit_transform(BMdata_set[['marital']])

BMdata_set['education'] = LE.fit_transform(BMdata_set[['education']])

BMdata_set['default'] = LE.fit_transform(BMdata_set[['default']])

BMdata_set['housing'] = LE.fit_transform(BMdata_set[['housing']])

BMdata_set['loan'] = LE.fit_transform(BMdata_set[['loan']])

BMdata_set['contact'] = LE.fit_transform(BMdata_set[['contact']])

BMdata_set['month'] = LE.fit_transform(BMdata_set[['month']])

BMdata_set['day_of_week'] = LE.fit_transform(BMdata_set[['day_of_week']])

BMdata_set['poutcome'] = LE.fit_transform(BMdata_set[['poutcome']])

BMdata_set['y'] = LE.fit_transform(BMdata_set[['y']])


# In[6]:


#Display dataset information
BMdata_set.info()


# In[ ]:





# In[7]:


# Count the target or dependent variable by '0' & '1' and their proportion 
# (>= 10 : 1, then the dataset is imbalance data)

y_count = BMdata_set.y.value_counts()
print('Class 0:', y_count[0])
print('Class 1:', y_count[1])
print('Proportion:', round(y_count[0] / y_count[1], 2), ': 1')
print('Total Bank records:', len(BMdata_set))


# In[8]:


#count the missing values by each variables,
BMdata_set.isnull().sum()


# In[9]:


#display the 'y' variable by sub-variable count
BMdata_set['y'].value_counts()


# In[10]:


#Display duplicating data in dataset
BMdata_set_dup = BMdata_set[BMdata_set.duplicated(keep = 'last')]
BMdata_set_dup


# In[11]:


BMdata_set.duplicated().any()


# In[12]:


#To remove the identified duplicate records
BMdata_set = BMdata_set.drop_duplicates()

#display the data
BMdata_set


# In[13]:


#re-setting the row index
BMdata_set = BMdata_set.reset_index(drop = True)

#Copy file to back-up file after deletion of duplicates records
BMdata_set = BMdata_set.copy()


# In[14]:


BMdata_set.duplicated().any()


# In[15]:


#identify the independent and target (dependent) variables

IndepVar = []
for col in BMdata_set.columns:
    if col != 'y':
        IndepVar.append(col)
TargetVar = 'y'

x = BMdata_set[IndepVar]
y = BMdata_set[TargetVar]


# In[16]:


# Split the data into Train and Test(random samping)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

# Display shape for train and test data

x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[17]:


#Scaling the features
from sklearn.preprocessing import MinMaxScaler

scaler =  MinMaxScaler(feature_range=(0,1))

x_train =  scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


# In[18]:


x_train
