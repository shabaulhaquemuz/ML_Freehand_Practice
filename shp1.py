#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("Covid_toy.csv")


# In[3]:


df.head(5)


# In[4]:


df.isnull().sum()


# In[7]:


from sklearn.impute import SimpleImputer
si=SimpleImputer()
df["fever"]=si.fit_transform(df[["fever"]])


# In[8]:


df.isnull().sum()


# In[10]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df["cough"]=oe.fit_transform(df[["cough"]])


# In[12]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["has_covid"]=le.fit_transform(df["has_covid"])


# In[13]:


df=pd.get_dummies(df, columns=["city","gender"])


# In[14]:


df.astype(int)


# In[15]:


x=df.drop(columns="has_covid")
y=df["has_covid"]


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)


# In[17]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[19]:


y_pred=dt.predict(x_test)


# In[21]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




