#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("Covid_toy.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


from sklearn.impute import SimpleImputer
si=SimpleImputer()


# In[6]:


df["fever"]=si.fit_transform(df[["fever"]])


# In[7]:


df.isnull().sum()


# In[8]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df["cough"]=oe.fit_transform(df[["cough"]])


# In[10]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["has_covid"]=le.fit_transform(df["has_covid"])


# In[13]:


df=pd.get_dummies(df,columns=["gender","city"])


# In[14]:


df.head()


# In[15]:


df.astype(int)


# In[16]:


x=df.drop(columns="has_covid")
y=df["has_covid"]


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)


# In[28]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[29]:


y_pred = lr.predict(x_test)


# In[30]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




