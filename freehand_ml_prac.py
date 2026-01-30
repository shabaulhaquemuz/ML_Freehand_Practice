#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv("Covid_toy.csv")


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


from sklearn.impute import SimpleImputer


# In[8]:


si=SimpleImputer()
df["fever"]=si.fit_transform(df[["fever"]])


# In[9]:


df.isnull().sum()


# In[10]:


df.head()


# In[12]:


df=pd.get_dummies(df, columns=["gender"])


# In[14]:


df=pd.get_dummies(df, columns=["city"])


# In[13]:


df.head()


# In[16]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["has_covid"]=le.fit_transform(df["has_covid"])


# In[17]:


df.head()


# In[20]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df["cough"]=oe.fit_transform(df[["cough"]])


# In[21]:


df.head()


# In[22]:


df.astype(int)


# In[23]:


x=df.drop(columns=["has_covid"])


# In[24]:


y=df["has_covid"]


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[29]:


y_pred=lr.predict(x_test)


# In[30]:


from sklearn.metrics import accuracy_score


# In[31]:


accuracy_score(y_test, y_pred)


# In[32]:


from sklearn.metrics import confusion_matrix


# In[33]:


confusion_matrix(y_test, y_pred)


# In[ ]:




