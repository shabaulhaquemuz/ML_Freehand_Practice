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


# In[6]:


from sklearn.impute import SimpleImputer
si=SimpleImputer()
df["fever"]=si.fit_transform(df[["fever"]])


# In[9]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df["cough"]=oe.fit_transform(df[["cough"]])


# In[11]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["has_covid"]=le.fit_transform(df["has_covid"])


# In[12]:


df=pd.get_dummies(df, columns=["gender","city"])


# In[13]:


df.sample()


# In[14]:


df.astype(int)


# In[15]:


x=df.drop(columns="has_covid")
y=df["has_covid"]


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)


# In[20]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)


# In[22]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train, y_train)


# In[31]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train, y_train)


# In[23]:


ypredlr=lr.predict(x_test)


# In[24]:


ypredrf=rf.predict(x_test)


# In[32]:


ypreddt=dt.predict(x_test)


# In[27]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, ypredlr)


# In[28]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, ypredrf)


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, ypreddt)


# In[34]:


from  sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypreddt)


# In[ ]:




