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
df["fever"]=si.fit_transform(df[["fever"]])


# In[6]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df["cough"]=oe.fit_transform(df[["cough"]])


# In[9]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["has_covid"]=le.fit_transform(df["has_covid"])


# In[10]:


df=pd.get_dummies(df, columns=["gender", "city"])


# In[11]:


df.head()


# In[12]:


df.astype(int)


# In[13]:


x=df.drop(columns="has_covid")
y=df["has_covid"]


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)


# In[17]:


y_pred=lr.predict(x_test)


# In[18]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[21]:


y_pred_rf=rf.predict(x_test)


# In[22]:


accuracy_score(y_test,y_pred_rf)


# In[23]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[24]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_rf)


# In[ ]:




