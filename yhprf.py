#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd


# In[13]:


df=pd.read_csv("Covid_toy.csv")


# In[14]:


df.head()


# In[15]:


df.isnull().sum()


# In[16]:


from sklearn.impute import SimpleImputer
si=SimpleImputer()
df["fever"]=si.fit_transform(df[["fever"]])


# In[17]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df["cough"]=oe.fit_transform(df[["cough"]])


# In[18]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["has_covid"]=le.fit_transform(df["has_covid"])


# In[19]:


df=pd.get_dummies(df, columns=["gender","city"])


# In[20]:


df.astype(int)


# In[21]:


x=df.drop(columns="has_covid")
y=df["has_covid"]


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=42)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[25]:


y_pred=rf.predict(x_test)


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[27]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[ ]:




