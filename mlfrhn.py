#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd


# In[19]:


df=pd.read_csv("Covid_toy.csv")


# In[20]:


df.head()


# In[21]:


df.isnull().sum()


# In[22]:


from sklearn.impute import SimpleImputer
si=SimpleImputer()
df["fever"]=si.fit_transform(df[["fever"]])


# In[23]:


df.isnull().sum()


# In[24]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df["cough"]=oe.fit_transform(df[["cough"]])


# In[25]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["has_covid"]=le.fit_transform(df["has_covid"])


# In[26]:


df=pd.get_dummies(df, columns=["city","gender"])


# In[27]:


df.head()


# In[28]:


df.astype(int)


# In[29]:


x=df.drop(columns="has_covid")


# In[30]:


y=df["has_covid"]


# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)


# In[32]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train, y_train)


# In[33]:


y_pred=rf.predict(x_test)


# In[34]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[35]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[ ]:




