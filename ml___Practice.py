#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd


# In[32]:


df=pd.read_csv("Covid_toy.csv")


# In[33]:


df.head()


# In[34]:


df.isnull().sum()


# In[35]:


from sklearn.impute import SimpleImputer
si=SimpleImputer()


# In[36]:


df["fever"]=si.fit_transform(df[["fever"]])


# In[37]:


df.isnull().sum()


# In[38]:


df = pd.get_dummies(df, columns=["gender", "city"])


# In[39]:


from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
df["cough"] = oe.fit_transform(df[["cough"]])


# In[40]:


print(df.columns)


# In[42]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["has_covid"]=le.fit_transform(df["has_covid"])


# In[43]:


df.head()


# In[45]:


df.astype(int)


# In[47]:


x=df.drop(columns=("has_covid"))
y=df["has_covid"]


# In[48]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)


# In[53]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[54]:


y_pred=lr.predict(x_test)


# In[56]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[57]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[ ]:




