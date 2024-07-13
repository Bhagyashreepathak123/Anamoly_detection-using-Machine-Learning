#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# In[8]:


file_path = "C:/Users/Bhagyashree Pathak/Desktop/creditcard.csv"
df = pd.read_csv("creditcard.csv")


# In[9]:


print("Shape of the dataset:", df.shape)


# In[10]:


print("Null values in the dataset:\n", df.isnull().sum())


# In[12]:


features = df.drop(columns=['Class'])
target = df['Class']


# In[13]:


# Displaying class distribution
print("Class distribution:\n", target.value_counts())


# In[14]:


# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[15]:


clf = IsolationForest(contamination=0.0017, random_state=42) 


# In[16]:


clf.fit(features_scaled)


# In[18]:


# Predicting anomalies
y_pred = clf.predict(features_scaled)
y_pred = np.where(y_pred == -1, 1, 0) 


# In[20]:


print("Number of detected anomalies:", sum(y_pred))


# In[21]:


#Evaluating the model
print("Classification Report:")
print(classification_report(target, y_pred))


# In[22]:


# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(target, y_pred))

