#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


data = pd.read_csv(r'C:\Users\Asus\Downloads\industry.zip')
data


# In[3]:


print("Dataset Head:\n", data.head())
print("\nDataset Info:\n")
print(data.info())
print("\nSummary Statistics:\n", data.describe())


# In[4]:


data.fillna(data.median(), inplace=True)


# In[5]:


data = pd.get_dummies(data, columns=['equipment', 'location'], drop_first=True)


# In[6]:


X = data.drop('faulty', axis=1)  
y = data['faulty']  


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


best_accuracy = 0
best_params = {}


# In[11]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[12]:


y_pred = model.predict(X_test)


# In[13]:


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, y_pred))



# In[14]:


importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", importance)


# In[15]:


plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=importance.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# In[ ]:




