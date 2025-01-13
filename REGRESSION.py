#!/usr/bin/env python
# coding: utf-8

# # Regression model : Random Forest Regressor 

# # Importing Packages

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# # Importing dataset

# In[2]:


data = pd.read_csv(r"C:\Users\Asus\Downloads\industry.zip")
data


# # Encoding categorical variables

# In[3]:


equipment_encoder = LabelEncoder()
location_encoder = LabelEncoder()
data['equipment'] = equipment_encoder.fit_transform(data['equipment'])
data['location'] = location_encoder.fit_transform(data['location'])


# In[4]:


X = data.drop(columns=['faulty'])
y = data['faulty']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Training Regression Model

# In[7]:


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# # Predicted 

# In[8]:


y_pred = model.predict(X_test)


# # Model evaluation

# In[9]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[10]:


print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


# In[ ]:




