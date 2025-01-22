#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


data = pd.read_csv(r"C:\Users\Asus\Downloads\industry.zip")
data


# In[10]:


data.fillna(data.mean()) 


# In[11]:


label_encoder = LabelEncoder()
data['equipment_encoded'] = label_encoder.fit_transform(data['equipment'])
data['location_encoded'] = label_encoder.fit_transform(data['location'])


# In[12]:


X = data[['temperature', 'pressure', 'vibration', 'humidity', 'equipment_encoded', 'location_encoded']]


# In[13]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[14]:


iso_forest = IsolationForest(contamination=0.05) 
outliers = iso_forest.fit_predict(X_scaled)


# In[15]:


data['outlier'] = np.where(outliers == -1, 1, 0)


# In[16]:


kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)


# In[17]:


plt.figure(figsize=(8, 6))
plt.scatter(data['temperature'], data['vibration'], c=data['cluster'], cmap='viridis')
plt.title('Clustering of Equipment States')
plt.xlabel('Temperature')
plt.ylabel('Vibration')
plt.colorbar(label='Cluster')
plt.show()


# In[21]:


sil_score = silhouette_score(X_scaled, data['cluster'])
print(f"Silhouette Score: {sil_score}")


# In[22]:


db_score = davies_bouldin_score(X_scaled, data['cluster'])
print(f"Davies-Bouldin Score: {db_score}")


# In[23]:


inertia = kmeans.inertia_
print(f"Inertia: {inertia}")


# In[24]:


cluster_analysis = data.groupby('cluster').mean()
print(cluster_analysis)


# In[25]:


faulty_analysis = data.groupby('cluster')['faulty'].mean()
print(faulty_analysis)


# In[ ]:




