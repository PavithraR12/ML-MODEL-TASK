#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"C:\Users\Asus\Downloads\industry.zip")
data


# In[3]:


data = data.fillna(data.mean())


# In[4]:


label_encoder = LabelEncoder()
data['equipment_encoded'] = label_encoder.fit_transform(data['equipment'])
data['location_encoded'] = label_encoder.fit_transform(data['location'])


# In[5]:


X = data[['temperature', 'pressure', 'vibration', 'humidity', 'equipment_encoded', 'location_encoded']]


# In[6]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[7]:


pca = PCA(n_components=2)  
X_pca = pca.fit_transform(X_scaled)


# In[8]:


print(f"Explained variance ratio of the first two principal components: {pca.explained_variance_ratio_}")
print(f"Total variance explained by these 2 components: {sum(pca.explained_variance_ratio_):.2f}")


# In[9]:


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['faulty'], cmap='viridis', alpha=0.7)
plt.title('PCA - Dimensionality Reduction (2 components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Faulty (0: No, 1: Yes)')
plt.show()


# In[10]:





# In[ ]:




