#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
from scipy.io import loadmat,savemat


# In[21]:


# READING EXISTING FILE
example='/work/Camelyon17/work/DECONVOLUCIONES/BKSVD_MB/center_0/patient_000_node_0/no_annotated/patient_000_node_0_xini_133804_yini_73696.Results.mat'

f = h5py.File(example, 'r')
CT=f['stains']


# In[8]:


print(f.keys())


# In[9]:


print(CT.shape)


# In[13]:


plt.subplot(1,2,1), plt.imshow(CT[0,:,:]),plt.title('H')
plt.subplot(1,2,2), plt.imshow(CT[1,:,:]),plt.title('E')


# In[18]:


#SAVING A NEW FILE
save_file= '/work/work_fperez/My_Deep_BCD/new_example.mat'

h5f = h5py.File(save_file, 'w')
h5f.create_dataset('stains', data=CT)
# CT is the OD image of H and E with format 2x224x224


# In[19]:


glob.glob(save_file)


# In[20]:


#READING THE NEW CREATED FILE TO CHECK IT WAS CORRECT
f = h5py.File(save_file, 'r')
CT2=f['stains']
plt.subplot(1,2,1), plt.imshow(CT2[0,:,:]),plt.title('H')
plt.subplot(1,2,2), plt.imshow(CT2[1,:,:]),plt.title('E')


# In[28]:


################################### HOW TO SAVE M #####################################
M_file='/work/Camelyon17/work/DECONVOLUCIONES/BKSVD_MB/center_0/patient_000_node_0/M.mat'

M=loadmat(M_file)
print(M.keys())
M=M['M']
print(M.shape) #M should be size 3x2 (one column for H and one for E)

save_M='/work/work_fperez/My_Deep_BCD/new_example_M.mat'
savemat(save_M,{'M':M})

