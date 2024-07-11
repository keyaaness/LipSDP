#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing necessary libraries

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torchvision.models as models


# ## Power Iteration Algorithm

# In[3]:


# function for the Power Iteration Algorithm

def power_iteration(G, num_iter):
    u = np.random.rand(G.shape[1])
    u = u / np.linalg.norm(u)
    
    # Power Iteration
    for _ in range(num_iter):
        u = np.dot(G, u)
        u = u / np.linalg.norm(u)
    
    # calculating the eigenvalue
    sigma_1 = np.dot(u, np.dot(G, u)) / np.dot(u, u)
    # returning the eigenvalue and eigenvector
    return sigma_1, u


# ## Gram Iteration Algorithm

# Gram Iteration is an itertive algorithm as described in [Delattre, Blaise, et al. "Efficient bound of Lipschitz constant for convolutional layers by gram iteration." International Conference on Machine Learning. PMLR, 2023.] (http://proceedings.mlr.press/v202/delattre23a.html). It gives us an upper bound on the Lipschitz constant (spectral norm) of a neural network layer taking into consideration a single layer's weight matrix. This algorithm exhibits superlinear convergence.

# In[4]:


# function for the Gram Iteration Algorithm

def gram_iteration(G, N_iter):
    # initializing rescaling
    r = 0
    G_norm = np.linalg.norm(G, ord='fro')
    for _ in range(N_iter):
        # rescaling to avoid overflow
        G = G / np.linalg.norm(G, ord='fro')
        # gram iteration
        G = G @ G.T
        # cumulate rescaling
        r = 2*(r + np.log(np.linalg.norm(G, ord='fro')))
    # computing final result
    sigma_1 = np.linalg.norm(G, ord=2)**(1/(2**N_iter)) * np.exp(r / (2**N_iter))* G_norm * G_norm
    
    return sigma_1


# ## Actual Spectral Norm

# In[5]:


# function to calculate the actual spectral norm

def spectral_norm(G):
    return np.linalg.norm(G, ord=2)


# ### Testing the above iterative algorithms:

# In[6]:


# identity matrix test case
G = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
num_iter = 1000
spectral_norm_trivial = spectral_norm(G)
print("Actual Spectral Norm: ", spectral_norm_trivial)
eigenvalue, eigenvector = power_iteration(G, num_iter)
print("Largest Eigenvalue found using Power Iteration:", eigenvalue)
spectral_norm_gi = gram_iteration(G, num_iter)
print("Spectral Norm found using Gram Iteration:", np.sqrt(spectral_norm_gi))


# In[7]:


# randomly initialized G
G = np.random.rand(2, 2)
num_iter = 1000
spectral_norm_trivial = spectral_norm(G)
print("Actual Spectral Norm: ", spectral_norm_trivial)
eigenvalue, eigenvector = power_iteration(G, num_iter)
print("Largest Eigenvalue found using Power Iteration:", eigenvalue)
spectral_norm_gi = gram_iteration(G, num_iter)
print("Spectral Norm found using Gram Iteration:", np.sqrt(spectral_norm_gi))


# ## Testing Gram Iteration AlexNet

# In[10]:


# AlexNet
alexnet = models.alexnet(weights=True)


# In[11]:


# accessing the first fully connected layer
fc6 = alexnet.classifier[1]

# extracting the weight matrix G as a NumPy array
G = fc6.weight.data.numpy()
print("Weight matrix G shape:", G.shape)


# In[12]:


num_iter = 10
spectral_norm_gi = gram_iteration(G, num_iter)
print("Spectral Norm using Gram Iteration:", np.sqrt(spectral_norm_gi))
spectral_norm_actual = spectral_norm(G)
print("Actual Spectral Norm:", spectral_norm_actual)


# ## Testing Gram Iteration GoogleNet

# In[14]:


# GoogleNet
googlenet = models.googlenet(weights=True)


# In[15]:


# accessing the first fully connected layer
fc1 = googlenet.fc

# extracting the weight matrix G as a NumPy array
G = fc1.weight.data.numpy()
print(G.shape)


# In[16]:


num_iter = 10
spectral_norm_gi = gram_iteration(G, num_iter)
print("Spectral Norm using Gram Iteration:", np.sqrt(spectral_norm_gi))
spectral_norm_actual = spectral_norm(G)
print("Actual Spectral Norm:", spectral_norm_actual)

