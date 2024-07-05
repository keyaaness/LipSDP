#!/usr/bin/env python
# coding: utf-8

# # Testing LipSDP on CNN Architectures 

# ***Aim***: To test the LipSDP Algorithm on various CNN models to show that the Lipschitz bound obtained via LipSDP is tighter than the one obtained trivially. 
# I.e. the Lipschitz constant obtained via LipSDP is smaller than the trivial Lipschitz constant (despite LipSDP not being informed about the existence of a trivial Lipschitz constant in any scenario).

# In[2]:


# importing necessary libraries

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torchvision.models as models
from sklearn.decomposition import PCA


# ## Defining important functions:

# In[3]:


# function to compute the trivial Lipschitz constant (multiplication of Spectral Norm of the weight matrices)

def trivial_lipschitz_constant(W0, W1):
    # computes 2-norm also known as Spectral norm
    norm_W0 = np.linalg.norm(W0, ord=2)
    norm_W1 = np.linalg.norm(W1, ord=2)
    lipschitz_constant = norm_W0 * norm_W1
    return lipschitz_constant


# In[4]:


# function to compute the Lipschitz constant using LipSDP

def lipsdp_lipschitz_constant(W0, W1, alpha, beta):
    
    n = W0.shape[1]  # number of columns in W0 (hidden layer size)
    m = W1.shape[0]  # number of rows in W1 (number of neurons in the hidden layer)

    # defining the decision variable
    rho = cp.Variable(nonneg=True)
    diag_entries = cp.Variable(m, nonneg=True)
    Tm = cp.diag(diag_entries)

    # defining the matrix inequality M(ρ, Tm)
    M_upper_left = -2 * alpha * beta * (W0.T @ Tm @ W0) - rho * np.eye(W0.shape[0])
    M_upper_right = (alpha + beta) * (W0.T @ Tm)
    M_lower_left = (alpha + beta) * (Tm @ W0)
    M_lower_right = -2 * Tm + W1.T @ W1

    # constructing the block matrix M(ρ, Tm)
    M = cp.bmat([
        [M_upper_left, M_upper_right],
        [M_lower_left, M_lower_right]
    ])

    # defining the constraints
    # M(ρ, Tm) is negative semidefinite and Tm has only positive diagonal entries
    constraints = [M << 0, diag_entries >= 0]

    # defining the objective function
    objective = cp.Minimize(rho)

    # defining the problem
    problem = cp.Problem(objective, constraints)
    
    # solving the problem
    problem.solve(solver=cp.SCS)  
    
    if problem.status not in ["infeasible", "unbounded"]:
        lipschitz_constant = np.sqrt(rho.value)
        return lipschitz_constant
    else:
        return None
    
#setting alpha and beta considering the ReLU activation function is slope restricted on [0, 1]
alpha = 0.0
beta = 1.0


# In[5]:


# function to transform a 4-D weight matrix to a 2-D weight matrix

def weights_4d_to_2d(weights_4d):
    out_channels, in_channels, k_h, k_w = weights_4d.shape
    weight_2d = np.zeros((out_channels * k_h * k_w, in_channels * k_h * k_w))
    
    for i in range(out_channels):
        for j in range(in_channels):
            weight_block = weights_4d[i, j, :, :].flatten()
            weight_2d[i*k_h*k_w:(i+1)*k_h*k_w, j*k_h*k_w:(j+1)*k_h*k_w] = np.diag(weight_block)
    
    return weight_2d


# In[6]:


# function to reduce the 2-D weight matrix (NumPy array) size using PCA and transforming it to a PyTorch tensor

def reduce_size(weight, new_dimensions):
    pca = PCA(n_components=new_dimensions)
    reduced_weight = pca.fit_transform(weight)
    reduced_weight = torch.tensor(reduced_weight)
    return reduced_weight


# In[8]:


# load the pre-trained CNN models to be tested

# AlexNet
alexnet = models.alexnet(weights=True)

# VGG-16
vgg16 = models.vgg16(weights=True)

# VGG-19
vgg19 = models.vgg19(weights=True)

# GoogleNet
googlenet = models.googlenet(weights=True)

# ResNet-18
resnet18 = models.resnet18(weights=True)

# DenseNet-121
densenet121 = models.densenet121(weights=True)

# SqueezeNet
squeezenet = models.squeezenet1_1(weights=True)


# ## Testing LipSDP on AlexNet

# In[7]:


# studying the model architecture
print(alexnet)


# In[8]:


# accessing the first fully connected layer
fc6 = alexnet.classifier[1]

# extracting the weight matrix W1 as a NumPy array
W1 = fc6.weight.data.numpy()
print("Weight matrix W1 shape:", W1.shape)


# In[9]:


# accessing the layer before the first fully connected layer (5th convolutional layer)
conv5 = alexnet.features[10]

# extracting the weight matrix W0 as a NumPy array
W0 = conv5.weight.data.numpy()
print("Weight matrix W0 shape:", W0.shape)


# In[10]:


# since W0 is 4-D, we convert it to a 2-D NumPy array

W0 = weights_4d_to_2d(W0)
print("Weight matrix W0 shape:", W0.shape)


# In[11]:


# reducing the size of W0 to a 64*64 PyTorch tensor

W0 = reduce_size(W0, 64)
print("Weight matrix W0 shape:", W0.shape)
W0_transpose = reduce_size(W0.T, 64)
W0 = W0_transpose.T
print("Weight matrix W0 shape:", W0.shape)


# In[12]:


# reducing the size of W1 to a 64*64 PyTorch tensor

W1 = reduce_size(W1, 64)
print("Weight matrix W1 shape:", W1.shape)
W1_transpose = reduce_size(W1.T, 64)
W1 = W1_transpose.T
print("Weight matrix W1 shape:", W1.shape)


# In[13]:


# printing the trivial Lipschitz constant of the first fully connected layer

trivial_lip_constant = trivial_lipschitz_constant(W0, W1)
print(f"Trivial Lipschitz constant of the first fully connected layer of AlexNet: {trivial_lip_constant}")


# In[14]:


# printing the Lipschitz constant obtained via LipSDP of the first fully connected layer

lipsdp_lip_constant = lipsdp_lipschitz_constant(W0, W1, alpha, beta)
print(f"Lipschitz constant obtained via LipSDP of the first fully connected layer of AlexNet: {lipsdp_lip_constant}")


# ## Testing LipSDP on VGG-16

# In[15]:


# studying the model architecture
print(vgg16)


# In[16]:


# accessing the first fully connected layer
fc = vgg16.classifier[0]

# extracting the weight matrix W1 as a NumPy array
W1 = fc.weight.data.numpy()
print("Weight matrix W1 shape:", W1.shape)


# In[17]:


# accessing the layer before the first fully connected layer (5th convolutional layer)
conv5 = vgg16.features[28]

# extracting the weight matrix W0 as a NumPy array
W0 = conv5.weight.data.numpy()
print("Weight matrix W0 shape:", W0.shape)


# In[18]:


# since W0 is 4-D, we convert it to a 2-D NumPy array

W0 = weights_4d_to_2d(W0)
print("Weight matrix W0 shape:", W0.shape)


# In[19]:


# reducing the size of W0 to a 64*64 PyTorch tensor

W0 = reduce_size(W0, 64)
print("Weight matrix W0 shape:", W0.shape)
W0_transpose = reduce_size(W0.T, 64)
W0 = W0_transpose.T
print("Weight matrix W0 shape:", W0.shape)


# In[20]:


# reducing the size of W1 to a 64*64 PyTorch tensor

W1 = reduce_size(W1, 64)
print("Weight matrix W1 shape:", W1.shape)
W1_transpose = reduce_size(W1.T, 64)
W1 = W1_transpose.T
print("Weight matrix W1 shape:", W1.shape)


# In[21]:


# printing the trivial Lipschitz constant of the first fully connected layer

trivial_lip_constant = trivial_lipschitz_constant(W0, W1)
print(f"Trivial Lipschitz constant of the first fully connected layer of VGG-16: {trivial_lip_constant}")


# In[22]:


# printing the Lipschitz constant obtained via LipSDP of the first fully connected layer

lipsdp_lip_constant = lipsdp_lipschitz_constant(W0, W1, alpha, beta)
print(f"Lipschitz constant obtained via LipSDP of the first fully connected layer of VGG-16: {lipsdp_lip_constant}")


# ## Testing LipSDP on VGG-19

# In[23]:


# studying the model architecture
print(vgg19)


# In[24]:


# accessing the first fully connected layer
fc = vgg19.classifier[0]

# extracting the weight matrix W1 as a NumPy array
W1 = fc.weight.data.numpy()
print("Weight matrix W1 shape:", W1.shape)


# In[25]:


# accessing the layer before the first fully connected layer 
conv = vgg19.features[34]

# extracting the weight matrix W0 as a NumPy array
W0 = conv.weight.data.numpy()
print("Weight matrix W0 shape:", W0.shape)


# In[26]:


# since W0 is 4-D, we convert it to a 2-D NumPy array

W0 = weights_4d_to_2d(W0)
print("Weight matrix W0 shape:", W0.shape)


# In[27]:


# reducing the size of W0 to a 64*64 PyTorch tensor

W0 = reduce_size(W0, 64)
print("Weight matrix W0 shape:", W0.shape)
W0_transpose = reduce_size(W0.T, 64)
W0 = W0_transpose.T
print("Weight matrix W0 shape:", W0.shape)


# In[28]:


# reducing the size of W1 to a 64*64 PyTorch tensor

W1 = reduce_size(W1, 64)
print("Weight matrix W1 shape:", W1.shape)
W1_transpose = reduce_size(W1.T, 64)
W1 = W1_transpose.T
print("Weight matrix W1 shape:", W1.shape)


# In[29]:


# printing the trivial Lipschitz constant of the first fully connected layer

trivial_lip_constant = trivial_lipschitz_constant(W0, W1)
print(f"Trivial Lipschitz constant of the first fully connected layer of VGG-19: {trivial_lip_constant}")


# In[30]:


# printing the Lipschitz constant obtained via LipSDP of the first fully connected layer

lipsdp_lip_constant = lipsdp_lipschitz_constant(W0, W1, alpha, beta)
print(f"Lipschitz constant obtained via LipSDP of the first fully connected layer of VGG-19: {lipsdp_lip_constant}")


# ## Testing LipSDP on GoogleNet

# In[31]:


# studying the model architecture
print(googlenet)


# In[34]:


# accessing the first fully connected layer
fc1 = googlenet.fc

# extracting the weight matrix W1 as a NumPy array
W1 = fc1.weight.data.numpy()
print(W1.shape)


# In[45]:


# accessing the layer before the first fully connected layer 
conv = googlenet.inception5b.branch4[1].conv

# extracting the weight matrix W0 as a NumPy array
W0 = conv.weight.data.numpy()
print("Weight matrix W0 shape:", W0.shape)


# In[46]:


# since W0 is 4-D, we convert it to a 2-D NumPy array

W0 = weights_4d_to_2d(W0)
print("Weight matrix W0 shape:", W0.shape)


# In[47]:


# reducing the size of W0 to a 64*64 PyTorch tensor

W0 = reduce_size(W0, 64)
print("Weight matrix W0 shape:", W0.shape)
W0_transpose = reduce_size(W0.T, 64)
W0 = W0_transpose.T
print("Weight matrix W0 shape:", W0.shape)


# In[48]:


# reducing the size of W1 to a 64*64 PyTorch tensor

W1 = reduce_size(W1, 64)
print("Weight matrix W1 shape:", W1.shape)
W1_transpose = reduce_size(W1.T, 64)
W1 = W1_transpose.T
print("Weight matrix W1 shape:", W1.shape)


# In[50]:


# printing the trivial Lipschitz constant of the first fully connected layer

trivial_lip_constant = trivial_lipschitz_constant(W0, W1)
print(f"Trivial Lipschitz constant of the first fully connected layer of GoogleNet: {trivial_lip_constant}")


# In[51]:


# printing the Lipschitz constant obtained via LipSDP of the first fully connected layer

lipsdp_lip_constant = lipsdp_lipschitz_constant(W0, W1, alpha, beta)
print(f"Lipschitz constant obtained via LipSDP of the first fully connected layer of GoogleNet: {lipsdp_lip_constant}")


# ## Testing LipSDP on ResNet-18

# In[52]:


# studying the model architecture
print(resnet18)


# In[9]:


# accessing the first fully connected layer
fc1 = resnet18.fc

# extracting the weight matrix W1 as a NumPy array
W1 = fc1.weight.data.numpy()
print(W1.shape)


# In[10]:


# accessing the layer before the first fully connected layer 
conv = resnet18.layer4[1].conv1

# extracting the weight matrix W0 as a NumPy array
W0 = conv.weight.data.numpy()
print("Weight matrix W0 shape:", W0.shape)


# In[11]:


# since W0 is 4-D, we convert it to a 2-D NumPy array

W0 = weights_4d_to_2d(W0)
print("Weight matrix W0 shape:", W0.shape)


# In[12]:


# reducing the size of W0 to a 64*64 PyTorch tensor

W0 = reduce_size(W0, 64)
print("Weight matrix W0 shape:", W0.shape)
W0_transpose = reduce_size(W0.T, 64)
W0 = W0_transpose.T
print("Weight matrix W0 shape:", W0.shape)


# In[13]:


# reducing the size of W1 to a 64*64 PyTorch tensor

W1 = reduce_size(W1, 64)
print("Weight matrix W1 shape:", W1.shape)
W1_transpose = reduce_size(W1.T, 64)
W1 = W1_transpose.T
print("Weight matrix W1 shape:", W1.shape)


# In[14]:


# printing the trivial Lipschitz constant of the first fully connected layer

trivial_lip_constant = trivial_lipschitz_constant(W0, W1)
print(f"Trivial Lipschitz constant of the first fully connected layer of ResNet-18: {trivial_lip_constant}")


# In[15]:


# printing the Lipschitz constant obtained via LipSDP of the first fully connected layer

lipsdp_lip_constant = lipsdp_lipschitz_constant(W0, W1, alpha, beta)
print(f"Lipschitz constant obtained via LipSDP of the first fully connected layer of ResNet-18: {lipsdp_lip_constant}")


# ## Testing LipSDP on DenseNet-121

# In[60]:


# studying the model architecture
print(densenet121)


# In[16]:


# accessing the first fully connected layer
fc1 = densenet121.classifier

# extracting the weight matrix W1 as a NumPy array
W1 = fc1.weight.data.numpy()
print(W1.shape)


# In[17]:


# accessing the layer before the first fully connected layer 
conv = densenet121.features.denseblock4.denselayer16.conv1

# extracting the weight matrix W0 as a NumPy array
W0 = conv.weight.data.numpy()
print("Weight matrix W0 shape:", W0.shape)


# In[18]:


# since W0 is 4-D, we convert it to a 2-D NumPy array

W0 = weights_4d_to_2d(W0)
print("Weight matrix W0 shape:", W0.shape)


# In[19]:


# reducing the size of W0 to a 64*64 PyTorch tensor

W0 = reduce_size(W0, 64)
print("Weight matrix W0 shape:", W0.shape)
W0_transpose = reduce_size(W0.T, 64)
W0 = W0_transpose.T
print("Weight matrix W0 shape:", W0.shape)


# In[20]:


# reducing the size of W1 to a 64*64 PyTorch tensor

W1 = reduce_size(W1, 64)
print("Weight matrix W1 shape:", W1.shape)
W1_transpose = reduce_size(W1.T, 64)
W1 = W1_transpose.T
print("Weight matrix W1 shape:", W1.shape)


# In[21]:


# printing the trivial Lipschitz constant of the first fully connected layer

trivial_lip_constant = trivial_lipschitz_constant(W0, W1)
print(f"Trivial Lipschitz constant of the first fully connected layer of DenseNet-121: {trivial_lip_constant}")


# In[22]:


# printing the Lipschitz constant obtained via LipSDP of the first fully connected layer

lipsdp_lip_constant = lipsdp_lipschitz_constant(W0, W1, alpha, beta)
print(f"Lipschitz constant obtained via LipSDP of the first fully connected layer of DenseNet-121: {lipsdp_lip_constant}")


# ## Testing LipSDP on SqueezeNet

# In[10]:


# studying the model architecture
print(squeezenet)


# In[24]:


# accessing the first fully connected layer
fc1 = squeezenet.classifier[1]

# extracting the weight matrix W1 as a NumPy array
W1 = fc1.weight.data.numpy()
print(W1.shape)


# In[25]:


# accessing the layer before the first fully connected layer 
conv = squeezenet.features[12].squeeze

# extracting the weight matrix W0 as a NumPy array
W0 = conv.weight.data.numpy()
print("Weight matrix W0 shape:", W0.shape)


# In[26]:


# since W0 is 4-D, we convert it to a 2-D NumPy array

W0 = weights_4d_to_2d(W0)
print("Weight matrix W0 shape:", W0.shape)


# In[27]:


# since W1 is 4-D, we convert it to a 2-D NumPy array

W1 = weights_4d_to_2d(W1)
print("Weight matrix W1 shape:", W1.shape)


# In[28]:


# reducing the size of W0 to a 64*64 PyTorch tensor

W0 = reduce_size(W0, 64)
print("Weight matrix W0 shape:", W0.shape)
W0_transpose = reduce_size(W0.T, 64)
W0 = W0_transpose.T
print("Weight matrix W0 shape:", W0.shape)


# In[29]:


# reducing the size of W1 to a 64*64 PyTorch tensor

W1 = reduce_size(W1, 64)
print("Weight matrix W1 shape:", W1.shape)
W1_transpose = reduce_size(W1.T, 64)
W1 = W1_transpose.T
print("Weight matrix W1 shape:", W1.shape)


# In[30]:


# printing the trivial Lipschitz constant of the first fully connected layer

trivial_lip_constant = trivial_lipschitz_constant(W0, W1)
print(f"Trivial Lipschitz constant of the first fully connected layer of SqueezeNet: {trivial_lip_constant}")


# In[31]:


# printing the Lipschitz constant obtained via LipSDP of the first fully connected layer

lipsdp_lip_constant = lipsdp_lipschitz_constant(W0, W1, alpha, beta)
print(f"Lipschitz constant obtained via LipSDP of the first fully connected layer of SqueezeNet: {lipsdp_lip_constant}")


# **Conclusion**: The Lipschitz constant obtained via LipSDP was smaller than the trivial Lipschitz constant which points out the fact that LipSDP provides a tighter estimation of the Lipschitz constant.
