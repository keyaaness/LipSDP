#!/usr/bin/env python
# coding: utf-8

# # LipSDP improved: Encoding ***Monotonocity*** of ReLU for a single layer NN

# In[1]:


# importing necessary libraries
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# ### Trivial Lipschitz constant:

# In[2]:


def trivial(W0, W1):
    
    # computes 2-norm also known as Spectral norm
    
    norm_W0 = np.linalg.norm(W0, ord=2)
    norm_W1 = np.linalg.norm(W1, ord=2)
    lipschitz_constant = norm_W0 * norm_W1
    
    return lipschitz_constant


# ### Original LipSDP algorithm:

# In[3]:


def lipsdp(W0, W1, alpha, beta):
    
    n = W0.shape[1]  # number of columns in W0 (hidden layer size)
    m = W1.shape[0]  # number of rows in W1 (number of neurons in the hidden layer)

    # defining the decision variables
    rho = cp.Variable(nonneg=True)
    diag_entries = cp.Variable(m, nonneg=True)
    T1 = cp.diag(diag_entries)
    
    # defining the matrix inequality M(ρ, Tm)
    M_upper_left = -2 * alpha * beta * (W0.T @ T1 @ W0) - rho * np.eye(W0.shape[0])
    M_upper_right = (alpha + beta) * (W0.T @ T1)
    M_lower_left = (alpha + beta) * (T1 @ W0)
    M_lower_right = -2 * T1 + W1.T @ W1

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


# ## Monotonocity of ReLU and expressing using quadratic constraints:
# 
# 
# Monotonocity condition expressed as an inequality: $(x-y)({\phi}(x) -{\phi}(y)) \geq 0$
# 
# This inequality when expressed using quadratic constraints translates as: 
#  $ \begin{bmatrix} x-y \\ {\phi}(x) -{\phi}(y)  \end{bmatrix} ^\top \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} x-y \\ {\phi}(x) -{\phi}(y)  \end{bmatrix} \geq 0 \tag{a}$

# ## How to encode this into the LipSDP algorithm? 
# The setup is as following: 
# 
# 

# ![Screenshot%20%28275%29.png](attachment:Screenshot%20%28275%29.png)

# ![Screenshot%20%28252%29.png](attachment:Screenshot%20%28252%29.png)

# Details of the theorem and it's backgroud can be found in this paper: https://arxiv.org/abs/1906.04893v2
# [Fazlyab, M., Robey, A., Hassani, H., Morari, M., & Pappas, G. J. (2019). Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks. ArXiv. /abs/1906.04893]

# ![Screenshot%20%28273%29.png](attachment:Screenshot%20%28273%29.png)

# ### We suggest two ways:
# 
# 1. We introduce $T_{1}$, $T_{2}$ and a scalar quantity $c$. Here, $T_{1}$ represents $T$ in the original theorem while we set $T_{2} := I_{n}$ and  transform $(a)$ similar to $(17)$ using $T_{2}$ and take a linear combination of the two matrix inequalities i.e. $(18)$ and the one obtained from expressing monotonocity as a matrix inequality using quadratic constraints with the coefficients $1$ and $c$ respectively for each of them. This gives us the new matrix $M_{1}(\rho, T)$.
# 2. Here we scrap the $c$ introduced above and use the same form for $T_{1}$ and $T_{2}$ as expressed in $(8)$ and do perform the same operations mentioned above. This gives us the new matrix $M_{2}(\rho, T)$.

# ### 1st variant of LipSDP:

# In[4]:


def lipsdp1(W0, W1, alpha, beta):
    
    n = W0.shape[1]  # number of columns in W0 (hidden layer size)
    m = W1.shape[0]  # number of rows in W1 (number of neurons in the hidden layer)

    # defining the decision variables
    rho = cp.Variable(nonneg=True)
    diag_entries = cp.Variable(m, nonneg=True)
    T1 = cp.diag(diag_entries)
    c = cp.Variable(nonneg=True)
    
    # defining the matrix inequality M(ρ, Tm)
    # here, c has to be +ve
    M1_upper_left = -2 * alpha * beta * (W0.T @ T1 @ W0) - rho * np.eye(W0.shape[0])
    M1_upper_right = (alpha + beta) * (W0.T @ T1) + c * W0.T
    M1_lower_left = (alpha + beta) * (T1 @ W0) + c * W0
    M1_lower_right = -2 * T1 + W1.T @ W1

    # constructing the block matrix M(ρ, Tm)
    M1 = cp.bmat([
        [M1_upper_left, M1_upper_right],
        [M1_lower_left, M1_lower_right]
    ])

    # defining the constraints
    # M(ρ, Tm) is negative semidefinite and Tm has only positive diagonal entries
    constraints = [M1 << 0, diag_entries >= 0]

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


# ### 2nd variant of LipSDP:

# In[5]:


def lipsdp2(W0, W1, alpha, beta):
    
    n = W0.shape[1]  # number of columns in W0 (hidden layer size)
    m = W1.shape[0]  # number of rows in W1 (number of neurons in the hidden layer)

    # defining the decision variables
    rho = cp.Variable(nonneg=True)
    diag_entries1 = cp.Variable(m, nonneg=True)
    diag_entries2 = cp.Variable(m, nonneg=True)
    T1 = cp.diag(diag_entries1)
    T2 = cp.diag(diag_entries2)
    
    # defining the matrix inequality M(ρ, Tm)
    M2_upper_left = -2 * alpha * beta * (W0.T @ T1 @ W0) - rho * np.eye(W0.shape[0])
    M2_upper_right = (alpha + beta) * (W0.T @ T1) + (W0.T @ T2)
    M2_lower_left = (alpha + beta) * (T1 @ W0) + (T2 @ W0)
    M2_lower_right = -2 * T1 + W1.T @ W1

    # constructing the block matrix M(ρ, Tm)
    M2 = cp.bmat([
        [M2_upper_left, M2_upper_right],
        [M2_lower_left, M2_lower_right]
    ])

    # defining the constraints
    # M(ρ, Tm) is negative semidefinite and Tm has only positive diagonal entries
    constraints = [M2 << 0, diag_entries1 >= 0, diag_entries2 >= 0]

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


# In[6]:


alpha = 0.0  # setting alpha = 0
beta = 1.0   # setting beta = 1


# In[7]:


# test case 1: 

# weight matrices are 2*2 identity matrices
W0 = np.array([[1, 0], [0, 1]]) 
W1 = np.array([[1, 0], [0, 1]])

# computing the trivial Lipschitz constant
trivial_constant = trivial(W0, W1)

# computing the LipSDP constants
print(f"Trivial Lipschitz constant: {trivial_constant}")
print("Running LipSDP...")
print("Lipschitz constant computed by deploying LipSDP:")
lipsdp_constant = lipsdp(W0, W1, alpha, beta)
print(f"LipSDP Lipschitz constant: {lipsdp_constant}")
lipsdp1_constant = lipsdp1(W0, W1, alpha, beta)
print(f"LipSDP1 Lipschitz constant: {lipsdp1_constant}")
lipsdp2_constant = lipsdp2(W0, W1, alpha, beta)
print(f"LipSDP2 Lipschitz constant: {lipsdp2_constant}")


# In[8]:


# test case 2:

# weight matrices are 3*3 identity matrices
W0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 
W1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 

# computing the trivial Lipschitz constant
trivial_constant = trivial(W0, W1)

# computing the LipSDP constants
print(f"Trivial Lipschitz constant: {trivial_constant}")
print("Running LipSDP...")
print("Lipschitz constant computed by deploying LipSDP:")
lipsdp_constant = lipsdp(W0, W1, alpha, beta)
print(f"LipSDP Lipschitz constant: {lipsdp_constant}")
lipsdp1_constant = lipsdp1(W0, W1, alpha, beta)
print(f"LipSDP1 Lipschitz constant: {lipsdp1_constant}")
lipsdp2_constant = lipsdp2(W0, W1, alpha, beta)
print(f"LipSDP2 Lipschitz constant: {lipsdp2_constant}")


# In[9]:


# test case 3:

# weight matrices are 4*4 identity matrices
W0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) 
W1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) 

# computing the trivial Lipschitz constant
trivial_constant = trivial(W0, W1)

# computing the LipSDP constants
print(f"Trivial Lipschitz constant: {trivial_constant}")
print("Running LipSDP...")
print("Lipschitz constant computed by deploying LipSDP:")
lipsdp_constant = lipsdp(W0, W1, alpha, beta)
print(f"LipSDP Lipschitz constant: {lipsdp_constant}")
lipsdp1_constant = lipsdp1(W0, W1, alpha, beta)
print(f"LipSDP1 Lipschitz constant: {lipsdp1_constant}")
lipsdp2_constant = lipsdp2(W0, W1, alpha, beta)
print(f"LipSDP2 Lipschitz constant: {lipsdp2_constant}")


# ### Evaluating the above two LipSDP algorithm variants on encoding the monotonocity constraints: 

# In[10]:


# checking the number of testcases where the variants give a tighter estimation on ->

# iterating over 1000 random testcases
iter = 1000
lip1_lip = 0 # counter for when LipSDP1 gives tighter estimation than LipSDP
lip2_lip = 0 # counter for when LipSDP2 gives tighter estimation than LipSDP

for iteration in range(iter):
    
    # random normalized 2*2 weight matrices
    W0 = np.random.randint(0, 1, size=(2, 2))   
    W1 = np.random.randint(0, 1, size=(2, 2))
    
    # computing the trivial Lipschitz constant
    trivial_const = trivial(W0, W1)
    
    # computing LipSDP Lipschitz constants and evaluating the results
    lip = lipsdp(W0, W1, alpha, beta)
    lip1 = lipsdp1(W0, W1, alpha, beta)
    if lip1<lip:
        lip1_lip+=1
    lip2 = lipsdp2(W0, W1, alpha, beta)
    if lip2<lip:
        lip2_lip+=1
    
print(f"%age of testcases where LipSDP1 a gives tighter bound than LipSDP : {(lip1_lip/iter)*100}%")
print(f"%age of testcases where LipSDP2 a gives tighter bound than LipSDP : {(lip2_lip/iter)*100}%")


# **Conclusion:** Both the variants of LipSDP work better than the original LipSDP algorithm on normalized weight matrices by giving a tighter bound for Lipschitz constant estimation.
