#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries
import cvxpy as cp
import numpy as np


# ### Trivial Lipschitz constant (product of spectral norm)

# In[2]:


def trivial(W0, W1, W2):
    
    # computes 2-norm also known as Spectral norm
    
    norm_W0 = np.linalg.norm(W0, ord=2)
    norm_W1 = np.linalg.norm(W1, ord=2)
    norm_W2 = np.linalg.norm(W2, ord=2)
    lipschitz_constant = norm_W0 * norm_W1 * norm_W2
    
    return lipschitz_constant


# ### Original LipSDP algorithm:

# In[3]:


def lipsdp_multilayer(W0, W1, W2):
    
    n0 = W0.shape[1]
    n1 = W1.shape[1]
    n2 = W2.shape[1]
    
    n = n1 + n2 # total number of hidden neurons
    
    # defining the decision variables
    rho = cp.Variable(nonneg=True)
    diag_entries = cp.Variable(n, nonneg=True)
    T = cp.diag(diag_entries)


    # constructing zero matrices with matching dimensions
    W0_zeros = cp.Constant(np.zeros((W0.shape[0], W0.shape[1])))
    W1_zeros = cp.Constant(np.zeros((W1.shape[0], W1.shape[1])))
    n0_zeros = cp.Constant(np.zeros((n0, n0)))
    n1_zeros = cp.Constant(np.zeros((n1, n1)))
    n2_zeros = cp.Constant(np.zeros((n2, n2)))
    
    # formulating the matrix M
    
    A = cp.bmat([[W0, W0_zeros , W0_zeros], [W1_zeros, W1, W1_zeros]])
    B = cp.bmat([[n1_zeros, np.eye(n1), n1_zeros], [n2_zeros, n2_zeros, np.eye(n2)]])
    P = cp.bmat([[-rho*np.eye(n0), n0_zeros, n0_zeros], [n1_zeros, n1_zeros, n1_zeros], [n2_zeros, n2_zeros, W2.T@W2]])
    
    X = cp.bmat([[A], [B]])
    M1 = cp.bmat([[-2 * alpha * beta * T, (alpha + beta) * T], [(alpha + beta) * T, - 2 * T]])
    M = X.T  @ M1 @ X + P
    
    # defining the constraints
    constraints = [M << 0, diag_entries >= 0]

    # defining the objective
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


# In[4]:


# since ReLU is slope restricted on [0, 1]
alpha = 0  # lower bound 
beta = 1   # upper bound


# In[5]:


# identity weight matrix test case
W0 = np.array([[1, 0], [0, 1]])
W1 = np.array([[1, 0], [0, 1]])
W2 = np.array([[1, 0], [0, 1]])
l = lipsdp_multilayer(W0, W1, W2)
print(f"Lipschitz constant is: {l}")


# In[6]:


# randomly initialized weight matrices
W0 = np.random.rand(2, 2)
W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 2)
l = lipsdp_multilayer(W0, W1, W2)
print(f"Lipschitz constant is: {l}")


# ## Improving LipSDP: Encoding ***Monotonocity*** of ReLU for a multi-layer NN

# In[7]:


def lipsdp_multilayer1(W0, W1, W2):
    
    n0 = W0.shape[1]
    n1 = W1.shape[1]
    n2 = W2.shape[1]
    
    n = n1 + n2 # total number of hidden neurons
    
    # defining the decision variables
    rho = cp.Variable(nonneg=True)
    diag_entries = cp.Variable(n, nonneg=True)
    T = cp.diag(diag_entries)
    
    diag_entries1 = cp.Variable(n0, nonneg=True)
    T1 = cp.diag(diag_entries1)
    diag_entries2 = cp.Variable(n1, nonneg=True)
    T2 = cp.diag(diag_entries2)
    diag_entries3 = cp.Variable(n1, nonneg=True)
    T3 = cp.diag(diag_entries3)
    diag_entries4 = cp.Variable(n2, nonneg=True)
    T4 = cp.diag(diag_entries4)


    # constructing zero matrices with matching dimensions
    W0_zeros = cp.Constant(np.zeros((W0.shape[0], W1.shape[1])))
    W1_zeros = cp.Constant(np.zeros((W1.shape[0], W1.shape[1])))
    n0_zeros = cp.Constant(np.zeros((n0, n0)))
    n1_zeros = cp.Constant(np.zeros((n1, n1)))
    n2_zeros = cp.Constant(np.zeros((n2, n2)))
    
    # formulating the matrix M
    
    A = cp.bmat([[W0, W0_zeros , W0_zeros], 
                 [W1_zeros, W1, W1_zeros]])
    B = cp.bmat([[n1_zeros, np.eye(n1), n1_zeros], 
                 [n2_zeros, n2_zeros, np.eye(n2)]])
    P = cp.bmat([[-rho*np.eye(n0), n0_zeros, n0_zeros], 
                 [n1_zeros, n1_zeros, n1_zeros], 
                 [n2_zeros, n2_zeros, W2.T@W2]])
    S = cp.bmat([[n0_zeros, T1, n0_zeros, n0_zeros], 
                 [n1_zeros, n1_zeros, n1_zeros, T2], 
                 [T3, n1_zeros, n1_zeros, n1_zeros], 
                 [n2_zeros, n2_zeros, T4, n2_zeros]])
                 
    
    X = cp.bmat([[A], [B]])
    M1 = cp.bmat([[-2 * alpha * beta * T, (alpha + beta) * T ], [(alpha + beta) * T , - 2 * T]])
    M = X.T  @ M1 @ X + P + X.T  @ S @ X
    
    # defining the constraints
    constraints = [M << 0, diag_entries >= 0, diag_entries1 >= 0, diag_entries2 >= 0, diag_entries3 >= 0, diag_entries4 >=0]

    # defining the objective
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


# In[8]:


# randomly initialized weight matrices
W0 = np.random.rand(2, 2)
W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 2)
l1 = lipsdp_multilayer(W0, W1, W2)
l2 = lipsdp_multilayer1(W0, W1, W2)
trivial_l = trivial(W0, W1, W2)
print(f"Trivial Lipschitz constant is: {trivial_l}")
print(f"Lipschitz constant using LipSDP is: {l1}")
print(f"Lipschitz constant by introducing monotonicity constraints is: {l2}") 


# In[12]:


# iterating over random testcases
iter = 100
lip_triv = 0 # counter for when Multi-layer LipSDP gives tighter estimation than trivial bound
lip1_triv = 0 # counter for when Multi-layer LipSDP1 gives tighter estimation than the trivial bound
lip1_lip = 0 # counter for when Multi-layer LipSDP1 gives tighter estimation than Multilayer LipSDP


for iteration in range(iter):
    
    # random normalized 2*2 weight matrices

    W0 = np.random.rand(2, 2)
    W1 = np.random.rand(2, 2)
    W2 = np.random.rand(2, 2)
    
    # computing LipSDP Lipschitz constants and evaluating the results
    l = lipsdp_multilayer(W0, W1, W2)
    l1 = lipsdp_multilayer1(W0, W1, W2)
    triv = trivial(W0, W1, W2)
    if l1<l:
        lip1_lip+=1
    if l<triv:
        lip_triv+=1
    if l1<triv:
        lip1_triv+=1
    
print(f"% of testcases where Multi-Layer LipSDP gives a tighter bound than the trivial one:{(lip_triv/iter)*100}")
print(f"% of testcases where Multi-Layer LipSDP1 gives a tighter bound than the trivial one:{(lip1_triv/iter)*100}")
print(f"% of testcases where Multi-Layer LipSDP1 gives a tighter bound than Multi-Layer LipSDP:{(lip1_lip/iter)*100}")


# In[ ]:




