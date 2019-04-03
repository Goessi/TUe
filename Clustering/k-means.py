# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:47:33 2019

@author: JingQIN
"""

'''
visual k-means step
'''

import numpy as np
import matplotlib.pyplot as plt

def d(c1, c2):
    diff = c1 - c2
    return diff.dot(diff)

def cost(X, R ,M):
    cost = 0
    for k in range(len(M)):
        for n in range(len(X)):
            cost += R[n, k]*d(M[k], X[n])
    return cost

def plot_k_means(X, K, max_iter=20, beta=10):
    N, D = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))
    
    for k in range(K):
        M[k] = X[np.random.choice(N)]
        
    grid_width = 5
    grid_height = max_iter / grid_width
    random_colors = np.random.random((K, 3))
    plt.figure()
    
    plt.title("Process")    
    costs = np.zeros(max_iter)
    for i in range(max_iter):
        colors = R.dot(random_colors)
        plt.subplot(grid_width, grid_height, i+1)
        plt.scatter(X[:, 0], X[:,1], c = colors)
        for k in range(K):
            for n in range(N):
                R[n, k] = np.exp(-beta * d(M[k], X[n])) / np.sum( np.exp(-beta * d(M[j], X[n])) for j in range(K))
            
        for k in range(K):
            M[k] = R[:, k].dot(X) / R[:, k].sum()
        
        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 0.1:
                break
    
    plt.savefig("Process.png")
    
    plt.figure()     
    plt.plot(costs)
    plt.title("Costs")
    plt.xlabel("Iterations")
    plt.ylabel("weighted arithmetic mean")
    plt.savefig("Costs.png")

    
    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.figure()
    plt.scatter(X[:, 0],X[:, 1], c=colors)
    plt.title("Outputs")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.savefig("outputs.png")


def generate_X():
    D = 2
    s = 4
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([s, 0])
    
    N = 900
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3
    
    return X
    
    

    

    
if __name__ == '__main__':
    X = generate_X()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Original data")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.savefig("Original_points.png")
    
#    K = 3
#    plot_k_means(X, K)
    
#    K = 5
#    plot_k_means(X, K)
    
#    K = 5
#    plot_k_means(X, K, beta = 0.3)
    
    K = 5
    plot_k_means(X, K, max_iter=30, beta = 0.3)