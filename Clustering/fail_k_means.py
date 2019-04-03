# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:54:07 2019

@author: JingQIN
"""

import numpy as np

def donut():
    N = 1000
    D = 2
    
    R_inner = 5
    R_outer = 10
    
    R1 = np.random.randn(N//2) + R_inner
    theta = 2 * np.pi * np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
    
    R2 = np.random.randn(N//2) + R_inner
    theta = 2 * np.pi * np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
    
    X = np.concatenate([X_inner, X_outer])
    return X

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
        plt.axis('equal')
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
    plt.axis('equal')
    plt.title("Outputs")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.savefig("outputs.png")

def main():
#    X = donut()
#    plt.plot(X)
#    plot_k_means(X, 2, beta = 0.1)
    
    X = np.zeros((1000, 2))
    X[:500, :] = np.random.multivariate_normal([0,0],[[1,0],[0,20]], 500)
    X[500:, :] = np.random.multivariate_normal([5,0],[[1,0],[0,20]], 500)
    plot_k_means(X, 2, beta = 0.1)
    
    X = np.zeros([1000, 2])
    X[:950, :] = np.array([0, 0]) + np.random.randn(950, 2)
    X[950:, :] = np.array([4, 0]) + np.random.randn(50, 2)
    plot_k_means(X, 2)
    
    
if __name__ == '__main__':
    main()