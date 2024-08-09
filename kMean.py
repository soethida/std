import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
x11 = 5 + 2*np.random.rand(50,1) - 2*np.random.rand(50,1) 
x12 = 5 + 2*np.random.rand(50,1) - 2*np.random.rand(50,1) 
X_1 = np.concatenate((x11, x12),axis = 1)
x21 = 7.5 + 2*np.random.rand(50,1) - 2*np.random.rand(50,1) 
x22 = 7.5 + 2*np.random.rand(50,1) - 2*np.random.rand(50,1)
X_2 = np.concatenate((x21, x22),axis = 1)
x31 = 10 + 2*np.random.rand(50,1) - 2*np.random.rand(50,1) 
x32 = 10 + 2*np.random.rand(50,1) - 2*np.random.rand(50,1)
X_3 = np.concatenate((x31, x32),axis = 1)


fig,ax = plt.subplots()
ax.plot(X_1[:,0],X_1[:,1],'bo')
ax.plot(X_2[:,0],X_2[:,1],'kx')
ax.plot(X_3[:,0],X_3[:,1],'m+')
ax.set_xlabel('X1')
ax.set_xlabel('X2')

X = np.concatenate((X_1, X_2, X_3),axis = 0)


fig,ax = plt.subplots()
ax.plot(X[:,0], X[:,1],'b.')
ax.set_xlabel('X1')
ax.set_xlabel('X2')

def initializeCGs(X, K):
    # Initialize CG as zeros
    CGs = np.zeros((K, X.shape[1]))

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as CGs
    CGs = X[randidx[:K], :]

    return CGs


nClusters = 3
mu = initializeCGs(X, nClusters)
mu

fig,ax = plt.subplots()
ax.plot(X[:,0], X[:,1],'b.')
ax.set_xlabel('X1')
ax.set_xlabel('X2')
colors = ['ro','kx','m+']
for (k,color) in zip(range(nClusters),colors):
    ax.plot(mu[k,0],mu[k,1],color,markersize = 10)


def clusterbyClosestCGs(X, CGs):
    # get number of clusters
    nClusters = CGs.shape[0]

    # Initialize idx. Index array size = (number of samples in dataset)
    idx = np.zeros(X.shape[0], dtype=int)

    # Iterate over each example in the dataset X
    for i in range(X.shape[0]):
        # Compute the squared Euclidean distance to each centroid
        d = np.sum((X[i] -  CGs) ** 2, axis=1)

        # Find the index of the closest centroid
        idx[i] = np.argmin(d) 

    return idx


idx = clusterbyClosestCGs(X,mu)
idx


fig,ax = plt.subplots()
colors = ['ro','kx','m+']
for (k,color) in zip(range(nClusters),colors):
    ax.plot(X[np.where(idx==k),0],X[np.where(idx==k),1],color)
ax.set_xlabel('X1')
ax.set_xlabel('X2')

def calculateCGs(X, idx, nClusters):
    # Get sample size and feature size
    m, n = X.shape

    # Initialize centroids matrix. 
    CGs = np.zeros((nClusters, n))

    # Loop over every cluster and compute the mean value of data samples in cluster k
    for i in range(nClusters):
        # Find indices of data samples assigned to cluster i
        index = np.where(idx == i)[0]

        # Compute the mean of the data samples assigned to cluster i
        if len(index) > 0:
            CGs[i - 1, :] = np.mean(X[index, :], axis=0)

    return CGs

mu_new = calculateCGs(X, idx, nClusters)
mu_new

fig,ax = plt.subplots()
ax.plot(X[:,0], X[:,1],'b.')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
colors = ['ro','kx','m+']
for (k,color) in zip(range(nClusters),colors):
    ax.plot(mu_new[k,0],mu_new[k,1],color,markersize = 10)

    
def runkMeans(X, initial_CGs, max_iters):
    # Initialize values
    m, n = X.shape
    nClusters = initial_CGs.shape[0]
    CGs = initial_CGs
    previous_CGs = CGs
    idx = np.zeros(m, dtype=int)

    # Run K-Means
    for i in range(max_iters):
        # Output progress
        print(f'K-Means iteration {i + 1}/{max_iters}...')
        
        # For each example in X, assign it to the closest centroid
        idx = clusterbyClosestCGs(X, CGs)

        # Given the memberships, compute new centroids
        CGs = calculateCGs(X, idx, nClusters)
    
    return CGs, idx

# Settings for running K-Means
max_iters = 10
nClusters = 3
initial_CGs = initializeCGs(X,nClusters)

CGs, idx = runkMeans(X, initial_CGs, max_iters)

fig,ax = plt.subplots()
colors = ['bo','kx','m+']
for (k,color) in zip(range(nClusters),colors):
    ax.plot(X[np.where(idx==k),0],X[np.where(idx==k),1],color)
ax.set_xlabel('X1')
ax.set_xlabel('X2')

