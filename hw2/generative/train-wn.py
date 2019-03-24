import numpy as np
def covariance(x):
    N,K = x.shape
    mu = np.mean(x, axis = 0)
    sigma = np.zeros((K,K))
    for i in range(N):
        v = (x[i] - mu).reshape(K,1)
        sigma += np.matmul(v, v.reshape(1,K)) / N
    return sigma 
# preprocess data
x_data = np.genfromtxt('../X_train', delimiter=',')[1:,]
N, K  = x_data.shape
mean = np.zeros(K)
std = np.ones(K)
x_data = (x_data - mean) / std
y_data = np.genfromtxt('../Y_train', dtype=int)[1:,]
# 
x = [[],[]]
for i in range(N):
    x[y_data[i]].append(x_data[i])
for i in range(2):
    x[i] = np.array(x[i])
N0 = x[0].shape[0]
N1 = x[1].shape[0]
K = x_data.shape[1]
mu0 = np.mean(x[0], axis = 0)
mu1 = np.mean(x[1], axis = 0)
sigma = N0/N * np.cov(x[0],rowvar=False) + N1/N * np.cov(x[1],rowvar=False)
si = np.linalg.pinv(sigma)
w = (mu1 - mu0) @ si
b = -0.5 * mu1.reshape(1,K) @ si @ mu1 + 0.5 * mu0.reshape(1,K) @ si @ mu0 + np.log(N1/N0)
z = np.matmul(x_data, w) + b
fw = 1/(1+np.exp(-z))
accuracy = np.sum((fw > 0.5) == y_data) / N
print(accuracy)
np.savez('weight-wn', mean=mean, std=std, w=w,b=b)
