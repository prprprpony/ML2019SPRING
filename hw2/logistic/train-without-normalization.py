import numpy as np
# preprocess data
x_data = np.genfromtxt('../X_train', delimiter=',')[1:,]
N,K = x_data.shape
mean = np.zeros(K)
std = np.ones(K)
for i in range(N):
    for j in range(x_data.shape[1]):
        x_data[i][j] = (x_data[i][j] - mean[j]) / std[j]
x_data = np.concatenate((np.ones((N,1)),x_data),axis=1).astype(float)
y_data = np.genfromtxt('../Y_train', dtype=int)[1:,].astype(bool)
# 
w = np.zeros(x_data.shape[1])
lr = 1
T = 10**4+1
prev_gra = 0
xt = x_data.T
for t in range(T):
    z = np.matmul(x_data, w)
    fw = 1/(1+np.exp(-z))
    grad = np.matmul(xt, fw - y_data)
    prev_gra += np.sum(np.square(grad))
    ada = np.sqrt(prev_gra)
    w -= lr / ada * grad
    if t < 100 or t % 1000 == 0:
        accuracy = np.sum((fw > 0.5) == y_data) / N
        print(t,accuracy)
np.savez('weight-without-normalization', mean=mean, std=std, w=w)
