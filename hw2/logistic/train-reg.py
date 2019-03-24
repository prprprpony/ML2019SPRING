import numpy as np
# preprocess data
x_data = np.genfromtxt('../X_train', delimiter=',')[1:,]
N = x_data.shape[0]
mean = np.mean(x_data, axis = 0)
std = np.std(x_data, axis = 0)
for i in range(N):
    for j in range(x_data.shape[1]):
        x_data[i][j] = (x_data[i][j] - mean[j]) / std[j]
x_data = np.concatenate((np.ones((N,1)),x_data),axis=1).astype(float)
y_data = np.genfromtxt('../Y_train', dtype=int)[1:,].astype(bool)
# 
w = np.zeros(x_data.shape[1])
lr = 1
ld = 1e3
T = 10**4+1
prev_gra = 0
xt = x_data.T
for t in range(T):
    z = np.matmul(x_data, w)
    fw = 1/(1+np.exp(-z))
    grad = np.matmul(xt, fw - y_data) + ld * np.concatenate(([0],w[1:]))
    prev_gra += np.sum(np.square(grad))
    ada = np.sqrt(prev_gra)
    w -= lr / ada * grad
    if t < 100 or t % 1000 == 0:
        accuracy = np.sum((fw > 0.5) == y_data) / N
        print(t,accuracy)
np.savez('weight-reg'+str(ld), mean=mean, std=std, w=w)
