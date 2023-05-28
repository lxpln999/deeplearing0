import numpy as np
import torch

x = np.array([0,1])
w = np.array([0.5,0.5])
b = -0.7

r=np.sum(w*x) + b

print(r)

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = 0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

a = np.array([0.3,2.9,4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
y = exp_a/sum_exp_a
print(y)

def sofmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y=exp_a / sum_exp_a
    return y

tt = sofmax(np.array([0.2,0.8,0.9,0.22]))
print(tt)

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

m = torch.arange(4)
print(m)
