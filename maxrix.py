import numpy as np

A = np.array([[3,1],[1,3]])

eginval,eginvec = np.linalg.eig(A)

print("eginval:",eginval)
print("eginvec:",eginvec)