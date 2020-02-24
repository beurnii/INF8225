import numpy as np

a = np.random.random((1000,784,1))
b = np.random.random((1000,1,300))

np.outer(a,b).shape
breakpoint()