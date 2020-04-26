import numpy as np
import random
import operator

a = np.array([[0,1,2],[3,4,5],[6,7,8],[1,2,3]])
b = np.array([[10,11,12],[13,41,51],[16,17,18],[1,2,43]])

c = np.copy(a.T)
c[1] = b.T[1]
c = c.T
breakpoint()