import numpy as np
import scipy.misc
# 1 numpy
## 1.1 Basics

a = np.arange(15)
print(type(a),a)

print(a.shape)
print(a.ndim)

print(a.flags)
print(a.base)
print(a.trace)
np.linalg.solve(a,y)