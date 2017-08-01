# import matplotlib
# matplotlib.use("gtk")

import numpy as np
import matplotlib.pyplot as plt



# matplotlib.rcParams['backend'] = "Qt4Agg"

# print(plt.get_backend())
# plt.switch_backend(plt.get_backend())

x = np.arange(0,2*np.pi,0.1)
y = np.sin(x)
plt.plot(x,y)
plt.show()
print("finished")
