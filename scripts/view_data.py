import numpy as np
import matplotlib.pyplot as plt

x = np.load('data/record_70.npy')
print(x.shape, x.dtype)

y = x[3]    # this will have 4 shapes
plt.imshow(y)
plt.colorbar()
plt.show()