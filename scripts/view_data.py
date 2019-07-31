import numpy as np
import matplotlib.pyplot as plt

x = np.load('data/record_0.npy')
print(x.shape)

y = x[701] # after 500, we should have 4 shapes!
plt.imshow(y)
plt.show()