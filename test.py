import matplotlib.pyplot as plt
import numpy as np

mean = np.zeros(7)
cov = np.random.rand(7,7)  # diagonal covariance

x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
# %%
x = 3*np.random.rand(7,1)
a = x*x.T
print((a==a.T).all())
# %%
np.random.multivariate_normal(mean, a, 5000).T