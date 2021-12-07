import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

sampleNo = 1000
mu = 85
sigma = 4
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo )
plt.hist(s, 30, density=True)
print(s)

