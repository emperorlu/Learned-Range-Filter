import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import sys

a = int(sys.argv[1])  
# b = int(sys.argv[2])

X = np.linspace(1,a,a)
print(X)
y =  np.array([x  for x in np.random.randint(0,2,a)])
print(y)

sx=np.linspace(1,a,a+2)
func2=interpolate.UnivariateSpline(X,y,s=8)
sy=func2(sx)


plt.plot(X,y,'.')
plt.plot(sx,sy)
plt.savefig("spline.png")
plt.show()