import numpy as np
import matplotlib.pyplot as plt

d = 1200
t = np.arange(1,d+1)/d
alpha = 1
quantile = lambda t: 1/alpha * np.tan(np.pi * (t-1/2))
unif = np.random.uniform(size=d)
draw = quantile(unif)
y = np.cumsum(draw)
plt.plot(t,y)
plt.xlabel('$t$',fontsize=14)
plt.ylabel('$g(t)$',fontsize=14)
plt.show()