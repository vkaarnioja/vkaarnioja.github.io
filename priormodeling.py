# Draw from the Gaussian prior with positivity constraint
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt

# Draw from the Gaussian prior with positivity constraint
p = 100
alpha = 1
quantile = lambda t: np.sqrt(2)*alpha*erfinv(t) # inverse CDF
# Use inverse transform sampling: draw "u" from U(0,1)
# and obtain realization by computing quantile(u)
u = np.random.uniform(size=(p,p))
draw = quantile(u)
fig,ax = plt.subplots()
ax.imshow(draw,cmap='gray')
ax.set_title('White noise prior',fontsize=15)
plt.show()

# Draw from the l^1 prior with positivity constraint
p = 100
alpha = 1
quantile = lambda t: -1/alpha * np.log(1-t) # inverse CDF
# Use inverse transform sampling: draw "u" from U(0,1)
# and obtain realization by computing quantile(u)
u = np.random.uniform(size=(p,p))
draw = quantile(u)
fig,ax = plt.subplots()
ax.imshow(draw,cmap='gray')
ax.set_title('$\ell^1$ prior',fontsize=15)
plt.show()

# Draw from the Cauchy prior with positivity constraint
p = 100
alpha = 1
quantile = lambda t: 1/alpha * np.tan(np.pi*t/2) # inverse CDF
# Use inverse transform sampling: draw "u" from U(0,1)
# and obtain realization by computing quantile(u)
u = np.random.uniform(size=(p,p))
draw = quantile(u)
fig,ax = plt.subplots()
ax.imshow(draw,cmap='gray')
ax.set_title('Cauchy prior',fontsize=15)
plt.show()