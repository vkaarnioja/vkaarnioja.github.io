# Source localization example
import numpy as np
import matplotlib.pyplot as plt

# Form the posterior density using voltage measurements
# at both end points x = 0 and x = 1
x_ast = 1/np.pi # Fix "ground truth", i.e., particle location
sigma = .2 # Std for simulated noise
v = 1/np.abs(x_ast-np.array([0,1])) # Measurements at end points
np.random.seed(321); # Reproducible experiments
v = v+sigma*np.random.normal(size=v.shape) # Add noise to measurements
x = np.linspace(0,1) # Discretize the unit interval
x = x[1:-1] # Drop end points to avoid numerical issues...
# Define the (unnormalized) posterior density
p = lambda x: (x >= 0) * (x <= 1) * np.exp(-1/(2*sigma**2)*((v[0]-1/np.abs(x-0))**2\
               +(v[1]-1/np.abs(x-1))**2))
P = np.cumsum(p(x)) # Form (unnormalized) cumulative distribution function
p_int = 1/(x.size-1) * (.5*p(x[0]) + np.sum(p(x[1:-1])) + .5*p(x[-1])) # Normalization coefficient (trapezoidal rule)
p_normalized = lambda x: p(x)/p_int # Form normalized posterior density
# Visualize the posterior density and the location of the ground truth
plt.plot(x,p_normalized(x),label='posterior density')
plt.plot(x_ast,p_normalized(x_ast),'r*',label='ground truth')
plt.legend()
plt.show()

# Form the posterior density using voltage measurements
# at only one end point x = 1
x_ast = 1/np.pi # Fix "ground truth", i.e., particle location
sigma = .2 # Std for simulated noise
v = 1/np.abs(x_ast-np.array([1])) # Measurement at one end point
np.random.seed(321); # Reproducible experiments
v = v+sigma*np.random.normal(size=v.shape) # Add noise to measurements
x = np.linspace(0,1) # Discretize the unit interval
x = x[1:-1] # Drop end points to avoid numerical issues...
# Define the (unnormalized) posterior density
p = lambda x: (x >= 0) * (x <= 1) * np.exp(-1/(2*sigma**2)*(v[0]-1/np.abs(x-1))**2)
P = np.cumsum(p(x)) # Form (unnormalized) cumulative distribution function
p_int = 1/(x.size-1) * (.5*p(x[0]) + np.sum(p(x[1:-1])) + .5*p(x[-1])) # Normalization coefficient (trapezoidal rule)
p_normalized = lambda x: p(x)/p_int # Form normalized posterior density
# Visualize the posterior density and the location of the ground truth
plt.plot(x,p_normalized(x),label='posterior density')
plt.plot(x_ast,p_normalized(x_ast),'r*',label='ground truth')
plt.legend()
plt.show()

