import numpy as np

def ReLU(Z, return_derivative=False):
    if return_derivative:
        return 1 * (Z > 0)
    else:
        return np.multiply(Z, (Z > 0))


x = np.random.randn(3,4)

y = ReLU(x)

z = ReLU(x, return_derivative=True)

a = 5