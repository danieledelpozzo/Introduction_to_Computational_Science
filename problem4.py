import numpy as np
import matplotlib.pyplot as plt

# Cubic B-spline basis function
def B3(x):
    x = np.atleast_1d(x)  # ensure array input
    y = np.zeros_like(x, dtype=np.float64)

    # Piecewise cases
    i1 = (-2 < x) & (x < -1)
    i2 = (-1 <= x) & (x < 1)
    i3 = (1 <= x) & (x < 2)

    y[i1] = 0.5 * (x[i1] + 2)**3
    y[i2] = 0.5 * (3 * np.abs(x[i2])**3 - 6 * x[i2]**2 + 4)
    y[i3] = 0.5 * (2 - x[i3])**3

    return y / 3

# Function to compute spline coefficients (alpha)
def b3interpolate(y):
    n = len(y)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(-1, 3):
            idx = i + j
            if 0 <= idx < n:
                A[i, idx] += B3(j)

    # Natural boundary conditions
    A[0] = 0
    A[0, 0:3] = [1, -2, 1]
    y[0] = 0

    A[-1] = 0
    A[-1, -3:] = [1, -2, 1]
    y[-1] = 0

    alpha = np.linalg.solve(A, y)
    return alpha

# Function to evaluate the spline at given x values
def spline_curve(alpha, x):
    x = np.asarray(x)
    s = np.zeros_like(x, dtype=np.float64)
    n = len(alpha)

    for k in range(n):
        s += alpha[k] * B3(x - k)

    return s

# Given data points
y_data = np.array([
    -0.0044, -0.0213, -0.0771, -0.2001, -0.3521, -0.3520,
     0, 1, 2, 3, 4, 5, 0.5741, 0.8673, 0.5741,
     0, -0.3520
])

# Compute spline coefficients
alpha = b3interpolate(y_data.copy())

# Evaluation points
x = np.arange(0, 16.01, 0.01)
v = spline_curve(alpha, x)

# Plot
plt.plot(x, v)
plt.xlabel('x')
plt.ylabel('Spline value')
plt.title('Cubic Spline Interpolation')
plt.show()