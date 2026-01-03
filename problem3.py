import numpy as np
import matplotlib.pyplot as plt

# Define the integrand
def f(x):
    return 1 - 4 * (x - 0.5)**2

# Trapezoidal rule implementation
def trapezoidal_rule(f, a, b, num_intervals):
    x = np.linspace(a, b, int(num_intervals))
    h = (b - a) / (len(x) - 1)
    y = f(x)
    integral = h / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

# Integration bounds
a, b = 0, 4

# Interval counts
n_values = np.arange(3, 51)
errors = []

# Exact integral value (computed analytically)
exact_value = -160 / 3

# Compute the approximate integral and error for each n
for n in n_values:
    approx = trapezoidal_rule(f, a, b, n)
    error = abs(exact_value - approx)
    errors.append(error)

# Plot the error on a semilog-y scale
plt.figure(figsize=(8, 5))
plt.semilogy(n_values, errors, marker='o')
plt.xlabel('Number of intervals (n)')
plt.ylabel('Absolute Error')
plt.title('Error of Trapezoidal Rule')
plt.grid(True, which='both', linestyle='--')
plt.show()