import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr, cond

# Step (a): Generate data
n = 100
x = np.linspace(-5, 5, n)
f = lambda x: np.sin(2 * x) - x**2
y_true = f(x)
y = y_true + 0.5 * np.random.rand(len(x))  # add noise

# Step (b): Construct Vandermonde matrix for degree d = 8
d = 8
V = np.vander(x, d + 1, increasing=True)

# Step (c1): Solve using normal equations
V_T_V = V.T @ V
V_T_y = V.T @ y
a_normal = inv(V_T_V) @ V_T_y

# Step (c2): Solve using QR decomposition
Q, R = qr(V)
a_qr = inv(R) @ Q.T @ y

# Step (d): Print condition numbers
cond_VTV = cond(V_T_V)
cond_R = cond(R)
print("Condition number of V^T V (Normal Equation):", cond_VTV)
print("Condition number of R (QR Decomposition):", cond_R)

# Step (e): Plot results
x_plot = np.linspace(-5, 5, 500)
y_f = f(x_plot)

# Polynomial predictions
V_plot = np.vander(x_plot, d + 1, increasing=True)
y_poly_normal = V_plot @ a_normal
y_poly_qr = V_plot @ a_qr

# Plot for Normal Equations
plt.figure()
plt.plot(x_plot, y_f, label='Original f(x)')
plt.plot(x_plot, y_poly_normal, label='Poly fit (Normal Eq)')
plt.scatter(x, y, color='gray', alpha=0.3, label='Noisy Data')
plt.title('Polynomial Fit using Normal Equations')
plt.legend()
plt.grid(True)
plt.savefig("normal_eq_fit.png", dpi=300)

# Plot for QR Decomposition
plt.figure()
plt.plot(x_plot, y_f, label='Original f(x)')
plt.plot(x_plot, y_poly_qr, label='Poly fit (QR)')
plt.scatter(x, y, color='gray', alpha=0.3, label='Noisy Data')
plt.title('Polynomial Fit using QR Decomposition')
plt.legend()
plt.grid(True)
plt.savefig("qr_fit.png", dpi=300)

plt.show()
