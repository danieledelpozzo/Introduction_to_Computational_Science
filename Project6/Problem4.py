import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# --- Parameters ---
A_true, k_true, B_true = 5, 0.6, 1
sigma = 0.3
n_points = 40
t = np.linspace(0, 10, n_points)
np.random.seed(42)

# --- Data Generation ---
C_true = A_true * (1 - np.exp(-k_true * t)) + B_true
noise = np.random.normal(0, sigma, size=n_points)
C_noisy = C_true + noise

# --- Custom least squares for 4th-degree Taylor polynomial ---
# Create design matrix for t^0 to t^4
X = np.vstack([np.ones_like(t), t, t**2, t**3, t**4]).T

# Solve least squares: beta = (X^T X)^-1 X^T y
XtX = X.T @ X
Xty = X.T @ C_noisy
beta = np.linalg.inv(XtX) @ Xty
C_fit_custom = X @ beta

# --- 3rd-degree polynomial using numpy.polyfit ---
coeffs_polyfit = np.polyfit(t, C_noisy, 3)
C_fit_polyfit = np.polyval(coeffs_polyfit, t)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.scatter(t, C_noisy, color='gray', label='Noisy data')
plt.plot(t, C_fit_custom, label='4th-degree Taylor Fit (custom)', linewidth=2)
plt.plot(t, C_fit_polyfit, label='3rd-degree Polyfit', linewidth=2)
plt.title("Polynomial Fits to Noisy Data")
plt.xlabel("Time t")
plt.ylabel("Concentration C(t)")
plt.legend()
plt.grid(True)
plt.savefig("polynomial_fits.png")
plt.show()

# --- Residuals and Metrics ---
residuals_custom = C_noisy - C_fit_custom
residuals_polyfit = C_noisy - C_fit_polyfit
RSS_custom = np.sum(residuals_custom**2)
RSS_polyfit = np.sum(residuals_polyfit**2)
TSS = np.sum((C_noisy - np.mean(C_noisy))**2)
R2_custom = 1 - RSS_custom / TSS
R2_polyfit = 1 - RSS_polyfit / TSS

print("4th-degree Taylor Polynomial Coefficients (custom least squares):")
print(f"beta_0 = {beta[0]:.4f}, beta_1 = {beta[1]:.4f}, beta_2 = {beta[2]:.4f}, beta_3 = {beta[3]:.4f}, beta_4 = {beta[4]:.4f}\n")

print("3rd-degree Polynomial Coefficients (np.polyfit):")
print(f"coeff_0 = {coeffs_polyfit[0]:.4f}, coeff_1 = {coeffs_polyfit[1]:.4f}, coeff_2 = {coeffs_polyfit[2]:.4f}, coeff_3 = {coeffs_polyfit[3]:.4f}\n")

print("Model Comparison:")
print(f"4th-degree Taylor Fit -> RSS = {RSS_custom:.4f}, R^2 = {R2_custom:.4f}")
print(f"3rd-degree Polyfit    -> RSS = {RSS_polyfit:.4f}, R^2 = {R2_polyfit:.4f}")