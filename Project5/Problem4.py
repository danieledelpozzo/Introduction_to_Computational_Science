import numpy as np
import matplotlib.pyplot as plt

def midpoint(f, a, b, h):
    n = max(1, round((b - a) / h))
    h = (b - a) / n
    x_mid = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(x_mid))

def trapezoidal(f, a, b, h):
    n = max(1, round((b - a) / h))
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = f(x)
    return (h / 2) * (fx[0] + 2.0 * np.sum(fx[1:-1]) + fx[-1])

def f(x):
    return np.sqrt(4 * np.sin(x)**2 + np.cos(x)**2)

a, b = 0.0, 2.0 * np.pi
exact = 9.688448220547674

i_vals = np.arange(1, 31)
h_vals = np.pi / i_vals
err_mid = []
err_trap = []

for i in i_vals:
    h = np.pi / i
    err_mid.append(abs(midpoint(f, a, b, h) - exact))
    err_trap.append(abs(trapezoidal(f, a, b, h) - exact))

err_mid = np.array(err_mid)
err_trap = np.array(err_trap)

plt.figure()
plt.semilogy(i_vals, err_mid, label='Midpoint Error')
plt.semilogy(i_vals, err_trap, label='Trapezoidal Error')
plt.xlabel('i (h = π/i)')
plt.ylabel('Absolute Error (log scale)')
plt.title('Error Comparison: Midpoint vs Trapezoidal Rule')
plt.legend()
plt.grid(True)
plt.show()
