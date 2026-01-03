import numpy as np
import matplotlib.pyplot as plt

def compress(x, y, tol):
    comp_x = [x[0]]
    comp_y = [y[0]]
    start_idx = 0
    max_err = 0.0

    for i in range(2, len(x)):
        x_start, x_curr = x[start_idx], x[i]
        y_start, y_curr = y[start_idx], y[i]
        local_max_err = 0.0
        for j in range(start_idx + 1, i):
            t = (x[j] - x_start) / (x_curr - x_start)
            y_lin = (1 - t) * y_start + t * y_curr
            err = abs(y[j] - y_lin)
            local_max_err = max(local_max_err, err)
        if local_max_err > tol:
            comp_x.append(x[i - 1])
            comp_y.append(y[i - 1])
            start_idx = i - 1
            max_err = max(max_err, local_max_err)

    comp_x.append(x[-1])
    comp_y.append(y[-1])
    max_err = 0.0
    for k in range(len(comp_x) - 1):
        left_idx = np.where(x == comp_x[k])[0][0]
        right_idx = np.where(x == comp_x[k + 1])[0][0]
        for j in range(left_idx + 1, right_idx):
            t = (x[j] - comp_x[k]) / (comp_x[k + 1] - comp_x[k])
            y_lin = (1 - t) * comp_y[k] + t * comp_y[k + 1]
            err = abs(y[j] - y_lin)
            max_err = max(max_err, err)

    return np.array(comp_x), np.array(comp_y), max_err

# --------------------- Testing the compress function --------------------- #

delta = np.linspace(0, 2 * np.pi, 51)
y = np.sin(delta)
tol = 0.01

comp_x, comp_y, max_err = compress(delta, y, tol)

print("Original number of nodes:", len(delta))
print("Compressed number of nodes:", len(comp_x))
print("Maximum interpolation error:", max_err)

plt.figure(figsize=(8, 4))
plt.plot(delta, y, 'b.-', label='Original')
plt.plot(comp_x, comp_y, 'ro-', label='Compressed')
plt.title('Adaptive Interpolation with Tolerance = 0.01')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig("compression_plot.png")
plt.close()
