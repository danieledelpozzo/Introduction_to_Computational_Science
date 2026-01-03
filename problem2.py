import numpy as np
import matplotlib.pyplot as plt


def barycentric_weights(x_nodes):
    n = len(x_nodes)
    tilde_rho = np.zeros(n)
    sigma = np.ones(n)

    for i in range(n):
        log_sum = 0.0
        sign_product = 1
        for j in range(n):
            if j != i:
                diff = x_nodes[i] - x_nodes[j]
                log_sum += np.log(abs(diff))
                if diff < 0:
                    sign_product = -sign_product
        tilde_rho[i] = log_sum
        sigma[i] = sign_product

    min_tilde_rho = np.min(tilde_rho)
    rho = sigma * np.exp(-(tilde_rho - min_tilde_rho))
    return rho

def barycentric_interpolation(f_values, rho, x, x_nodes):
    p = np.zeros_like(x, dtype=float)
    n = len(x_nodes)

    for j in range(len(x)):
        node_index = np.where(x_nodes == x[j])[0]
        if len(node_index) > 0:
            p[j] = f_values[node_index[0]]
        else:
            denom = 0.0
            numer = 0.0
            for i in range(n):
                weight = rho[i] / (x[j] - x_nodes[i])
                denom += weight
                numer += weight * f_values[i]
            p[j] = numer / denom
    return p

if __name__ == "__main__":
    x_nodes = np.array([0, 1, 2, 3, 4], dtype=float)
    rho = barycentric_weights(x_nodes)

    x_plot = np.linspace(0, 4, 401)

    plt.figure()
    for i in range(len(x_nodes)):
        f_vals = np.zeros_like(x_nodes, dtype=float)
        f_vals[i] = 1.0
        L_i = barycentric_interpolation(f_vals, rho, x_plot, x_nodes)
        plt.plot(x_plot, L_i, label=f"L_{i}")

    plt.scatter(x_nodes, np.eye(len(x_nodes))[range(len(x_nodes)), range(len(x_nodes))], color='black')
    plt.legend()
    plt.title("Lagrange Basis Polynomials (Barycentric Form)")
    plt.show()