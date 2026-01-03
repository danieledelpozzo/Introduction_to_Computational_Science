import numpy as np
import matplotlib.pyplot as plt


def runge_function(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)


def equidistant_nodes(n):
    return np.linspace(-1, 1, n)


def chebyshev_nodes_first_kind(n):
    k = np.arange(n)
    return np.cos((2 * k + 1) * np.pi / (2 * n))


def chebyshev_nodes_second_kind(n):
    k = np.arange(n)
    return np.cos(k * np.pi / (n - 1))


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

    min_val = np.min(tilde_rho)
    rho = sigma * np.exp(-(tilde_rho - min_val))
    return rho

def barycentric_interpolation(f_values, rho, x_eval, x_nodes):
    p = np.zeros_like(x_eval, dtype=float)
    n = len(x_nodes)

    for i, x in enumerate(x_eval):
        # Check if x == any node
        idx = np.where(np.isclose(x_nodes, x))[0]
        if len(idx) > 0:
            p[i] = f_values[idx[0]]
        else:
            denom = 0.0
            numer = 0.0
            for k in range(n):
                weight = rho[k] / (x - x_nodes[k])
                denom += weight
                numer += weight * f_values[k]
            p[i] = numer / denom
    return p


if __name__ == "__main__":
    node_counts = [5, 10, 15]
    x_fine = np.linspace(-1, 1, 400)
    f_exact = runge_function(x_fine)

    plt.figure(figsize=(10, 8))

    node_types = [
        ("Equidistant", equidistant_nodes),
        ("Chebyshev-1", chebyshev_nodes_first_kind),
        ("Chebyshev-2", chebyshev_nodes_second_kind),
    ]

    index_plot = 1

    for n in node_counts:
        for (label, node_func) in node_types:
            x_nodes = node_func(n)
            y_nodes = runge_function(x_nodes)

            rho = barycentric_weights(x_nodes)
            p_approx = barycentric_interpolation(y_nodes, rho, x_fine, x_nodes)

            plt.subplot(len(node_counts), len(node_types), index_plot)
            plt.plot(x_fine, f_exact, 'k-', label="Exact f(x)")
            plt.plot(x_fine, p_approx, '--', label="Interp")
            plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label="Nodes")
            plt.ylim([-0.2, 1.2])
            plt.title(f"N={n}, {label}")
            plt.legend(fontsize=7, loc="upper left")

            index_plot += 1

    plt.tight_layout()
    plt.savefig("runge.png")
    plt.show()
