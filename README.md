# Computation Science Projects
This repository contains a collection of projects developed for the Introduction to Computational Science course at USI.


# Project 3 – Polynomial Interpolation & the Runge Phenomenon

This project investigates the properties of Lagrange basis polynomials and the impact of node selection on interpolation accuracy, specifically focusing on overcoming the Runge phenomenon.

1. Barycentric Interpolation

Implementation of an efficient, numerically stable barycentric formula to compute Lagrange interpolation polynomials.

2. Lagrange Basis Analysis

Verification of the partition of unity property, confirming that the sum of all basis polynomials equals exactly one for any input value.

3. Runge Function Study

Interpolation of the Runge function to observe numerical instability and the divergence of errors as the polynomial degree increases.

4. Node Selection Strategies

A comparative analysis of node distributions to determine their impact on stability:

Equidistant Nodes: Demonstrates heavy oscillations near the edges of the interval as the number of nodes increases.

Chebyshev Nodes (1st and 2nd Kind): Uses non-uniform spacing to minimize the maximum error and effectively suppress oscillations.

5. Error Estimation

Analytical derivation of error bounds using high-order derivatives to validate the observed numerical results.


# Project 4 – Linear and Cubic Spline Interpolation

This project explores piecewise polynomial interpolation, focusing on the construction and error analysis of linear and cubic B-splines to provide smooth approximations of functions.

1. Linear Spline Construction: Implementation of piecewise linear functions to connect data points, ensuring continuity across the interval.

2. Error Bound Analysis: Mathematical derivation of error estimates for linear splines, demonstrating how the maximum error depends on the second derivative of the function and the square of the step size.

3. Cubic B-Splines: Implementation of third-degree splines to achieve higher smoothness. This involves solving a linear system of equations to find coefficients that ensure continuity of the first and second derivatives at every node.

4. Boundary Conditions: Application of natural boundary conditions to close the linear system, ensuring the spline behaves predictably at the endpoints of the data range.

5. Numerical Evaluation: Comparison between the original function and the spline approximations to visualize how increasing the number of nodes improves the fit and smoothness.


# Project 5 – Numerical Integration and Newton-Cotes Formulas

This project explores various numerical methods for approximating definite integrals, comparing their precision against analytical solutions and evaluating their computational efficiency.

1. Basic Newton-Cotes Methods: Implementation and evaluation of the Trapezoidal Rule and Simpson’s Rule to approximate the integral of a function.

2. Analytical Validation: Calculation of exact integral values using antiderivatives to establish a baseline for error measurement.

3. Composite Rules: Development of composite versions of the Trapezoidal and Simpson’s rules to improve accuracy by dividing the integration interval into smaller sub-segments.

4. Adaptive Quadrature: Implementation of an advanced recursive algorithm that automatically adjusts the step size to meet a predefined error tolerance.

5. Error Analysis and Comparison: Quantitative study of error convergence, demonstrating that the Composite Simpson’s Rule provides significantly higher accuracy than basic methods for the same computational cost.

6. Numerical Stability: Investigation of different rules, such as the Midpoint and Trapezoidal rules, for periodic functions to identify their respective strengths in terms of convergence rates.
