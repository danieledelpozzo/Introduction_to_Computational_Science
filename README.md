# Computation Science Projects
This repository contains a collection of projects developed for the Introduction to Computational Science course.


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
