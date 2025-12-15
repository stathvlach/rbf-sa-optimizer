![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# RBF-SA Optimizer
Radial Basis Function Networks with Fuzzy-Means Center Selection and Simulated Annealing Optimization

**Why this project matters:**
This repository demonstrates the design of a fully custom, optimization-driven neural network architecture — combining deterministic fuzzy partitioning with stochastic global search.
It highlights experience in numerical optimization, algorithmic engineering, reproducible research, and executable ML workflows.

## Overview

This repository contains a modern Python re-implementation and extension of the system I originally developed as the core of my Bachelor thesis.
The thesis investigated how the performance and structure of Radial Basis Function (RBF) neural networks evolve when:

1. The network centers are determined using the *fuzzy-means* algorithm, and
2. The model hyper-parameters are optimized using *Simulated Annealing (SA)*.

The fuzzy-means algorithm used in this project is **not** classical fuzzy c-means.
It is the specialized, highly efficient method introduced in:

- [A Radial Basis Function Network Training Algorithm Using a Non-Symmetric Partition of the Input Space](https://www.sciencedirect.com/science/article/abs/pii/S0965997811001335)

- [A New Algorithm for Online Structure and Parameter Adaptation of RBF Networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608003000522)

This methodology partitions the input space into fuzzy subspaces and selects RBF centers based on multidimensional membership functions and relative distance criteria.
The aim of the thesis — and this rewrite — is to study how this deterministic center-selection behaves when combined with a global stochastic optimizer such as SA, focusing on model compactness, accuracy, and stability.

All experiments and visualizations in this repository are presented through **Jupyter notebooks**.

---

## Run the Notebooks on Binder

You can execute the notebooks directly in your browser using Binder.
Click below to launch the main demonstration notebook:

**[A simple demonstration of training an RBF model to approximate a noisy 1D sine
wave.](https://mybinder.org/v2/gh/stathvlach/rbf-sa-optimizer/HEAD?filepath=notebooks%2F01_rbf_sine_demo_kmeans.ipynb)**

**[A demonstration of training an RBF model using fuzzy means algorithm to approximate a synthetic two-dimensional
nonlinear surface with mixed curvature.
](https://mybinder.org/v2/gh/stathvlach/rbf-sa-optimizer/HEAD?filepath=notebooks%2F02_rbf_sine_demo_fuzzymeans.ipynb)**

---

## What You Will Learn From This Repository

- How to implement an RBF neural network from scratch
- How fuzzy partitions can replace traditional clustering
- How to build a global optimization loop (SA) around a model
- How to structure ML experiments using notebooks and Binder
- How to design reproducible scientific workflows in Python

## High-Level Architecture

     +--------------------+
     |  Fuzzy-Means       |  →  Determines candidate RBF centers
     +--------------------+
                |
                v
     +--------------------+
     |   RBF Model        |  →  Gaussian kernel, linear weights
     +--------------------+
                |
                v
     +--------------------+
     | Simulated Annealing|
     +--------------------+
                |
                v
     +-------------------+
     |  Best Model Export|
     +-------------------+

## Scientific Background

### Fuzzy-Means Center Selection

Following the methodology described in the referenced papers, the input space is partitioned into fuzzy sets, forming multidimensional subspaces.
A subspace becomes a candidate RBF center when its membership function is non-zero for incoming data.

**Key properties:**

- Extremely fast center selection
- Compact networks (fewer centers)
- Strong approximation capability
- Deterministic structure (before SA optimization)


### Simulated Annealing Optimization

Simulated Annealing explores hyper-parameters such as:

- Number of fuzzy partitions per dimension
- Neighborhood radius variables
- Gaussian kernel width (σ)
- Other structural parameters

Early iterations allow exploration, while later stages converge to optimal structures.

The central goal of this project is to analyze the interaction between the **fuzzy-means structural prior** and the **SA stochastic optimizer**.
