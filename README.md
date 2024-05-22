# PageRank Algorithm Implementation with CUDA and OpenMP

This repository contains an implementation of the PageRank algorithm using both CUDA and OpenMP. The PageRank algorithm, originally developed by Google's founders Larry Page and Sergey Brin, is a method for ranking web pages in search engine results.

## Overview

The implementation leverages the power of parallel computing to efficiently compute the PageRank of a large number of web pages. It uses CUDA, a parallel computing platform and application programming interface model created by NVIDIA, to offload computation to the GPU. This allows for significant speedup as the GPU can perform many computations simultaneously.

In addition to CUDA, this implementation also uses OpenMP, a portable, scalable model that gives shared-memory parallel programmers a simple and flexible interface for developing parallel applications for platforms ranging from the standard desktop computer to the supercomputer.

## Structure

The main components of the implementation are:

- **Matrix Generation**: A CUDA kernel is used to generate the adjacency matrix representing the web graph. The generation process is parallelized across the GPU threads for efficiency.
- **PageRank Calculation**: The PageRank calculation involves iteratively multiplying the adjacency matrix by a vector. This operation is also parallelized using a CUDA kernel.
- **Vector Comparison**: After each iteration, the resulting vector is compared with the vector from the previous iteration to check for convergence. This is done using an OpenMP parallel for loop.
