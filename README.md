# Simulations investigating the identifiability of an additive process

## Overview

We consider a toy simulation model, inspired by neuroscience measurements, as follows.

Define the "mark space," $S$, as the set of all functions $u$ from the integers to $\{0, 1, ..., M\}$, with the following conditions:

- **Support condition:** $u(t) = 0$ for all $t$ outside the interval $\{-L, ..., L\}$.
- **Centering condition:** The sum $\sum_{t=-L}^L t \cdot u(t)$ equals zero.

Let $\pi$ be a probability mass function (PMF) over this mark space. We consider a random vector $X$ in $\mathbb{R}^{T-2L+1}$, generated as follows:

- For each $t$ in $\{0, 1, ..., T\}$, sample $U_t$ independently from $\pi$.
- For each $t$ in $\{L, ..., T-L\}$, set $X_t$ to the sum over $\tau$ of $U_\tau(t-\tau)$.

The process $X$ can approximate a wide range of processes as $T$ grows large. For example, to connect this discretization to traditional Poisson processes, let $p$ denote the probability that $u$ is identically zero under $\pi$.  The number of nonzero impulses contributing to $X$ is approximately Poisson distributed with rate $T p$ for large $T$.

The model is computationally tractable for moderate values of $M$, $T$, and $L$. To compute the log likelihood of a particular realization of $X$, note that $X$ is a sum of $T+1$ independent vectors. We can compute its PMF iteratively using discrete convolution. Specifically, after computing the PMF for the sum of the first $k$ terms, we can obtain the PMF for the sum of the first $k+1$ terms by convolving with the next term. Using this approach, the PMF for $X$ can be computed in $O((T+1)(2ML+M+1)^{T-2L} M^{2L+1})$ operations. For example, with $T=7$, $L=2$, and $M=3$, this requires about ten million operations. The marginal posterior of each $U_t$ given $X$ can also be computed in roughly the same amount of time, since each posterior is proportional to the gradient of the log likelihood with respect to $\pi$.

This repository contains code for running simulations that test the identifiability of an unknown PMF $\pi$ given perfect knowledge of the law of $X$.

## Files

* `scripts/neuron_spike_simulation.py` create a plot that demonstrates the "deblending" problem.
* `scripts/emalg.py` contains an EM algorithm for fitting $\pi$ using knowledge of the law of $X$.
* `scripts/run_finite_data.py` runs the algorithm in cases with finite data
* `scripts/plot_finite_data_sims.py` plots the results of the above
* `scripts/run_infinite_data.py` runs the algorithm with infinite data, tracking the optimization progress
* `scripts/plot_infinite_data_sims.py` plots the results of the above
