# Gaussian Bump Additive Model

## Overview

This directory implements estimation of a shape distribution from noisy
additive observations, using **rectified flow matching** in JAX.

The setup generalises the discrete mark-space model in `scripts/` to a
continuous mark space $S = \mathbb{R}^2$.

## Model

| Symbol | Meaning |
|--------|---------|
| $L$ | support window (integer) |
| $T$ | visible window (integer) |
| $\pi$ | shape distribution on $\mathbb{R}^2$ |
| $(\rho, \log\sigma)$ | a point in the mark space |

**Shape function.**
Each mark $(\rho, \log\sigma) \in \mathbb{R}^2$ defines a shape on
$\{-L, \dots, L\}$:

$$f(x) = \rho \exp\!\bigl(-x^2 / 2e^{2\log\sigma}\bigr).$$

When $\rho = 0$ the shape is identically zero (null shape).

**Generative process.**
Given $(L, T, \pi)$, a single draw of $X \in \mathbb{R}^{T-2L+1}$ is:

1. For each $t \in \{0, 1, \dots, T\}$, sample $U_t = (\rho_t, \log\sigma_t) \sim \pi$.
2. For each $t \in \{L, \dots, T-L\}$, set

$$X_t = \sum_{\tau=0}^{T}
        \rho_\tau \exp\!\bigl(-(t-\tau)^2 / 2e^{2\log\sigma_\tau}\bigr).$$

The **null shape probability** is $\pi(\rho = 0) = \pi(\{0\} \times \mathbb{R})$.

## Goal

Estimate $\pi$ from samples of $X$ alone (the latent $U$ is unobserved).

## Approach — Rectified Flow Matching

We train a *conditional* rectified flow matching network that learns to
map noise to $U \in \mathbb{R}^{(T+1) \times 2}$ given $X$.

### Phase I — Warm-up

Train the flow network on $(X, U)$ pairs sampled from an initial guess
$\hat\pi$.  Each gradient step uses freshly simulated data (no fixed
dataset, no repeated samples).

### Phase II — Iterative refinement

Repeat:

1. Draw `n_source` samples of $X$ from the **true** model $(L, T, \pi)$.
2. Push each $X$ through the current flow network to obtain
   `n_source × (T+1)` samples of $U \in \mathbb{R}^2$ (stop-gradient).
3. Define $\hat\pi$ as the empirical distribution on these samples.
4. Train the flow network for one epoch on freshly simulated $(X, U)$
   pairs under the new $\hat\pi$ (never reuse samples).

## Files

| File | Description |
|------|-------------|
| `README.md` | This file |
| `DETAILED_PLAN.md` | Implementation road-map and design notes |
| `model.py` | Generative model: sampling $U$ and computing $X$ |
| `flow_matching.py` | MLP velocity network and flow-matching utilities |
| `train.py` | End-to-end training script (Phase I + Phase II) |
