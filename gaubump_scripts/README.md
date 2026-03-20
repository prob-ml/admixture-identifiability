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

**Shape function (softplus nonlinearity).**
Each mark $(\rho, \log\sigma) \in \mathbb{R}^2$ defines a shape on
$\{-L, \dots, L\}$:

$$f(x) = \operatorname{softplus}(\rho)\;\exp\!\bigl(-x^2 / 2e^{2\log\sigma}\bigr).$$

The softplus nonlinearity ($\log(1+e^\rho)$) ensures positive amplitudes
and provides smooth gradients near zero.  Null marks use $\rho = -10$
($\operatorname{softplus}(-10) \approx 4.5 \times 10^{-5}$), which
produces faint background-like noise rather than an exact zero — this
gives the model a well-behaved gradient signal for learning zero amplitude.

**Generative process.**
Given $(L, T, \pi)$, a single draw of $X \in \mathbb{R}^{T-2L+1}$ is:

1. For each $t \in \{0, 1, \dots, T\}$, sample $U_t = (\rho_t, \log\sigma_t) \sim \pi$.
2. For each $t \in \{L, \dots, T-L\}$, set

$$X_t = \sum_{\tau=0}^{T}
        \operatorname{softplus}(\rho_\tau) \exp\!\bigl(-(t-\tau)^2 / 2e^{2\log\sigma_\tau}\bigr).$$

The **null shape probability** is the mass $\pi$ assigns near
$\rho = -10$ (i.e.\ where $\operatorname{softplus}(\rho) \approx 0$).

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

Phase II restarts the optimiser from scratch (fresh cosine schedule and
Adam moments) to avoid stale momentum from Phase I.

## Files

| File | Description |
|------|-------------|
| `README.md` | This file |
| `model.py` | Generative model: shape function (softplus), sampling $U$, computing $X$ |
| `flow_matching.py` | MLP velocity network, flow-matching loss, ODE sampling |
| `train.py` | Local end-to-end training script (Phase I + Phase II), CLI |
| `run_modal_experiment.py` | Self-contained Modal GPU experiment runner (parameterised) |
| `results/` | Output directories from completed experiment runs |

### Running an experiment on Modal

```bash
modal run gaubump_scripts/run_modal_experiment.py
```

Edit the **CASE CONFIGURATION** block inside `run_modal_experiment.py` to
change `null_prob_true`, `null_prob_init`, the mixture components, and
`case_name`.  Results are saved to `results/<case_name>/`.

Each run produces:

| Output | Description |
|--------|-------------|
| `training_log.txt` | Phase I + II training metrics |
| `gpu_utilization.txt` | GPU utilisation time series |
| `xdata_true.png` | Example X observations from the true model |
| `xdata_pihat.png` | Example X observations from the initial guess |
| `posterior_vs_truth.png` | Scatter: ground-truth π vs inferred aggregate posterior |
| `rho_marginal.png` | Histogram: marginal of ρ (true vs inferred) |
| `rho_survival.png` | Survival function: P(softplus(ρ) > t) |
| `true_vs_inferred_U.png` | Scatter: true latent U vs inferred U |

### Running locally (without Modal)

```bash
cd gaubump_scripts
python train.py [--L 3] [--T 20] [--n_phase1 2000] [--n_phase2 200] ...
```

## Current results

### `case_softplus_np95_pihat95`

Configuration: $L=3$, $T=20$, `null_prob_true=0.95`, `null_prob_init=0.95`,
10 000 Phase I steps, 150 000 Phase II steps, seed 42.

| Metric | Value |
|--------|-------|
| Post-Phase I frac(softplus(ρ) < 0.1) | 0.956 |
| Final frac(softplus(ρ) < 0.1) | 0.952 |
| True null probability | 0.950 |

Phase II bootstrap remains **stable** throughout all 150k iterations
(frac ≈ 0.95 ± 0.01), thanks to the softplus nonlinearity.
