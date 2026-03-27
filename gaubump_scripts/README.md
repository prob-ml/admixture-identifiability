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
| $V$ | binary gate (0 = null, 1 = active) |
| $(\rho, \log\sigma)$ | a point in the mark space |

**Shape function (V-binary gating).**
Each mark has a binary gate $V \in \{0,1\}$ and parameters
$(\rho, \log\sigma) \in \mathbb{R}^2$.  The shape on $\{-L, \dots, L\}$ is:

$$f(x) = V \cdot \rho \;\exp\!\bigl(-x^2 / 2e^{2\log\sigma}\bigr).$$

When $V=0$ (null mark), the contribution to $X$ is exactly zero and
$(ρ, \log σ) = (0, 0)$ by convention.  When $V=1$ (active mark),
$\rho$ is used directly as the amplitude (no softplus).

**Generative process.**
Given $(L, T, \pi)$, a single draw of $X \in \mathbb{R}^{T-2L+1}$ is:

1. For each $t \in \{0, 1, \dots, T\}$, sample $(V_t, U_t) = (V_t, \rho_t, \log\sigma_t) \sim \pi$.
2. For each $t \in \{L, \dots, T-L\}$, set

$$X_t = \sum_{\tau=0}^{T}
        V_\tau \cdot \rho_\tau \;\exp\!\bigl(-(t-\tau)^2 / 2e^{2\log\sigma_\tau}\bigr).$$

The **null probability** is $\Pr(V=0)$.

## Goal

Estimate $\pi$ from samples of $X$ alone (the latent $(V, U)$ is unobserved).

## Approach — Two-Model Rectified Flow Matching

We train **two** models simultaneously:

1. **V-classifier** (`VClassifierMLP`): maps $X \to V$ logits
   (one per position), trained with binary cross-entropy.

2. **U-flow** (`VelocityMLP`): rectified flow matching for
   $U \in \mathbb{R}^{(T+1) \times 2}$, conditioned on $(X, V)$.
   The flow matching loss is **masked** so that only active ($V=1$)
   positions contribute — null positions are trivially zero and don't
   waste model capacity.

### Phase I — Warm-up

Train both networks on $(X, V, U)$ triples sampled from an initial
guess $\hat\pi$.  Each gradient step uses freshly simulated data.

### Phase II — Iterative refinement

Repeat:

1. Draw `n_source` samples of $X$ from the **true** model $(L, T, \pi)$.
2. Predict $\hat V$ using the V-classifier (stop-gradient).
3. Infer $\hat U$ using the U-flow conditioned on $\hat V$ (stop-gradient).
   Null positions ($\hat V = 0$) get $U = (0, 0)$.
4. Define $\hat\pi$ as the empirical distribution over $(\hat V, \hat U)$.
5. Train both models on freshly simulated $(X, V, U)$ from $\hat\pi$.

Phase II restarts both optimisers from scratch (fresh cosine schedules
and Adam moments).  An **oracle** pair of models is trained in parallel
on ground-truth $(X, V, U)$ from the true $\pi$ as a reference.

## Files

| File | Description |
|------|-------------|
| `README.md` | This file |
| `model.py` | Generative model: V-binary shape function, sampling $(V, U)$, computing $X$ |
| `flow_matching.py` | V-classifier, velocity network, masked flow loss, ODE sampling |
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
| `posterior_vs_truth.png` | Scatter: ground-truth π vs inferred (V, U) |
| `rho_marginal.png` | Histogram: marginal of ρ for V=1 marks |
| `rho_survival.png` | Survival function of ρ for V=1 marks |
| `true_vs_inferred_U.png` | Scatter: true latent (V, U) vs inferred |
| `loss_curves.png` | Phase I + Phase II loss curves (bootstrap vs oracle) |

### Running locally (without Modal)

```bash
cd gaubump_scripts
python train.py [--L 3] [--T 20] [--n_phase1 2000] [--n_phase2 200] ...
```
