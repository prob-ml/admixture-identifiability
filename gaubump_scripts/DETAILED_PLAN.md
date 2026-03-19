# Detailed Plan — Gaussian Bump Flow Matching

## 1. Generative Model (`model.py`)

### Data types

* **Mark** $u = (\rho, \log\sigma) \in \mathbb{R}^2$.
* **Latent sequence** $U \in \mathbb{R}^{(T+1) \times 2}$, one mark per
  time-step.
* **Observation** $X \in \mathbb{R}^{T-2L+1}$.

### Sampling

```
sample_pi(key, pi, n)  -> U[n, 2]
```

$\pi$ will initially be represented as a mixture:
an atom at $(\rho=0, \cdot)$ with some weight, plus a mixture of
axis-aligned Gaussians for the non-null component.  This is flexible
enough to encode the ground-truth distributions we want to test
(e.g. a large atom at $(0,0)$ plus a few Gaussian clusters).

```
compute_X(U, L, T)  -> X[T-2L+1]
```

Given $U \in \mathbb{R}^{(T+1) \times 2}$, compute $X_t$ for
$t \in \{L, \dots, T-L\}$.

### Batch helpers

```
sample_XU(key, pi, L, T, n_samples) -> (X[n, T-2L+1], U[n, T+1, 2])
```

Draw `n_samples` independent $(X, U)$ pairs.

---

## 2. Flow Matching Network (`flow_matching.py`)

### Architecture

An MLP (using Equinox) that takes

* `noisy_U` $\in \mathbb{R}^{(T+1) \times 2}$ — the interpolated
  latent state at diffusion time $s$
* `X` $\in \mathbb{R}^{T-2L+1}$ — the conditioning observation
* `s` $\in \mathbb{R}$ — the flow time

and outputs a velocity $v \in \mathbb{R}^{(T+1) \times 2}$.

Internally we flatten `(noisy_U, X, s)` into a single vector, pass
through several hidden layers with activation, and reshape the output.

### Rectified flow matching loss

Given a pair $(X, U)$:

1. Sample $s \sim \text{Uniform}(0,1)$ and $\epsilon \sim \mathcal{N}(0,I)$.
2. Form $z_s = (1-s)\,\epsilon + s\,U$.
3. Target velocity $v^* = U - \epsilon$.
4. Loss $= \|v_\theta(z_s, X, s) - v^*\|^2$.

### Sampling (inference)

Given $X$, integrate the ODE from $s=0$ to $s=1$ with a simple
Euler discretisation:

$$z_{s+\Delta s} = z_s + \Delta s \; v_\theta(z_s, X, s).$$

---

## 3. Training Script (`train.py`)

### Inputs

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 3 | support window |
| `T` | 20 | visible window |
| `n_init_samples` | 2048 | samples per batch in Phase I |
| `n_source` | 256 | true-model $X$ samples per Phase II step |
| `n_eachstep_samples` | 2048 | training samples per Phase II step |
| `n_phase1_steps` | 2000 | gradient steps in Phase I |
| `n_phase2_steps` | 200 | outer iterations of Phase II |
| `lr` | 1e-3 | Adam learning rate |

### Phase I

```
for step in range(n_phase1_steps):
    key, subkey = jax.random.split(key)
    X, U = sample_XU(subkey, pihat, L, T, n_init_samples)
    loss, grads = flow_loss_and_grad(model, X, U, subkey)
    model, opt_state = update(model, grads, opt_state)
```

### Phase II

```
for step in range(n_phase2_steps):
    # 1. draw X from true model
    key, subkey = jax.random.split(key)
    X_true, _ = sample_XU(subkey, pi_true, L, T, n_source)

    # 2. infer U from current model (stop-gradient)
    U_inferred = jax.lax.stop_gradient(
        sample_flow(model, X_true, subkey))           # [n_source, T+1, 2]
    U_flat = U_inferred.reshape(-1, 2)                # [n_source*(T+1), 2]

    # 3. build empirical pihat from U_flat
    pihat_empirical = EmpiricalDistribution(U_flat)

    # 4. train one epoch on fresh samples from pihat_empirical
    key, subkey = jax.random.split(key)
    X_new, U_new = sample_XU(subkey, pihat_empirical, L, T,
                              n_eachstep_samples)
    loss, grads = flow_loss_and_grad(model, X_new, U_new, subkey)
    model, opt_state = update(model, grads, opt_state)
```

---

## 4. Evaluation (future)

* Compare the empirical $\hat\pi$ (output of Phase II) to the true $\pi$.
* Metrics: TV distance on discretised grid, estimated null-shape
  probability vs true null-shape probability.
* Sweep null-shape probability from 0.9 (easy) down to 0.1 (hard) and
  plot recovery quality.

---

## 5. Diagnostic Experiments and Findings

### 5.1 Experiment summary

Three Modal GPU experiments were run to compare oracle mode, full pipeline
(Phase I + II), and a diagnostic warm-start test:

| Experiment | Script | null_prob_true | Phase I data | Phase II | Final frac(|ρ|<0.1) | Status |
|---|---|---|---|---|---|---|
| Oracle | `run_modal_oracle.py` | 0.95 | TRUE π | none | 0.947 | ✅ |
| Easy pipeline | `run_modal_easy.py` | 0.95 | pihat (0.5) | 150k bootstrap | 0.931 | ✅ |
| Full pipeline | `run_modal.py` | 0.80 | pihat (0.5) | 150k bootstrap | 0.470 | ❌ |
| Diag warm-start | `run_modal_diag.py` | 0.80 | TRUE π (oracle) | 150k bootstrap | 0.326 | ❌ |

### 5.2 Root cause: Phase II bootstrap instability

The diagnostic experiment (`run_modal_diag.py`) is the key finding. It gives
Phase II a **good warm-start** (oracle Phase I, frac=0.673) and Phase II
still **destroys** it (final frac=0.326, worse than the warm-start!).

Convergence trace for the diagnostic experiment:
```
Post-Phase-I (oracle):  frac=0.673  ← decent warm-start
Phase II iter     0:    frac=0.674
Phase II iter   100:    frac=0.526  ← immediate degradation
Phase II iter  1000:    frac=0.531  ← stuck
Phase II iter 15000:    frac=0.530
Phase II iter 45000:    frac=0.453  ← getting worse
Phase II iter 75000:    frac=0.195  ← catastrophic collapse
Phase II iter 90000:    frac=0.334  ← partial recovery
Final diagnostic:       frac=0.326  ← WORSE than warm-start
```

In contrast, the easy pipeline at null_prob=0.95 shows healthy convergence:
```
Post-Phase-I (pihat):   frac=0.327  ← bad warm-start
Phase II iter     0:    frac=0.323
Phase II iter  1000:    frac=0.729  ← rapid improvement
Phase II iter  5000:    frac=0.856
Phase II iter 30000:    frac=0.911
Phase II iter 120000:   frac=0.942
Final diagnostic:       frac=0.931  ← close to target 0.95
```

### 5.3 Mechanism: bias amplification in EM-style bootstrap

The Phase II bootstrap loop is an EM-style fixed-point iteration:

1. **E-step**: Infer latent U from true X using the current flow model
2. **M-step**: Build empirical π̂ from inferred U; train on fresh (X,U) from π̂

The instability arises because the flow network's posterior is **diffuse**:
it places some probability mass at ρ≠0 even for marks that are truly null.
This creates a systematic bias in the empirical π̂:

- Empirical π̂ **over-represents** non-null marks
- Training on this biased π̂ teaches the model to predict even more non-null
- Next iteration's π̂ is even more biased → **vicious cycle**

At null_prob=0.95, the bias is small (only 5% non-null signal) and the
feedback loop is stable. At null_prob=0.80, 20% non-null creates enough
signal to overwhelm the self-correction mechanism.

### 5.4 Potential fixes to investigate

1. **Regularised bootstrap**: Mix empirical π̂ with a prior that has high
   null_prob (e.g. 50% empirical + 50% prior with null_prob=0.9). This
   dampens the bias amplification.

2. **EMA of empirical distribution**: Instead of replacing π̂ each step,
   maintain a running exponential moving average of inferred U samples.
   Smooths out noise and reduces the feedback gain.

3. **Multiple flow steps per bootstrap**: Run several gradient steps on
   the same empirical π̂ before re-inferring U. Reduces the feedback
   frequency.

4. **Phase II learning rate schedule**: Use a fresh (non-decaying) LR for
   Phase II instead of continuing the cosine decay from Phase I. The current
   schedule decays LR from 0.001 to 0.0 over the combined 160k steps,
   leaving late Phase II with effectively zero LR.

5. **Larger inference batch**: Use more samples for the E-step (n_source)
   to reduce variance in the empirical π̂.

6. **Warm-restart optimizer**: Reset Adam moments at the Phase II transition
   to avoid momentum from the Phase I distribution carrying over.

---

## 6. Future extensions

* Replace MLP with 1-D ConvNet for the velocity network.
* Multiple gradient steps per Phase II iteration.
* Adaptive scheduling of `n_source` and `n_eachstep_samples`.
