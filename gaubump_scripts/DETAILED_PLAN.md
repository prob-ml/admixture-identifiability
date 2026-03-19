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

## 5. Future extensions

* Replace MLP with 1-D ConvNet for the velocity network.
* Multiple gradient steps per Phase II iteration.
* Adaptive scheduling of `n_source` and `n_eachstep_samples`.
