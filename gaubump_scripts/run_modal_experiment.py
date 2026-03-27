#!/usr/bin/env python3
"""Parameterised experiment runner for Gaussian-bump flow matching case studies.

Two-flow architecture: a V-flow MLP learns the joint distribution of the
binary gate vector V via rectified flow matching (rounded to {0,1} at
inference), and a U-flow MLP learns the mark parameters U = (rho, logsigma)
via masked rectified flow matching conditioned on V.

The shape function uses V directly:
    f(x) = V * rho * exp(-x^2 / (2 * exp(2*logsigma)))
Null marks have V=0, rho=0, logsigma=0 — they contribute exactly zero
to the observed signal.

**Masking**: the U-flow masks the interpolated state z_s, the target
velocity, and the predicted velocity at V=0 positions — not just the
loss.  This prevents the network from ever seeing or producing non-zero
values at null positions.

Each run produces a self-contained results subdirectory under
``gaubump_scripts/results/<case_name>/`` containing training logs,
GPU utilisation, and diagnostic plots.

The full pipeline (Phase I warm-up + Phase II bootstrap) never sees
ground-truth (X, V, U) triples.  Phase I trains both the V-flow
and U-flow on synthetic (X, V, U) drawn from the initial guess pihat.
Phase II uses only ground-truth X observations; V is sampled by the
bootstrap V-flow (then rounded), U is inferred by the bootstrap U-flow
conditioned on the rounded V, and the empirical distribution over
(V_hat, U_hat) is used to generate fresh training data.

**Phase II restarts both optimisers from scratch** — new learning-rate
schedules and fresh Adam moments — to avoid stale momentum from Phase I.

Usage:
    modal run gaubump_scripts/run_modal_experiment.py
"""

from __future__ import annotations

import modal

app = modal.App("gaubump-experiment")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]",
        "equinox",
        "optax",
        "matplotlib",
        "numpy",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
)
def run_training():
    """Run a full Phase I + Phase II experiment with configurable parameters."""
    import io
    import subprocess
    import threading
    import time as time_mod

    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ----------------------------------------------------------------
    # Logging helpers
    # ----------------------------------------------------------------
    log_lines: list[str] = []
    _t0 = time_mod.time()

    def log(msg: str):
        elapsed = time_mod.time() - _t0
        line = f"[{elapsed:7.1f}s] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    # ----------------------------------------------------------------
    # GPU utilization monitor
    # ----------------------------------------------------------------
    gpu_log_lines: list[str] = []
    _gpu_active = True

    def _poll_gpu():
        while _gpu_active:
            try:
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,utilization.gpu,utilization.memory,"
                        "memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if r.returncode == 0:
                    line = r.stdout.strip()
                    gpu_log_lines.append(line)
                    log(f"[GPU] {line}")
            except Exception:
                pass
            time_mod.sleep(3)

    gpu_thread = threading.Thread(target=_poll_gpu, daemon=True)
    gpu_thread.start()

    # ----------------------------------------------------------------
    # Device info
    # ----------------------------------------------------------------
    log(f"JAX devices: {jax.devices()}")
    log(f"JAX default backend: {jax.default_backend()}")

    # ---- model types ----
    from typing import NamedTuple

    class GaussianComponent(NamedTuple):
        weight: float
        mean: jnp.ndarray
        std: jnp.ndarray

    class ShapeDistribution(NamedTuple):
        null_prob: float
        components: list

    # ---- observation computation (V-binary shape function) ----

    def compute_X_batched(V, U, L):
        """Compute observations X from binary V and mark parameters U.

        Shape: f(x) = V * rho * exp(-x^2 / (2 * exp(2*logsigma)))
        """
        T_plus_1 = U.shape[1]
        T = T_plus_1 - 1
        n_obs = T - 2 * L + 1
        rho = U[:, :, 0]
        logsigma = U[:, :, 1]
        xs = jnp.arange(-L, L + 1, dtype=jnp.float32)
        logsigma = jnp.clip(logsigma, -4.0, 4.0)
        var = jnp.exp(2.0 * logsigma)
        shapes = (
            V[:, :, None]
            * rho[:, :, None]
            * jnp.exp(-xs[None, None, :] ** 2 / (2.0 * var[:, :, None]))
        )
        tau_grid = jnp.arange(T_plus_1)[:, None]
        dx_grid = jnp.arange(2 * L + 1)[None, :]
        obs_idx = tau_grid + (dx_grid - L) - L
        i_grid = jnp.arange(n_obs)[None, None, :]
        M = (obs_idx[:, :, None] == i_grid).astype(jnp.float32)
        M_flat = M.reshape(T_plus_1 * (2 * L + 1), n_obs)
        shapes_flat = shapes.reshape(U.shape[0], -1)
        X = shapes_flat @ M_flat
        return X

    # ---- sampling (JIT-compiled) — returns (X, V, U) tuples ----

    def _sample_XVU_mixture_impl(key, null_prob, weights, means, stds,
                                 L, T, n_samples):
        """Sample (X, V, U) from a mixture with binary V."""
        total = n_samples * (T + 1)
        k_null, k_which, k_gauss = jax.random.split(key, 3)
        is_null = jax.random.bernoulli(k_null, p=null_prob, shape=(total,))
        V_flat = 1.0 - is_null  # V=1 means active, V=0 means null

        comp_idx = jax.random.choice(
            k_which, weights.shape[0], shape=(total,), p=weights)
        chosen_mean = means[comp_idx]
        chosen_std = stds[comp_idx]
        z = jax.random.normal(k_gauss, shape=(total, 2))
        non_null = chosen_mean + chosen_std * z

        null_mark = jnp.zeros(2)  # V=0 → rho=0, logsigma=0
        marks = jnp.where(is_null[:, None], null_mark, non_null)

        V = V_flat.reshape(n_samples, T + 1)
        U = marks.reshape(n_samples, T + 1, 2)
        X = compute_X_batched(V, U, L)
        return X, V, U

    sample_XVU_mixture = jax.jit(
        _sample_XVU_mixture_impl, static_argnums=(5, 6, 7))

    def _sample_XVU_empirical_impl(key, V_samples, U_samples, L, T,
                                   n_samples):
        """Sample (X, V, U) from empirical (V, U) pairs."""
        total = n_samples * (T + 1)
        idxs = jax.random.randint(
            key, shape=(total,), minval=0, maxval=V_samples.shape[0])
        V_flat = V_samples[idxs]
        marks = U_samples[idxs]
        V = V_flat.reshape(n_samples, T + 1)
        U = marks.reshape(n_samples, T + 1, 2)
        X = compute_X_batched(V, U, L)
        return X, V, U

    sample_XVU_empirical = jax.jit(
        _sample_XVU_empirical_impl, static_argnums=(3, 4, 5))

    def stack_mixture_params(pi):
        """Pre-stack mixture params into JAX arrays."""
        weights = jnp.array([c.weight for c in pi.components])
        weights = weights / weights.sum()
        means = jnp.stack([c.mean for c in pi.components])
        stds = jnp.stack([c.std for c in pi.components])
        return pi.null_prob, weights, means, stds

    # ---- networks ----
    import equinox as eqx
    import optax

    class VFlowMLP(eqx.Module):
        """MLP for flow matching on V (binary gate vector).

        Learns the joint distribution over all V positions via rectified
        flow matching.  At inference the ODE output is rounded to {0, 1}.
        """
        layers: list
        v_dim: int = eqx.field(static=True)
        x_dim: int = eqx.field(static=True)

        def __init__(self, T, L, hidden_dims, *, key):
            v_dim = T + 1
            x_dim = T - 2 * L + 1
            in_dim = v_dim + x_dim + 1  # noisy_V + X + s
            self.v_dim = v_dim
            self.x_dim = x_dim
            dims = [in_dim] + hidden_dims + [v_dim]
            keys = jax.random.split(key, len(dims) - 1)
            self.layers = []
            for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
                self.layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))

        def __call__(self, noisy_V, X, s):
            s = jnp.atleast_1d(s)
            inp = jnp.concatenate([noisy_V, X, s])
            x = inp
            for layer in self.layers[:-1]:
                x = jax.nn.gelu(layer(x))
            return self.layers[-1](x)

    class VelocityMLP(eqx.Module):
        """MLP for flow matching, conditioned on (X, V, s)."""
        layers: list
        u_dim: int = eqx.field(static=True)
        x_dim: int = eqx.field(static=True)
        v_dim: int = eqx.field(static=True)

        def __init__(self, T, L, hidden_dims, *, key):
            u_dim = (T + 1) * 2
            x_dim = T - 2 * L + 1
            v_dim = T + 1
            in_dim = u_dim + x_dim + v_dim + 1
            self.u_dim = u_dim
            self.x_dim = x_dim
            self.v_dim = v_dim
            dims = [in_dim] + hidden_dims + [u_dim]
            keys = jax.random.split(key, len(dims) - 1)
            self.layers = []
            for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
                self.layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))

        def __call__(self, noisy_U_flat, X, V, s):
            s = jnp.atleast_1d(s)
            inp = jnp.concatenate([noisy_U_flat, X, V, s])
            x = inp
            for layer in self.layers[:-1]:
                x = jax.nn.gelu(layer(x))
            x = self.layers[-1](x)
            return x

    # ---- V-flow matching loss ----

    def v_flow_loss_single(v_model, V, X, key):
        """Rectified flow matching loss for one (X, V) pair."""
        k1, k2 = jax.random.split(key)
        s = jax.random.uniform(k1, shape=())
        eps = jax.random.normal(k2, shape=V.shape)
        z_s = (1.0 - s) * eps + s * V
        target = V - eps
        pred = v_model(z_s, X, s)
        return jnp.mean((pred - target) ** 2)

    @eqx.filter_value_and_grad
    def v_flow_loss_batch(v_model, V, X, keys):
        losses = jax.vmap(
            lambda v, x, k: v_flow_loss_single(v_model, v, x, k)
        )(V, X, keys)
        return jnp.mean(losses)

    # ---- masked flow matching loss ----

    def flow_matching_loss_single(model, U_flat, X, V, key):
        k1, k2 = jax.random.split(key)
        s = jax.random.uniform(k1, shape=())
        eps = jax.random.normal(k2, shape=U_flat.shape)
        V_mask = jnp.repeat(V, 2)
        # Mask interpolated state and target — null positions stay at zero
        z_s = ((1.0 - s) * eps + s * U_flat) * V_mask
        target = (U_flat - eps) * V_mask
        pred = model(z_s, X, V, s)
        pred = pred * V_mask  # mask predicted velocity
        sq_err = (pred - target) ** 2
        n_active = jnp.maximum(V_mask.sum(), 1.0)
        return sq_err.sum() / n_active

    @eqx.filter_value_and_grad
    def flow_matching_loss_batch(model, U, X, V, keys):
        batch = U.shape[0]
        U_flat = U.reshape(batch, -1)
        losses = jax.vmap(
            lambda u, x, v, k: flow_matching_loss_single(model, u, x, v, k)
        )(U_flat, X, V, keys)
        return jnp.mean(losses)

    # ---- flow sampling (conditioned on V) ----

    def sample_flow(model, X, V, key, n_steps=20):
        V_mask = jnp.repeat(V, 2)
        z = jax.random.normal(key, shape=(model.u_dim,))
        z = z * V_mask  # start from masked noise
        dt = 1.0 / n_steps

        def euler_step(i, z):
            s = i * dt
            v = model(z, X, V, jnp.array(s))
            v = v * V_mask  # mask dynamics
            return z + dt * v

        z = jax.lax.fori_loop(0, n_steps, euler_step, z)
        # Clip logsigma
        logsigma_mask = jnp.tile(jnp.array([0.0, 1.0]), model.u_dim // 2)
        z = jnp.where(logsigma_mask, jnp.clip(z, -4.0, 4.0), z)
        z = z * V_mask  # final mask
        return z

    @eqx.filter_jit
    def sample_flow_batch(model, X_batch, V_batch, key, n_steps=20):
        batch = X_batch.shape[0]
        keys = jax.random.split(key, batch)
        T_plus_1 = model.u_dim // 2
        U_flat = jax.vmap(
            lambda x, v, k: sample_flow(model, x, v, k, n_steps)
        )(X_batch, V_batch, keys)
        return U_flat.reshape(batch, T_plus_1, 2)

    # ---- V-flow sampling (ODE → round to {0, 1}) ----

    def sample_v_flow(v_model, X, key, n_steps=20):
        """Sample V by integrating the V-flow ODE, then round to {0, 1}."""
        z = jax.random.normal(key, shape=(v_model.v_dim,))
        dt = 1.0 / n_steps

        def euler_step(i, z):
            s = i * dt
            vel = v_model(z, X, jnp.array(s))
            return z + dt * vel

        z = jax.lax.fori_loop(0, n_steps, euler_step, z)
        return (z > 0.5).astype(jnp.float32)

    @eqx.filter_jit
    def sample_v_flow_batch(v_model, X_batch, key, n_steps=20):
        """Sample V for a batch of observations."""
        batch = X_batch.shape[0]
        keys = jax.random.split(key, batch)
        return jax.vmap(
            lambda x, k: sample_v_flow(v_model, x, k, n_steps)
        )(X_batch, keys)

    # ---- update steps ----

    @eqx.filter_jit
    def update_step_u(model, opt_state, optimizer, U, X, V, key):
        batch = X.shape[0]
        keys = jax.random.split(key, batch)
        loss, grads = flow_matching_loss_batch(model, U, X, V, keys)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    @eqx.filter_jit
    def update_step_v(v_model, opt_state, optimizer, X, V, key):
        batch = X.shape[0]
        keys = jax.random.split(key, batch)
        loss, grads = v_flow_loss_batch(v_model, V, X, keys)
        updates, new_opt_state = optimizer.update(grads, opt_state, v_model)
        new_model = eqx.apply_updates(v_model, updates)
        return new_model, new_opt_state, loss

    # ---- eval functions ----

    @eqx.filter_jit
    def eval_flow_loss(model, U, X, V, key):
        """Evaluate masked flow matching loss on a batch WITHOUT gradients."""
        batch = U.shape[0]
        U_flat = U.reshape(batch, -1)
        keys = jax.random.split(key, batch)
        losses = jax.vmap(
            lambda u, x, v, k: flow_matching_loss_single(model, u, x, v, k)
        )(U_flat, X, V, keys)
        return jnp.mean(losses)

    @eqx.filter_jit
    def eval_v_accuracy(v_model, X, V, key):
        """Sample V from V-flow and compute accuracy vs true V."""
        V_pred = sample_v_flow_batch(v_model, X, key)
        return jnp.mean(V_pred == V)

    # ================================================================
    # ██  CASE CONFIGURATION  ██
    # ================================================================
    case_name = "case_v_flow_masked_np95_pihat95"

    L = 3
    T = 20
    n_phase1_steps = 10_000
    n_phase2_steps = 150_000
    n_init_samples = 256
    n_source = 256
    n_eachstep_samples = 256
    lr = 1e-3
    hidden_dims = [256, 256, 256]
    seed = 42

    # ---- Ground-truth distribution ----
    null_prob_true = 0.95
    pi_true = ShapeDistribution(
        null_prob=null_prob_true,
        components=[
            GaussianComponent(
                weight=0.7,
                mean=jnp.array([1.0, 0.0]),
                std=jnp.array([0.3, 0.3]),
            ),
            GaussianComponent(
                weight=0.3,
                mean=jnp.array([2.0, -0.5]),
                std=jnp.array([0.2, 0.2]),
            ),
        ],
    )

    # ---- Initial guess (pihat) for Phase I ----
    null_prob_init = 0.95
    pihat = ShapeDistribution(
        null_prob=null_prob_init,
        components=[
            GaussianComponent(
                weight=1.0,
                mean=jnp.array([1.0, 0.0]),
                std=jnp.array([1.0, 1.0]),
            ),
        ],
    )
    # ================================================================

    log(f"=== Case: {case_name} ===")
    log(f"  null_prob_true = {null_prob_true}")
    log(f"  null_prob_init = {null_prob_init}")
    log(f"  pi_true components: {len(pi_true.components)}")
    log(f"  pihat  components: {len(pihat.components)}")

    key = jax.random.PRNGKey(seed)

    # Initialise networks
    key, v_model_key, u_model_key = jax.random.split(key, 3)
    v_model = VFlowMLP(T=T, L=L, hidden_dims=hidden_dims,
                       key=v_model_key)
    u_model = VelocityMLP(T=T, L=L, hidden_dims=hidden_dims,
                          key=u_model_key)

    # Phase I schedules
    schedule_v_p1 = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase1_steps)
    optimizer_v_p1 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_v_p1),
    )
    opt_state_v = optimizer_v_p1.init(v_model)

    schedule_u_p1 = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase1_steps)
    optimizer_u_p1 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_u_p1),
    )
    opt_state_u = optimizer_u_p1.init(u_model)

    # Pre-stack mixture parameters
    np_hat, w_hat, m_hat, s_hat = stack_mixture_params(pihat)
    np_true, w_true, m_true, s_true = stack_mixture_params(pi_true)

    # ================================================================
    # JIT warmup
    # ================================================================
    log("=== JIT warmup ===")
    t_warmup = time_mod.time()

    key, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 9)
    _X1, _V1, _U1 = sample_XVU_mixture(
        k1, np_hat, w_hat, m_hat, s_hat, L, T, n_init_samples)
    _X2, _V2, _ = sample_XVU_mixture(
        k2, np_true, w_true, m_true, s_true, L, T, n_source)
    _, _, _loss_u = update_step_u(
        u_model, opt_state_u, optimizer_u_p1, _U1, _X1, _V1, k3)
    _, _, _loss_v = update_step_v(
        v_model, opt_state_v, optimizer_v_p1, _X1, _V1, k4)
    # Warm up V-flow sampling
    _V_pred = sample_v_flow_batch(v_model, _X2, k5)
    _Uinf = sample_flow_batch(u_model, _X2, _V_pred, k6)
    _V_flat = _V_pred.reshape(-1)
    _U_flat = _Uinf.reshape(-1, 2)
    _X3, _V3, _U3 = sample_XVU_empirical(
        k7, _V_flat, _U_flat, L, T, n_eachstep_samples)
    _acc = eval_v_accuracy(v_model, _X1, _V1, k8)
    jax.block_until_ready((_X1, _X2, _X3, _loss_u, _loss_v, _Uinf, _acc))

    dt_warmup = time_mod.time() - t_warmup
    log(f"JIT warmup complete in {dt_warmup:.1f}s")

    # ================================================================
    # Fixed evaluation batch (ground-truth)
    # ================================================================
    key, k_eval_data = jax.random.split(key)
    n_eval = 512
    X_eval_gt, V_eval_gt, U_eval_gt = sample_XVU_mixture(
        k_eval_data, np_true, w_true, m_true, s_true, L, T, n_eval)
    eval_key_fixed = jax.random.PRNGKey(999)

    # JIT-warm the eval functions
    _eval_loss = eval_flow_loss(
        u_model, U_eval_gt, X_eval_gt, V_eval_gt, eval_key_fixed)
    _eval_acc = eval_v_accuracy(v_model, X_eval_gt, V_eval_gt,
                                eval_key_fixed)
    jax.block_until_ready((_eval_loss, _eval_acc))
    log(f"Eval warmed up (initial GT U-loss={float(_eval_loss):.4f}, "
        f"V-acc={float(_eval_acc):.4f})")

    # Loss tracking
    p1_steps_loss: list[int] = []
    p1_u_train_losses: list[float] = []
    p1_u_gt_losses: list[float] = []
    p1_v_losses: list[float] = []
    p2_steps_loss: list[int] = []
    p2_gt_losses: list[float] = []
    p2_oracle_gt_losses: list[float] = []
    eval_interval_p1 = 200
    eval_interval_p2 = 2500

    # ================================================================
    # Generate X-data diagnostic plots BEFORE training
    # ================================================================
    log("=== Generating X-data diagnostic plots ===")
    key, k_xdata_true, k_xdata_pihat = jax.random.split(key, 3)
    n_xdata = 8

    X_show_true, V_show_true, U_show_true = sample_XVU_mixture(
        k_xdata_true, np_true, w_true, m_true, s_true, L, T, n_xdata)
    X_show_true_np = np.array(X_show_true)
    V_show_true_np = np.array(V_show_true)
    U_show_true_np = np.array(U_show_true)

    X_show_pihat, V_show_pihat, U_show_pihat = sample_XVU_mixture(
        k_xdata_pihat, np_hat, w_hat, m_hat, s_hat, L, T, n_xdata)
    X_show_pihat_np = np.array(X_show_pihat)
    V_show_pihat_np = np.array(V_show_pihat)
    U_show_pihat_np = np.array(U_show_pihat)

    results: dict[str, bytes] = {}

    def _plot_xdata(X_np, V_np, U_np, suptitle, n_show):
        """Plot X observations with rug-plots marking V=1 positions."""
        n_obs = X_np.shape[1]
        T_plus_1 = U_np.shape[1]
        obs_grid = np.arange(n_obs)

        n_cols = min(4, n_show)
        n_rows = (n_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 3.5 * n_rows),
                                 squeeze=False)
        for idx in range(n_show):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.bar(obs_grid, X_np[idx], width=0.8, alpha=0.6,
                   color="tab:blue", label="X")

            # Rug-plot: mark time-positions where V=1
            v_row = V_np[idx]      # shape (T+1,)
            rhos = U_np[idx, :, 0]
            for tau in range(T_plus_1):
                if v_row[tau] > 0.5:
                    obs_pos = tau - L
                    if 0 <= obs_pos < n_obs:
                        ax.axvline(obs_pos, color="red", alpha=0.6,
                                   lw=1.5, ls="--")
                        ax.text(obs_pos, ax.get_ylim()[1] * 0.95,
                                f"r={rhos[tau]:.2f}",
                                fontsize=6, color="red",
                                ha="center", va="top", rotation=90)

            nonnull_count = int(np.sum(v_row > 0.5))
            ax.set_title(f"Sample {idx}  ({nonnull_count} non-null marks)",
                         fontsize=9)
            ax.set_xlabel("obs index")
            ax.set_ylabel("X value")

        for idx in range(n_show, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        return fig

    fig_xtrue = _plot_xdata(
        X_show_true_np, V_show_true_np, U_show_true_np,
        f"True-model X data  (null_prob={null_prob_true},"
        f"  L={L}, T={T})",
        n_xdata,
    )
    buf = io.BytesIO()
    fig_xtrue.savefig(buf, format="png", dpi=150)
    plt.close(fig_xtrue)
    results["xdata_true.png"] = buf.getvalue()

    fig_xpihat = _plot_xdata(
        X_show_pihat_np, V_show_pihat_np, U_show_pihat_np,
        f"Pihat (Phase I) X data  (null_prob={null_prob_init},"
        f"  L={L}, T={T})",
        n_xdata,
    )
    buf = io.BytesIO()
    fig_xpihat.savefig(buf, format="png", dpi=150)
    plt.close(fig_xpihat)
    results["xdata_pihat.png"] = buf.getvalue()

    log("  X-data plots saved")

    # ================================================================
    # Phase I — warm-up on pihat (train BOTH V-flow and U-flow)
    # ================================================================
    log(f"=== Phase I ({n_phase1_steps} steps, "
        f"pihat null_prob={null_prob_init}) ===")
    log_interval_p1 = max(1, n_phase1_steps // 20)
    t_phase1 = time_mod.time()

    for step in range(n_phase1_steps):
        key, k_data, k_step_u, k_step_v = jax.random.split(key, 4)
        X, V, U = sample_XVU_mixture(
            k_data, np_hat, w_hat, m_hat, s_hat, L, T, n_init_samples)
        u_model, opt_state_u, loss_u = update_step_u(
            u_model, opt_state_u, optimizer_u_p1, U, X, V, k_step_u)
        v_model, opt_state_v, loss_v = update_step_v(
            v_model, opt_state_v, optimizer_v_p1, X, V, k_step_v)
        if step % log_interval_p1 == 0 or step == n_phase1_steps - 1:
            log(f"  step {step:5d}  U-loss={float(loss_u):.6f}"
                f"  V-loss={float(loss_v):.6f}")
        if step % eval_interval_p1 == 0 or step == n_phase1_steps - 1:
            key, k_eval_v = jax.random.split(key)
            gt_u_loss = float(eval_flow_loss(
                u_model, U_eval_gt, X_eval_gt, V_eval_gt, eval_key_fixed))
            v_acc = float(eval_v_accuracy(
                v_model, X_eval_gt, V_eval_gt, k_eval_v))
            p1_steps_loss.append(step)
            p1_u_train_losses.append(float(loss_u))
            p1_u_gt_losses.append(gt_u_loss)
            p1_v_losses.append(float(loss_v))
            if step % log_interval_p1 != 0:
                log(f"  step {step:5d}  U-loss={float(loss_u):.6f}"
                    f"  gt_U-loss={gt_u_loss:.6f}"
                    f"  V-acc={v_acc:.4f}")

    jax.block_until_ready((loss_u, loss_v))
    dt_p1 = time_mod.time() - t_phase1
    log(f"Phase I complete in {dt_p1:.1f}s "
        f"({dt_p1 / n_phase1_steps * 1000:.1f} ms/step)")

    # ================================================================
    # Post-Phase-I diagnostic
    # ================================================================
    log("=== Post-Phase-I diagnostic ===")
    key, k_p1d_x, k_p1d_v, k_p1d_inf = jax.random.split(key, 4)
    n_p1_diag = 512
    X_p1d, V_p1d_true, U_p1d_true = sample_XVU_mixture(
        k_p1d_x, np_true, w_true, m_true, s_true, L, T, n_p1_diag)

    # Sample V with V-flow (round to {0,1})
    V_p1d_pred = sample_v_flow_batch(v_model, X_p1d, k_p1d_v)

    # Infer U with flow conditioned on predicted V
    U_p1d_inferred = jax.lax.stop_gradient(
        sample_flow_batch(u_model, X_p1d, V_p1d_pred, k_p1d_inf))

    V_p1d_flat = V_p1d_pred.reshape(-1)
    U_p1d_flat = U_p1d_inferred.reshape(-1, 2)
    frac_null_p1 = float(jnp.mean(V_p1d_flat == 0.0))
    v_acc_p1 = float(jnp.mean(V_p1d_pred == V_p1d_true))
    log(f"  After Phase I (trained on pihat null_prob={null_prob_init}):")
    log(f"  frac(V=0)       = {frac_null_p1:.3f}  (target={null_prob_true})")
    log(f"  V-classifier acc = {v_acc_p1:.3f}")

    # ================================================================
    # Phase II — iterative refinement
    # ================================================================
    log(f"=== Phase II ({n_phase2_steps} iterations, "
        f"true null_prob={null_prob_true}) ===")
    log("  [NOTE] Both optimisers restarted: fresh cosine schedule + "
        "Adam moments")

    # Bootstrap models: V-flow + U-flow (initialized from Phase I)
    schedule_v_p2 = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase2_steps)
    optimizer_v_p2 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_v_p2),
    )
    opt_state_v = optimizer_v_p2.init(v_model)

    schedule_u_p2 = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase2_steps)
    optimizer_u_p2 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_u_p2),
    )
    opt_state_u = optimizer_u_p2.init(u_model)

    # Oracle models: fresh V-flow + fresh U-flow (from scratch)
    key, oracle_v_key, oracle_u_key = jax.random.split(key, 3)
    oracle_v_model = VFlowMLP(T=T, L=L, hidden_dims=hidden_dims,
                              key=oracle_v_key)
    oracle_u_model = VelocityMLP(T=T, L=L, hidden_dims=hidden_dims,
                                 key=oracle_u_key)

    schedule_oracle_v = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase2_steps)
    optimizer_oracle_v = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_oracle_v),
    )
    oracle_opt_state_v = optimizer_oracle_v.init(oracle_v_model)

    schedule_oracle_u = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase2_steps)
    optimizer_oracle_u = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_oracle_u),
    )
    oracle_opt_state_u = optimizer_oracle_u.init(oracle_u_model)
    log("  [Oracle] Fresh V-flow + U-flow initialised from scratch")

    t_phase2 = time_mod.time()

    def should_log_p2(step):
        if step < 1000:
            return step % 100 == 0
        if step < 10000:
            return step % 1000 == 0
        return step % 7500 == 0 or step == n_phase2_steps - 1

    for step in range(n_phase2_steps):
        key, k_true, k_v_sample, k_infer, k_data, k_step_u, k_step_v, k_oracle = (
            jax.random.split(key, 8))

        # (a) Draw X from true distribution (discard true V, U)
        X_true, _, _ = sample_XVU_mixture(
            k_true, np_true, w_true, m_true, s_true, L, T, n_source)

        # (b) Sample V_hat with bootstrap V-flow (round to {0,1})
        V_hat = jax.lax.stop_gradient(
            sample_v_flow_batch(v_model, X_true, k_v_sample))

        # (c) Infer U_hat with bootstrap U-flow conditioned on V_hat
        U_inferred = jax.lax.stop_gradient(
            sample_flow_batch(u_model, X_true, V_hat, k_infer)
        )
        U_inferred = jnp.where(
            jnp.isfinite(U_inferred), U_inferred,
            jnp.array([0.0, 0.0]))

        # (d) Sample from empirical (V_hat, U_hat) to get training data
        V_flat = V_hat.reshape(-1)
        U_flat = U_inferred.reshape(-1, 2)
        X_new, V_new, U_new = sample_XVU_empirical(
            k_data, V_flat, U_flat, L, T, n_eachstep_samples)

        # (e) Train bootstrap V-flow and U-flow
        u_model, opt_state_u, loss_u = update_step_u(
            u_model, opt_state_u, optimizer_u_p2, U_new, X_new, V_new,
            k_step_u)
        v_model, opt_state_v, loss_v = update_step_v(
            v_model, opt_state_v, optimizer_v_p2, X_new, V_new, k_step_v)

        # (f) Train oracle V-flow and U-flow on ground-truth
        k_oracle_data, k_oracle_step_u, k_oracle_step_v = (
            jax.random.split(k_oracle, 3))
        X_oracle, V_oracle, U_oracle = sample_XVU_mixture(
            k_oracle_data, np_true, w_true, m_true, s_true,
            L, T, n_eachstep_samples)
        oracle_u_model, oracle_opt_state_u, oracle_loss_u = update_step_u(
            oracle_u_model, oracle_opt_state_u, optimizer_oracle_u,
            U_oracle, X_oracle, V_oracle, k_oracle_step_u)
        oracle_v_model, oracle_opt_state_v, oracle_loss_v = update_step_v(
            oracle_v_model, oracle_opt_state_v, optimizer_oracle_v,
            X_oracle, V_oracle, k_oracle_step_v)

        if should_log_p2(step):
            frac_null = float(jnp.mean(V_flat == 0.0))
            current_lr = float(schedule_u_p2(step))
            log(
                f"  iter {step:6d}  U-loss={float(loss_u):.6f}  "
                f"V-loss={float(loss_v):.6f}  "
                f"frac(V=0)={frac_null:.3f}  "
                f"lr={current_lr:.6f}"
            )

        if step % eval_interval_p2 == 0 or step == n_phase2_steps - 1:
            gt_loss = float(eval_flow_loss(
                u_model, U_eval_gt, X_eval_gt, V_eval_gt, eval_key_fixed))
            oracle_gt_loss = float(eval_flow_loss(
                oracle_u_model, U_eval_gt, X_eval_gt, V_eval_gt,
                eval_key_fixed))
            p2_steps_loss.append(step)
            p2_gt_losses.append(gt_loss)
            p2_oracle_gt_losses.append(oracle_gt_loss)
            if not should_log_p2(step):
                log(f"  iter {step:6d}  gt_U-loss={gt_loss:.6f}"
                    f"  oracle_gt_U-loss={oracle_gt_loss:.6f}")
            else:
                log(f"  iter {step:6d}  [eval] bootstrap_gt={gt_loss:.6f}"
                    f"  oracle_gt={oracle_gt_loss:.6f}")

    jax.block_until_ready((loss_u, loss_v))
    dt_p2 = time_mod.time() - t_phase2
    log(f"Phase II complete in {dt_p2:.1f}s "
        f"({dt_p2 / n_phase2_steps * 1000:.1f} ms/step)")

    # ================================================================
    # Final diagnostic inference
    # ================================================================
    log("=== Final diagnostic inference ===")
    key, k_diag_true, k_diag_v, k_diag_infer = jax.random.split(key, 4)
    n_diag = 1024
    X_diag, V_diag_true, U_diag_true = sample_XVU_mixture(
        k_diag_true, np_true, w_true, m_true, s_true, L, T, n_diag)

    # Sample V with V-flow
    V_diag_pred = sample_v_flow_batch(v_model, X_diag, k_diag_v)

    # Infer U conditioned on predicted V
    U_diag_inferred = jax.lax.stop_gradient(
        sample_flow_batch(u_model, X_diag, V_diag_pred, k_diag_infer)
    )
    V_diag_flat = V_diag_pred.reshape(-1)
    U_diag_flat = U_diag_inferred.reshape(-1, 2)
    V_true_flat = V_diag_true.reshape(-1)
    U_true_flat = U_diag_true.reshape(-1, 2)

    frac_v_zero = float(jnp.mean(V_diag_flat == 0.0))
    v_acc_final = float(jnp.mean(V_diag_pred == V_diag_true))
    log(f"  n_diag={n_diag}, total marks={V_diag_flat.shape[0]}")
    log(f"  frac(V=0) inferred   = {frac_v_zero:.3f}")
    log(f"  true null_prob       = {null_prob_true}")
    log(f"  V-flow acc           = {v_acc_final:.3f}")
    log(f"  post-Phase-I frac(V=0) = {frac_null_p1:.3f}")
    log(f"  post-Phase-I V-acc     = {v_acc_p1:.3f}")
    log(f"  total wall time        = {time_mod.time() - _t0:.1f}s")

    # ================================================================
    # Stop GPU monitor
    # ================================================================
    _gpu_active = False
    gpu_thread.join(timeout=5)

    # ================================================================
    # Generate training-result plots
    # ================================================================
    V_inf_np = np.array(V_diag_flat)
    U_inf_np = np.array(U_diag_flat)
    V_true_np = np.array(V_true_flat)
    U_true_np = np.array(U_true_flat)
    V_p1d_np = np.array(V_p1d_flat)
    U_p1d_np = np.array(U_p1d_flat)

    # Reference samples (single-mark)
    key, k_ref = jax.random.split(key)
    _, ref_V, ref_U = sample_XVU_mixture(
        k_ref, np_true, w_true, m_true, s_true, L=0, T=0, n_samples=10000)
    ref_V_np = np.array(ref_V.reshape(-1))
    ref_U_np = np.array(ref_U.reshape(-1, 2))

    # ---- Plot: Scatter of aggregate posterior vs ground truth ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Helper to plot V=0 and V=1 marks separately
    def _scatter_VU(ax, V_np, U_np, title, color_active, color_null,
                    label_active, label_null):
        active = V_np > 0.5
        null = ~active
        if null.sum() > 0:
            ax.scatter(U_np[null, 0], U_np[null, 1],
                       s=1, alpha=0.2, c=color_null, label=label_null)
        if active.sum() > 0:
            ax.scatter(U_np[active, 0], U_np[active, 1],
                       s=1, alpha=0.3, c=color_active, label=label_active)
        ax.set_xlabel("ρ")
        ax.set_ylabel("log σ")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3, 3)

    ax = axes[0]
    frac_ref = f"{np.mean(ref_V_np < 0.5):.1%}"
    _scatter_VU(ax, ref_V_np, ref_U_np,
                f"Ground-truth π\n(null_prob={null_prob_true}, "
                f"frac(V=0)={frac_ref})",
                "tab:blue", "gray", "V=1 (active)", "V=0 (null)")

    ax = axes[1]
    frac_p1_str = f"{frac_null_p1:.1%}"
    _scatter_VU(ax, V_p1d_np, U_p1d_np,
                f"Post-Phase-I inference\nfrac(V=0)={frac_p1_str}  "
                f"V-acc={v_acc_p1:.1%}",
                "tab:green", "gray", "V=1 (active)", "V=0 (null)")

    ax = axes[2]
    frac_str = f"{frac_v_zero:.1%}"
    _scatter_VU(ax, V_inf_np, U_inf_np,
                f"Final (post-Phase-II)\nfrac(V=0)={frac_str}  "
                f"V-acc={v_acc_final:.1%}",
                "tab:orange", "gray", "V=1 (active)", "V=0 (null)")

    fig.suptitle(
        f"{case_name}  L={L}, T={T}, seed={seed}\n"
        f"null_prob_true={null_prob_true}, null_prob_init={null_prob_init}, "
        f"phase1={n_phase1_steps}, phase2={n_phase2_steps}  "
        f"[V-flow + masked U-flow, Phase II: fresh optimisers]",
        fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["posterior_vs_truth.png"] = buf.getvalue()

    # ---- Plot: Marginal histogram of rho (V=1 marks only) ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    rho_ref_active = ref_U_np[ref_V_np > 0.5, 0]
    if len(rho_ref_active) > 0:
        ax.hist(rho_ref_active, bins=80, density=True,
                alpha=0.6, color="tab:blue", label="true π (V=1)")
    ax.set_xlabel("ρ")
    ax.set_ylabel("density")
    ax.set_title("Ground-truth marginal of ρ (V=1 only)")
    ax.legend(fontsize=8)

    ax = axes[1]
    rho_p1_active = U_p1d_np[V_p1d_np > 0.5, 0]
    rho_p1_active = rho_p1_active[np.isfinite(rho_p1_active)]
    if len(rho_p1_active) > 0:
        ax.hist(rho_p1_active, bins=80, density=True,
                alpha=0.6, color="tab:green", label="post-Phase-I (V=1)")
    ax.set_xlabel("ρ")
    ax.set_ylabel("density")
    ax.set_title(f"Post-Phase-I marginal of ρ (V=1)\n"
                 f"frac(V=0)={frac_p1_str}")
    ax.legend(fontsize=8)

    ax = axes[2]
    rho_inf_active = U_inf_np[V_inf_np > 0.5, 0]
    rho_inf_active = rho_inf_active[np.isfinite(rho_inf_active)]
    if len(rho_inf_active) > 0:
        ax.hist(rho_inf_active, bins=80, density=True,
                alpha=0.6, color="tab:orange", label="final inferred (V=1)")
    ax.set_xlabel("ρ")
    ax.set_ylabel("density")
    ax.set_title(f"Final marginal of ρ (V=1)\nfrac(V=0)={frac_str}")
    ax.legend(fontsize=8)

    fig.suptitle(f"Marginal ρ distributions — V=1 marks only ({case_name})",
                 fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["rho_marginal.png"] = buf.getvalue()

    # ---- Plot: True vs inferred latent (V, U) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _scatter_VU(axes[0], V_true_np, U_true_np,
                f"True latent (V, U) ({n_diag} observations)",
                "tab:green", "gray", "V=1", "V=0")
    _scatter_VU(axes[1], V_inf_np, U_inf_np,
                f"Inferred (V, U) ({n_diag} observations)",
                "tab:orange", "gray", "V=1", "V=0")

    fig.suptitle(f"True latent vs inferred latent ({case_name})", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["true_vs_inferred_U.png"] = buf.getvalue()

    # ---- Plot: Survival function of rho for V=1 marks ----
    def _survival(arr):
        s = np.sort(arr)
        surv = 1.0 - np.arange(1, len(s) + 1) / len(s)
        return s, surv

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(rho_ref_active) > 0:
        x_s, y_s = _survival(rho_ref_active)
        ax.plot(x_s, y_s, label="true π (V=1)", color="tab:blue", lw=1.5)
    if len(rho_p1_active) > 0:
        x_s, y_s = _survival(rho_p1_active)
        ax.plot(x_s, y_s, label="post-Phase-I (V=1)", color="tab:green",
                lw=1.5, ls="--")
    if len(rho_inf_active) > 0:
        x_s, y_s = _survival(rho_inf_active)
        ax.plot(x_s, y_s, label="final (post-Phase-II, V=1)",
                color="tab:orange", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("ρ")
    ax.set_ylabel("P(ρ > t) for V=1 marks")
    ax.set_title(
        f"Empirical survival of ρ for V=1 marks  ({case_name})\n"
        f"null_prob={null_prob_true}, post-P1 frac(V=0)={frac_p1_str}, "
        f"final frac(V=0)={frac_str}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-4, top=1.0)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["rho_survival.png"] = buf.getvalue()

    # ---- Plot: Loss curves ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Phase I: training U-flow loss, GT eval U-flow loss, V-classifier loss
    ax1.plot(p1_steps_loss, p1_u_train_losses,
             label="Training U-flow loss (π̂ data)", color="tab:blue",
             alpha=0.7)
    ax1.plot(p1_steps_loss, p1_u_gt_losses,
             label="GT eval U-flow loss", color="tab:orange", lw=2)
    ax1.plot(p1_steps_loss, p1_v_losses,
             label="V-flow loss", color="tab:red", lw=1.5, ls="--")
    ax1.set_xlabel("Phase I step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Phase I: Warm-up on π̂ (U-flow + V-flow)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Phase II: bootstrap vs oracle masked-U-flow GT loss
    ax2.plot(p2_steps_loss, p2_gt_losses,
             label="Bootstrap model (GT eval U-flow)",
             color="tab:orange", lw=2)
    ax2.plot(p2_steps_loss, p2_oracle_gt_losses,
             label="Oracle model (GT eval U-flow)",
             color="tab:green", lw=2, ls="--")
    ax2.set_xlabel("Phase II step")
    ax2.set_ylabel("Masked U-flow loss")
    ax2.set_title("Phase II: Bootstrap vs Oracle (GT eval, masked U-flow)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Smart y-limits for Phase II
    p2_boot = np.array(p2_gt_losses)
    p2_orac = np.array(p2_oracle_gt_losses)
    if len(p2_boot) > 0 and len(p2_orac) > 0:
        ylim_lo = min(p2_boot.min(), p2_orac.min()) * 0.95
        ylim_hi = min(p2_boot.max(), p2_orac.max()) * 1.05
        ax2.set_ylim(ylim_lo, ylim_hi)

    fig.suptitle(
        f"Loss curves ({case_name})\n"
        f"Eval on {n_eval} fixed ground-truth (X, V, U) triples",
        fontsize=12)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["loss_curves.png"] = buf.getvalue()

    # ---- Training log ----
    training_log = "\n".join(log_lines) + "\n"
    results["training_log.txt"] = training_log.encode("utf-8")

    # ---- GPU utilization log ----
    gpu_header = ("timestamp, utilization.gpu [%], utilization.memory [%], "
                  "memory.used [MiB], memory.total [MiB], temperature.gpu")
    gpu_utilization_log = gpu_header + "\n" + "\n".join(gpu_log_lines) + "\n"
    results["gpu_utilization.txt"] = gpu_utilization_log.encode("utf-8")

    log(f"Generated {len(results)} result files for case '{case_name}'.")
    return case_name, results


@app.local_entrypoint()
def main():
    import os

    case_name, result_files = run_training.remote()

    # Save to per-case subdirectory
    results_dir = os.path.join(os.path.dirname(__file__), "results", case_name)
    os.makedirs(results_dir, exist_ok=True)

    for name, data in result_files.items():
        path = os.path.join(results_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved {path} ({len(data)} bytes)")

    print(f"\nAll done!  Results in {results_dir}")
