#!/usr/bin/env python3
"""Parameterised experiment runner for Gaussian-bump flow matching case studies.

Each run produces a self-contained results subdirectory under
``gaubump_scripts/results/<case_name>/`` containing training logs,
GPU utilisation, and diagnostic plots — including visualisations of what
the true-X and pihat-X training data actually look like (with rug-plots
marking positions where the amplitude softplus(ρ) is appreciable).

The shape function uses a softplus nonlinearity:
    f(x) = softplus(rho) * exp(-x^2 / (2 * exp(2*logsigma)))
so that null marks (rho ≈ -10, softplus(-10) ≈ 0) produce only very faint
background noise rather than an exact zero.

The full pipeline (Phase I warm-up + Phase II bootstrap) never sees
ground-truth (X, U) pairs.  Phase I trains on synthetic (X, U) drawn
from the *initial guess* π̂.  Phase II uses only ground-truth X
observations; latent U is *inferred* by the current flow model, and
the empirical distribution over those inferred U is used to generate
fresh training (X, U) pairs.

**Phase II restarts the optimiser from scratch** — a new learning-rate
schedule and fresh Adam moments — to avoid any stale momentum from
Phase I.

Usage (example — first case study):

    modal run gaubump_scripts/run_modal_experiment.py

Edit the CASE CONFIGURATION block near the bottom of run_training() to
change parameters between runs.
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

    # ---- sampling (JIT-compiled) ----

    def _sample_XU_mixture_impl(key, null_prob, weights, means, stds, L, T, n_samples):
        """Pure-JAX mixture sampling; will be wrapped with jax.jit."""
        total = n_samples * (T + 1)
        k_null, k_which, k_gauss = jax.random.split(key, 3)
        is_null = jax.random.bernoulli(k_null, p=null_prob, shape=(total,))
        comp_idx = jax.random.choice(
            k_which, weights.shape[0], shape=(total,), p=weights)
        chosen_mean = means[comp_idx]
        chosen_std = stds[comp_idx]
        z = jax.random.normal(k_gauss, shape=(total, 2))
        non_null = chosen_mean + chosen_std * z
        null_ls = jax.random.normal(k_null, shape=(total,))
        null = jnp.stack([jnp.full(total, -10.0), null_ls], axis=-1)
        marks = jnp.where(is_null[:, None], null, non_null)
        U = marks.reshape(n_samples, T + 1, 2)
        X = compute_X_batched(U, L)
        return X, U

    sample_XU_mixture = jax.jit(
        _sample_XU_mixture_impl, static_argnums=(5, 6, 7))

    def _sample_XU_empirical_impl(key, samples, L, T, n_samples):
        """Pure-JAX empirical sampling; will be wrapped with jax.jit."""
        total = n_samples * (T + 1)
        idxs = jax.random.randint(
            key, shape=(total,), minval=0, maxval=samples.shape[0])
        marks = samples[idxs]
        U = marks.reshape(n_samples, T + 1, 2)
        X = compute_X_batched(U, L)
        return X, U

    sample_XU_empirical = jax.jit(
        _sample_XU_empirical_impl, static_argnums=(2, 3, 4))

    def stack_mixture_params(pi):
        """Pre-stack mixture params into JAX arrays."""
        weights = jnp.array([c.weight for c in pi.components])
        weights = weights / weights.sum()
        means = jnp.stack([c.mean for c in pi.components])
        stds = jnp.stack([c.std for c in pi.components])
        return pi.null_prob, weights, means, stds

    # ---- observation computation ----

    def compute_X_batched(U, L):
        T_plus_1 = U.shape[1]
        T = T_plus_1 - 1
        n_obs = T - 2 * L + 1
        rho = U[:, :, 0]
        logsigma = U[:, :, 1]
        xs = jnp.arange(-L, L + 1, dtype=jnp.float32)
        logsigma = jnp.clip(logsigma, -4.0, 4.0)
        var = jnp.exp(2.0 * logsigma)
        shapes = jax.nn.softplus(rho)[:, :, None] * jnp.exp(
            -xs[None, None, :] ** 2 / (2.0 * var[:, :, None])
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

    # ---- flow matching network ----
    import equinox as eqx
    import optax

    class VelocityMLP(eqx.Module):
        layers: list
        u_dim: int = eqx.field(static=True)
        x_dim: int = eqx.field(static=True)

        def __init__(self, T, L, hidden_dims, *, key):
            u_dim = (T + 1) * 2
            x_dim = T - 2 * L + 1
            in_dim = u_dim + x_dim + 1
            self.u_dim = u_dim
            self.x_dim = x_dim
            dims = [in_dim] + hidden_dims + [u_dim]
            keys = jax.random.split(key, len(dims) - 1)
            self.layers = []
            for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
                self.layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))

        def __call__(self, noisy_U_flat, X, s):
            s = jnp.atleast_1d(s)
            inp = jnp.concatenate([noisy_U_flat, X, s])
            x = inp
            for layer in self.layers[:-1]:
                x = jax.nn.gelu(layer(x))
            x = self.layers[-1](x)
            return x

    def flow_matching_loss_single(model, U_flat, X, key):
        k1, k2 = jax.random.split(key)
        s = jax.random.uniform(k1, shape=())
        eps = jax.random.normal(k2, shape=U_flat.shape)
        z_s = (1.0 - s) * eps + s * U_flat
        target = U_flat - eps
        pred = model(z_s, X, s)
        return jnp.mean((pred - target) ** 2)

    @eqx.filter_value_and_grad
    def flow_matching_loss_batch(model, U, X, keys):
        batch = U.shape[0]
        U_flat = U.reshape(batch, -1)
        losses = jax.vmap(
            lambda u, x, k: flow_matching_loss_single(model, u, x, k)
        )(U_flat, X, keys)
        return jnp.mean(losses)

    def sample_flow(model, X, key, n_steps=20):
        z = jax.random.normal(key, shape=(model.u_dim,))
        dt = 1.0 / n_steps

        def euler_step(i, z):
            s = i * dt
            v = model(z, X, jnp.array(s))
            return z + dt * v

        z = jax.lax.fori_loop(0, n_steps, euler_step, z)
        logsigma_mask = jnp.tile(jnp.array([0.0, 1.0]), model.u_dim // 2)
        z = jnp.where(logsigma_mask, jnp.clip(z, -4.0, 4.0), z)
        return z

    @eqx.filter_jit
    def sample_flow_batch(model, X_batch, key, n_steps=20):
        batch = X_batch.shape[0]
        keys = jax.random.split(key, batch)
        T_plus_1 = model.u_dim // 2
        U_flat = jax.vmap(
            lambda x, k: sample_flow(model, x, k, n_steps)
        )(X_batch, keys)
        return U_flat.reshape(batch, T_plus_1, 2)

    @eqx.filter_jit
    def update_step(model, opt_state, optimizer, U, X, key):
        batch = X.shape[0]
        keys = jax.random.split(key, batch)
        loss, grads = flow_matching_loss_batch(model, U, X, keys)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    @eqx.filter_jit
    def eval_flow_loss(model, U, X, key):
        """Evaluate flow matching loss on a batch WITHOUT gradients."""
        batch = U.shape[0]
        U_flat = U.reshape(batch, -1)
        keys = jax.random.split(key, batch)
        losses = jax.vmap(
            lambda u, x, k: flow_matching_loss_single(model, u, x, k)
        )(U_flat, X, keys)
        return jnp.mean(losses)

    # ================================================================
    # ██  CASE CONFIGURATION  ██
    # Change these for each case study.
    # ================================================================
    case_name = "case_softplus_np95_pihat95"

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
    null_prob_init = 0.95      # same as truth for this case
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

    # Initialise network
    key, model_key = jax.random.split(key)
    model = VelocityMLP(T=T, L=L, hidden_dims=hidden_dims, key=model_key)

    # Phase I uses its own cosine schedule over phase-1 steps only
    schedule_p1 = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase1_steps)
    optimizer_p1 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_p1),
    )
    opt_state = optimizer_p1.init(model)

    # Pre-stack mixture parameters for JIT-compiled sampling
    np_hat, w_hat, m_hat, s_hat = stack_mixture_params(pihat)
    np_true, w_true, m_true, s_true = stack_mixture_params(pi_true)

    # ================================================================
    # JIT warmup
    # ================================================================
    log("=== JIT warmup ===")
    t_warmup = time_mod.time()

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    _X1, _U1 = sample_XU_mixture(
        k1, np_hat, w_hat, m_hat, s_hat, L, T, n_init_samples)
    _X2, _ = sample_XU_mixture(
        k2, np_true, w_true, m_true, s_true, L, T, n_source)
    _, _, _loss = update_step(model, opt_state, optimizer_p1, _U1, _X1, k3)
    _Uinf = sample_flow_batch(model, _X2, k4)
    _X3, _ = sample_XU_empirical(
        k5, _Uinf.reshape(-1, 2), L, T, n_eachstep_samples)
    jax.block_until_ready((_X1, _X2, _X3, _loss, _Uinf))

    dt_warmup = time_mod.time() - t_warmup
    log(f"JIT warmup complete in {dt_warmup:.1f}s")

    # ================================================================
    # Fixed evaluation batch (ground-truth data for loss tracking)
    # ================================================================
    key, k_eval_data, k_eval_warm = jax.random.split(key, 3)
    n_eval = 512
    X_eval_gt, U_eval_gt = sample_XU_mixture(
        k_eval_data, np_true, w_true, m_true, s_true, L, T, n_eval)
    eval_key_fixed = jax.random.PRNGKey(999)  # fixed for reproducibility

    # JIT-warm the eval function
    _eval_loss = eval_flow_loss(model, U_eval_gt, X_eval_gt, eval_key_fixed)
    jax.block_until_ready(_eval_loss)
    log(f"Eval function warmed up (initial GT loss = {float(_eval_loss):.4f})")

    # Loss tracking lists
    p1_steps_loss: list[int] = []
    p1_train_losses: list[float] = []
    p1_gt_losses: list[float] = []
    p2_steps_loss: list[int] = []
    p2_gt_losses: list[float] = []
    eval_interval_p1 = 200
    eval_interval_p2 = 2500

    # ================================================================
    # Generate X-data diagnostic plots BEFORE training
    # ================================================================
    log("=== Generating X-data diagnostic plots ===")
    key, k_xdata_true, k_xdata_pihat = jax.random.split(key, 3)
    n_xdata = 8   # number of example observations to plot

    # True-model data
    X_show_true, U_show_true = sample_XU_mixture(
        k_xdata_true, np_true, w_true, m_true, s_true, L, T, n_xdata)
    X_show_true_np = np.array(X_show_true)
    U_show_true_np = np.array(U_show_true)

    # Pihat data (Phase I training data)
    X_show_pihat, U_show_pihat = sample_XU_mixture(
        k_xdata_pihat, np_hat, w_hat, m_hat, s_hat, L, T, n_xdata)
    X_show_pihat_np = np.array(X_show_pihat)
    U_show_pihat_np = np.array(U_show_pihat)

    results: dict[str, bytes] = {}

    def _plot_xdata(X_np, U_np, suptitle, n_show):
        """Plot X observations with rug-plots marking where softplus(rho) is appreciable."""
        n_obs = X_np.shape[1]    # T - 2L + 1
        T_plus_1 = U_np.shape[1]
        obs_grid = np.arange(n_obs)

        n_cols = min(4, n_show)
        n_rows = (n_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 3.5 * n_rows),
                                 squeeze=False)
        for idx in range(n_show):
            ax = axes[idx // n_cols, idx % n_cols]
            # Plot X as a bar/line chart
            ax.bar(obs_grid, X_np[idx], width=0.8, alpha=0.6,
                   color="tab:blue", label="X")

            # Rug-plot: mark time-positions where softplus(rho) > 0.05
            rhos = U_np[idx, :, 0]       # shape (T+1,)
            amps = np.log1p(np.exp(rhos))  # softplus
            for tau in range(T_plus_1):
                if amps[tau] > 0.05:
                    obs_pos = tau - L
                    if 0 <= obs_pos < n_obs:
                        ax.axvline(obs_pos, color="red", alpha=0.6,
                                   lw=1.5, ls="--")
                        ax.text(obs_pos, ax.get_ylim()[1] * 0.95,
                                f"sp={amps[tau]:.2f}",
                                fontsize=6, color="red",
                                ha="center", va="top", rotation=90)

            nonnull_count = int(np.sum(amps > 0.05))
            ax.set_title(f"Sample {idx}  ({nonnull_count} non-null marks)",
                         fontsize=9)
            ax.set_xlabel("obs index")
            ax.set_ylabel("X value")

        # Hide unused axes
        for idx in range(n_show, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        return fig

    fig_xtrue = _plot_xdata(
        X_show_true_np, U_show_true_np,
        f"True-model X data  (null_prob={null_prob_true},"
        f"  L={L}, T={T})",
        n_xdata,
    )
    buf = io.BytesIO()
    fig_xtrue.savefig(buf, format="png", dpi=150)
    plt.close(fig_xtrue)
    results["xdata_true.png"] = buf.getvalue()

    fig_xpihat = _plot_xdata(
        X_show_pihat_np, U_show_pihat_np,
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
    # Phase I — warm-up on pihat
    # ================================================================
    log(f"=== Phase I ({n_phase1_steps} steps, "
        f"pihat null_prob={null_prob_init}) ===")
    log_interval_p1 = max(1, n_phase1_steps // 20)
    t_phase1 = time_mod.time()

    for step in range(n_phase1_steps):
        key, k_data, k_step = jax.random.split(key, 3)
        X, U = sample_XU_mixture(
            k_data, np_hat, w_hat, m_hat, s_hat, L, T, n_init_samples)
        model, opt_state, loss = update_step(
            model, opt_state, optimizer_p1, U, X, k_step)
        if step % log_interval_p1 == 0 or step == n_phase1_steps - 1:
            log(f"  step {step:5d}  loss={float(loss):.6f}")
        if step % eval_interval_p1 == 0 or step == n_phase1_steps - 1:
            gt_loss = float(eval_flow_loss(
                model, U_eval_gt, X_eval_gt, eval_key_fixed))
            p1_steps_loss.append(step)
            p1_train_losses.append(float(loss))
            p1_gt_losses.append(gt_loss)
            if step % log_interval_p1 != 0:  # avoid double-logging
                log(f"  step {step:5d}  loss={float(loss):.6f}"
                    f"  gt_loss={gt_loss:.6f}")

    jax.block_until_ready(loss)
    dt_p1 = time_mod.time() - t_phase1
    log(f"Phase I complete in {dt_p1:.1f}s "
        f"({dt_p1 / n_phase1_steps * 1000:.1f} ms/step)")

    # ================================================================
    # Post-Phase-I diagnostic
    # ================================================================
    log("=== Post-Phase-I diagnostic ===")
    key, k_p1d_x, k_p1d_inf = jax.random.split(key, 3)
    n_p1_diag = 512
    X_p1d, U_p1d_true = sample_XU_mixture(
        k_p1d_x, np_true, w_true, m_true, s_true, L, T, n_p1_diag)
    U_p1d_inferred = jax.lax.stop_gradient(
        sample_flow_batch(model, X_p1d, k_p1d_inf))
    U_p1d_flat = U_p1d_inferred.reshape(-1, 2)
    frac_p1 = float(jnp.mean(jax.nn.softplus(U_p1d_flat[:, 0]) < 0.1))
    mean_rho_p1 = float(jnp.mean(jnp.abs(U_p1d_flat[:, 0])))
    log(f"  After Phase I (trained on pihat null_prob={null_prob_init}):")
    log(f"  frac(softplus(rho)<0.1) = {frac_p1:.3f}  (target={null_prob_true})")
    log(f"  mean|rho|               = {mean_rho_p1:.3f}")

    # ================================================================
    # Phase II — iterative refinement
    # **Fresh optimizer**: completely restart LR schedule and Adam state.
    # ================================================================
    log(f"=== Phase II ({n_phase2_steps} iterations, "
        f"true null_prob={null_prob_true}) ===")
    log("  [NOTE] Optimizer restarted: fresh cosine schedule + Adam moments")

    schedule_p2 = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase2_steps)
    optimizer_p2 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_p2),
    )
    opt_state = optimizer_p2.init(model)

    t_phase2 = time_mod.time()

    def should_log_p2(step):
        if step < 1000:
            return step % 100 == 0
        if step < 10000:
            return step % 1000 == 0
        return step % 7500 == 0 or step == n_phase2_steps - 1

    for step in range(n_phase2_steps):
        key, k_true, k_infer, k_data, k_step = jax.random.split(key, 5)

        # (a) Generate X from the true distribution (discard true U)
        X_true, _ = sample_XU_mixture(
            k_true, np_true, w_true, m_true, s_true, L, T, n_source)

        # (b) Infer U with the current flow model
        U_inferred = jax.lax.stop_gradient(
            sample_flow_batch(model, X_true, k_infer)
        )
        U_inferred = jnp.where(
            jnp.isfinite(U_inferred), U_inferred,
            jnp.array([-10.0, 0.0]))

        # (c) Sample new training data from empirical pihat
        U_flat = U_inferred.reshape(-1, 2)
        X_new, U_new = sample_XU_empirical(
            k_data, U_flat, L, T, n_eachstep_samples)

        # (d) Gradient step
        model, opt_state, loss = update_step(
            model, opt_state, optimizer_p2, U_new, X_new, k_step)

        if should_log_p2(step):
            mean_rho = float(jnp.mean(jnp.abs(U_flat[:, 0])))
            frac_null = float(jnp.mean(jax.nn.softplus(U_flat[:, 0]) < 0.1))
            current_lr = float(schedule_p2(step))
            log(
                f"  iter {step:6d}  loss={float(loss):.6f}  "
                f"mean|rho|={mean_rho:.3f}  "
                f"frac(sp(rho)<0.1)={frac_null:.3f}  "
                f"lr={current_lr:.6f}"
            )

        if step % eval_interval_p2 == 0 or step == n_phase2_steps - 1:
            gt_loss = float(eval_flow_loss(
                model, U_eval_gt, X_eval_gt, eval_key_fixed))
            p2_steps_loss.append(step)
            p2_gt_losses.append(gt_loss)
            if not should_log_p2(step):
                log(f"  iter {step:6d}  gt_loss={gt_loss:.6f}")

    jax.block_until_ready(loss)
    dt_p2 = time_mod.time() - t_phase2
    log(f"Phase II complete in {dt_p2:.1f}s "
        f"({dt_p2 / n_phase2_steps * 1000:.1f} ms/step)")

    # ================================================================
    # Final diagnostic inference
    # ================================================================
    log("=== Final diagnostic inference ===")
    key, k_diag_true, k_diag_infer = jax.random.split(key, 3)
    n_diag = 1024
    X_diag, U_diag_true = sample_XU_mixture(
        k_diag_true, np_true, w_true, m_true, s_true, L, T, n_diag)
    U_diag_inferred = jax.lax.stop_gradient(
        sample_flow_batch(model, X_diag, k_diag_infer)
    )
    U_diag_flat = U_diag_inferred.reshape(-1, 2)
    U_true_flat = U_diag_true.reshape(-1, 2)

    frac_near_zero = float(jnp.mean(jax.nn.softplus(U_diag_flat[:, 0]) < 0.1))
    log(f"  n_diag={n_diag}, total U pairs={U_diag_flat.shape[0]}")
    log(f"  frac(softplus(rho)<0.1) inferred = {frac_near_zero:.3f}")
    log(f"  true null_prob                    = {null_prob_true}")
    log(f"  post-Phase-I frac                 = {frac_p1:.3f}")
    log(f"  total wall time           = {time_mod.time() - _t0:.1f}s")

    # ================================================================
    # Stop GPU monitor
    # ================================================================
    _gpu_active = False
    gpu_thread.join(timeout=5)

    # ================================================================
    # Generate training-result plots
    # ================================================================
    U_inf_np = np.array(U_diag_flat)
    U_true_np = np.array(U_true_flat)
    U_p1d_np = np.array(U_p1d_flat)

    key, k_ref = jax.random.split(key)
    _, ref_U = sample_XU_mixture(
        k_ref, np_true, w_true, m_true, s_true, L=0, T=0, n_samples=10000)
    ref_samples = np.array(ref_U.reshape(-1, 2))

    # ---- Plot: Scatter of aggregate posterior vs ground truth ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.scatter(ref_samples[:, 0], ref_samples[:, 1],
               s=1, alpha=0.3, c="tab:blue", label="true π samples")
    ax.set_xlabel("ρ"); ax.set_ylabel("log σ")
    ax.set_title(f"Ground-truth π\n(null_prob={null_prob_true})")
    ax.axvline(-10, color="red", lw=0.8, ls="--", label="ρ=−10 (null)")
    ax.legend(fontsize=8)
    ax.set_xlim(-12, 4); ax.set_ylim(-3, 3)

    ax = axes[1]
    ax.scatter(U_p1d_np[:, 0], U_p1d_np[:, 1],
               s=1, alpha=0.3, c="tab:green", label="post-Phase-I inferred")
    ax.set_xlabel("ρ"); ax.set_ylabel("log σ")
    frac_p1_str = f"{frac_p1:.1%}"
    ax.set_title(
        f"Post-Phase-I inference\n"
        f"frac(sp(ρ)<0.1)={frac_p1_str}  (pihat null_prob={null_prob_init})")
    ax.axvline(-10, color="red", lw=0.8, ls="--", label="ρ=−10 (null)")
    ax.legend(fontsize=8)
    ax.set_xlim(-12, 4); ax.set_ylim(-3, 3)

    ax = axes[2]
    ax.scatter(U_inf_np[:, 0], U_inf_np[:, 1],
               s=1, alpha=0.3, c="tab:orange", label="final inferred U")
    ax.set_xlabel("ρ"); ax.set_ylabel("log σ")
    frac_str = f"{frac_near_zero:.1%}"
    ax.set_title(
        f"Final (post-Phase-II)\n"
        f"frac(sp(ρ)<0.1)={frac_str}  (true null={null_prob_true})")
    ax.axvline(-10, color="red", lw=0.8, ls="--", label="ρ=−10 (null)")
    ax.legend(fontsize=8)
    ax.set_xlim(-12, 4); ax.set_ylim(-3, 3)

    fig.suptitle(
        f"{case_name}  L={L}, T={T}, seed={seed}\n"
        f"null_prob_true={null_prob_true}, null_prob_init={null_prob_init}, "
        f"phase1={n_phase1_steps}, phase2={n_phase2_steps}  "
        f"[softplus(ρ) shape, Phase II: fresh optimizer]",
        fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["posterior_vs_truth.png"] = buf.getvalue()

    # ---- Plot: Marginal histogram of rho ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    ax.hist(ref_samples[:, 0], bins=80, density=True,
            alpha=0.6, color="tab:blue", label="true π")
    ax.set_xlabel("ρ"); ax.set_ylabel("density")
    ax.set_title("Ground-truth marginal of ρ")
    ax.axvline(-10, color="red", lw=0.8, ls="--"); ax.legend(fontsize=8)

    ax = axes[1]
    rho_p1 = U_p1d_np[:, 0][np.isfinite(U_p1d_np[:, 0])]
    ax.hist(rho_p1, bins=80, density=True,
            alpha=0.6, color="tab:green", label="post-Phase-I")
    ax.set_xlabel("ρ"); ax.set_ylabel("density")
    ax.set_title(f"Post-Phase-I marginal of ρ\nfrac(sp(ρ)<0.1)={frac_p1_str}")
    ax.axvline(-10, color="red", lw=0.8, ls="--"); ax.legend(fontsize=8)

    ax = axes[2]
    rho_finite = U_inf_np[:, 0][np.isfinite(U_inf_np[:, 0])]
    ax.hist(rho_finite, bins=80, density=True,
            alpha=0.6, color="tab:orange", label="final inferred")
    ax.set_xlabel("ρ"); ax.set_ylabel("density")
    ax.set_title(f"Final marginal of ρ\nfrac(sp(ρ)<0.1)={frac_str}")
    ax.axvline(-10, color="red", lw=0.8, ls="--"); ax.legend(fontsize=8)

    fig.suptitle(f"Marginal ρ distributions ({case_name})", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["rho_marginal.png"] = buf.getvalue()

    # ---- Plot: True vs inferred latent U ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(U_true_np[:, 0], U_true_np[:, 1],
               s=1, alpha=0.3, c="tab:green", label="true U (latent)")
    ax.set_xlabel("ρ"); ax.set_ylabel("log σ")
    ax.set_title(f"True latent U ({n_diag} observations)")
    ax.axvline(-10, color="red", lw=0.8, ls="--"); ax.legend(fontsize=8)
    ax.set_xlim(-12, 4); ax.set_ylim(-3, 3)

    ax = axes[1]
    ax.scatter(U_inf_np[:, 0], U_inf_np[:, 1],
               s=1, alpha=0.3, c="tab:orange", label="inferred U")
    ax.set_xlabel("ρ"); ax.set_ylabel("log σ")
    ax.set_title(f"Inferred U ({n_diag} observations)")
    ax.axvline(-10, color="red", lw=0.8, ls="--"); ax.legend(fontsize=8)
    ax.set_xlim(-12, 4); ax.set_ylim(-3, 3)

    fig.suptitle(f"True latent vs inferred latent ({case_name})", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["true_vs_inferred_U.png"] = buf.getvalue()

    # ---- Plot: Survival function of softplus(rho) ----
    sp_true = np.log1p(np.exp(ref_samples[:, 0]))
    sp_p1 = np.log1p(np.exp(rho_p1))
    sp_inf = np.log1p(np.exp(rho_finite))

    sp_true_sorted = np.sort(sp_true)
    surv_true = 1.0 - np.arange(1, len(sp_true_sorted) + 1) / len(sp_true_sorted)
    sp_p1_sorted = np.sort(sp_p1)
    surv_p1 = 1.0 - np.arange(1, len(sp_p1_sorted) + 1) / len(sp_p1_sorted)
    sp_inf_sorted = np.sort(sp_inf)
    surv_inf = 1.0 - np.arange(1, len(sp_inf_sorted) + 1) / len(sp_inf_sorted)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sp_true_sorted, surv_true,
            label="true π", color="tab:blue", lw=1.5)
    ax.plot(sp_p1_sorted, surv_p1,
            label="post-Phase-I", color="tab:green", lw=1.5, ls="--")
    ax.plot(sp_inf_sorted, surv_inf,
            label="final (post-Phase-II)", color="tab:orange", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("softplus(ρ)"); ax.set_ylabel("P(softplus(ρ) > t)")
    ax.set_title(
        f"Empirical survival of softplus(ρ)  ({case_name})\n"
        f"null_prob={null_prob_true}, post-P1={frac_p1_str}, final={frac_str}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=1e-4, top=1.0)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    results["rho_survival.png"] = buf.getvalue()

    # ---- Plot: Loss curves ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Phase I: training loss (pihat data) + GT eval loss
    ax1.plot(p1_steps_loss, p1_train_losses,
             label="Training loss (π̂ data)", color="tab:blue", alpha=0.7)
    ax1.plot(p1_steps_loss, p1_gt_losses,
             label="GT eval loss (true π data)", color="tab:orange", lw=2)
    ax1.set_xlabel("Phase I step")
    ax1.set_ylabel("Flow matching loss")
    ax1.set_title("Phase I: Warm-up on π̂")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Phase II: GT eval loss only
    ax2.plot(p2_steps_loss, p2_gt_losses,
             label="GT eval loss (true π data)", color="tab:orange", lw=2)
    ax2.set_xlabel("Phase II step")
    ax2.set_ylabel("Flow matching loss")
    ax2.set_title("Phase II: Bootstrap refinement (GT eval)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Flow matching loss curves ({case_name})\n"
        f"Eval on {n_eval} fixed ground-truth (X, U) pairs",
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
