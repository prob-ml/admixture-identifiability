#!/usr/bin/env python3
"""Diagnostic: oracle warm-start + Phase II bootstrap at null_prob=0.8.

Tests whether the full pipeline fails at null_prob=0.8 because of a bad
Phase I warm-start, or because Phase II's bootstrap loop is intrinsically
unstable at that null probability.

- Phase I: trains on the TRUE pi (like oracle), not on pihat
- Phase II: standard bootstrap refinement (identical to run_modal.py)

If this works → the issue is Phase I warm-start quality.
If this also fails → Phase II bootstrap is unstable at null_prob=0.8.

Usage:
    modal run gaubump_scripts/run_modal_diag.py
"""

from __future__ import annotations

import modal

app = modal.App("gaubump-diag-warmstart")

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
    """Oracle warm-start + Phase II at null_prob=0.8."""
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

    class EmpiricalDistribution(NamedTuple):
        samples: jnp.ndarray

    # ---- sampling (JIT-compiled) ----

    def _sample_XU_mixture_impl(key, null_prob, weights, means, stds, L, T, n_samples):
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
        null = jnp.stack([jnp.zeros(total), null_ls], axis=-1)
        marks = jnp.where(is_null[:, None], null, non_null)
        U = marks.reshape(n_samples, T + 1, 2)
        X = compute_X_batched(U, L)
        return X, U

    sample_XU_mixture = jax.jit(
        _sample_XU_mixture_impl, static_argnums=(5, 6, 7))

    def _sample_XU_empirical_impl(key, samples, L, T, n_samples):
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
        shapes = rho[:, :, None] * jnp.exp(
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

    # ================================================================
    # Training configuration
    # ================================================================
    L = 3
    T = 20
    null_prob_true = 0.8       # same as run_modal.py; tests Phase II stability
    n_phase1_steps = 10_000    # Phase I on TRUE pi (oracle warm-start)
    n_phase2_steps = 150_000   # Phase II bootstrap (same as run_modal.py)
    n_init_samples = 256
    n_source = 256
    n_eachstep_samples = 256
    lr = 1e-3
    hidden_dims = [256, 256, 256]
    seed = 42

    key = jax.random.PRNGKey(seed)

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

    # Initialise network
    key, model_key = jax.random.split(key)
    model = VelocityMLP(T=T, L=L, hidden_dims=hidden_dims, key=model_key)
    total_steps = n_phase1_steps + n_phase2_steps
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=total_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    )
    opt_state = optimizer.init(model)

    # Pre-stack mixture parameters for JIT-compiled sampling
    np_true, w_true, m_true, s_true = stack_mixture_params(pi_true)

    # ================================================================
    # JIT warmup
    # ================================================================
    log("=== JIT warmup ===")
    t_warmup = time_mod.time()

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    _X1, _U1 = sample_XU_mixture(
        k1, np_true, w_true, m_true, s_true, L, T, n_init_samples)
    _, _, _loss = update_step(model, opt_state, optimizer, _U1, _X1, k2)
    _Uinf = sample_flow_batch(model, _X1, k3)
    _X3, _ = sample_XU_empirical(
        k4, _Uinf.reshape(-1, 2), L, T, n_eachstep_samples)
    jax.block_until_ready((_X1, _U1, _X3, _loss, _Uinf))

    dt_warmup = time_mod.time() - t_warmup
    log(f"JIT warmup complete in {dt_warmup:.1f}s")

    # ================================================================
    # Phase I — ORACLE warm-start (train on TRUE pi, not pihat)
    # ================================================================
    log(f"=== Phase I — ORACLE warm-start ({n_phase1_steps} steps, "
        f"TRUE pi null_prob={null_prob_true}) ===")
    log_interval_p1 = max(1, n_phase1_steps // 20)
    t_phase1 = time_mod.time()

    for step in range(n_phase1_steps):
        key, k_data, k_step = jax.random.split(key, 3)
        # KEY DIFFERENCE: train on TRUE pi, not pihat
        X, U = sample_XU_mixture(
            k_data, np_true, w_true, m_true, s_true, L, T, n_init_samples)
        model, opt_state, loss = update_step(
            model, opt_state, optimizer, U, X, k_step)
        if step % log_interval_p1 == 0 or step == n_phase1_steps - 1:
            log(f"  step {step:5d}  loss={float(loss):.6f}")

    jax.block_until_ready(loss)
    dt_p1 = time_mod.time() - t_phase1
    log(f"Phase I (oracle) complete in {dt_p1:.1f}s "
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
    frac_p1 = float(jnp.mean(jnp.abs(U_p1d_flat[:, 0]) < 0.1))
    mean_rho_p1 = float(jnp.mean(jnp.abs(U_p1d_flat[:, 0])))
    log(f"  After oracle Phase I (true null_prob={null_prob_true}):")
    log(f"  frac(|rho|<0.1) = {frac_p1:.3f}  (target={null_prob_true})")
    log(f"  mean|rho|        = {mean_rho_p1:.3f}")
    log(f"  Current LR       = {float(schedule(n_phase1_steps)):.6f}")

    # ================================================================
    # Phase II — standard bootstrap (identical to run_modal.py)
    # ================================================================
    log(f"=== Phase II ({n_phase2_steps} iterations, "
        f"true null_prob={null_prob_true}) ===")
    t_phase2 = time_mod.time()

    def should_log_p2(step):
        if step < 1000:
            return step % 100 == 0
        if step < 10000:
            return step % 1000 == 0
        return step % 7500 == 0 or step == n_phase2_steps - 1

    for step in range(n_phase2_steps):
        key, k_true, k_infer, k_data, k_step = jax.random.split(key, 5)

        X_true, _ = sample_XU_mixture(
            k_true, np_true, w_true, m_true, s_true, L, T, n_source)

        U_inferred = jax.lax.stop_gradient(
            sample_flow_batch(model, X_true, k_infer)
        )

        U_inferred = jnp.where(
            jnp.isfinite(U_inferred), U_inferred, 0.0)

        U_flat = U_inferred.reshape(-1, 2)
        X_new, U_new = sample_XU_empirical(
            k_data, U_flat, L, T, n_eachstep_samples)

        model, opt_state, loss = update_step(
            model, opt_state, optimizer, U_new, X_new, k_step)

        if should_log_p2(step):
            mean_rho = float(jnp.mean(jnp.abs(U_flat[:, 0])))
            frac_small = float(jnp.mean(jnp.abs(U_flat[:, 0]) < 0.1))
            current_lr = float(schedule(n_phase1_steps + step))
            log(
                f"  iter {step:6d}  loss={float(loss):.6f}  "
                f"mean|rho|={mean_rho:.3f}  "
                f"frac(|rho|<0.1)={frac_small:.3f}  "
                f"lr={current_lr:.6f}"
            )

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

    frac_near_zero = float(jnp.mean(jnp.abs(U_diag_flat[:, 0]) < 0.1))
    log(f"  n_diag={n_diag}, total U pairs={U_diag_flat.shape[0]}")
    log(f"  frac(|rho|<0.1) inferred = {frac_near_zero:.3f}")
    log(f"  true null_prob            = {null_prob_true}")
    log(f"  post-Phase-I frac         = {frac_p1:.3f}")
    log(f"  total wall time           = {time_mod.time() - _t0:.1f}s")

    # ================================================================
    # Stop GPU monitor
    # ================================================================
    _gpu_active = False
    gpu_thread.join(timeout=5)

    # ================================================================
    # Generate plots
    # ================================================================
    U_inf_np = np.array(U_diag_flat)
    U_true_np = np.array(U_true_flat)
    U_p1d_np = np.array(U_p1d_flat)

    key, k_ref = jax.random.split(key)
    _, ref_U = sample_XU_mixture(
        k_ref, np_true, w_true, m_true, s_true, L=0, T=0, n_samples=10000)
    ref_samples = np.array(ref_U.reshape(-1, 2))

    results: dict[str, bytes] = {}

    # ---- Plot 1: 3-panel scatter ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.scatter(ref_samples[:, 0], ref_samples[:, 1],
               s=1, alpha=0.3, c="tab:blue", label="true π samples")
    ax.set_xlabel("ρ")
    ax.set_ylabel("log σ")
    ax.set_title(f"Ground-truth π\n(null_prob={null_prob_true})")
    ax.axvline(0, color="red", lw=0.8, ls="--", label="ρ=0 (null)")
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-3, 3)

    ax = axes[1]
    ax.scatter(U_p1d_np[:, 0], U_p1d_np[:, 1],
               s=1, alpha=0.3, c="tab:green",
               label="post-Phase-I (oracle)")
    ax.set_xlabel("ρ")
    ax.set_ylabel("log σ")
    frac_p1_str = f"{frac_p1:.1%}"
    ax.set_title(
        f"Post-Phase-I (oracle warm-start)\n"
        f"frac(|ρ|<0.1)={frac_p1_str}"
    )
    ax.axvline(0, color="red", lw=0.8, ls="--", label="ρ=0 (null)")
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-3, 3)

    ax = axes[2]
    ax.scatter(U_inf_np[:, 0], U_inf_np[:, 1],
               s=1, alpha=0.3, c="tab:orange", label="final inferred U")
    ax.set_xlabel("ρ")
    ax.set_ylabel("log σ")
    frac_str = f"{frac_near_zero:.1%}"
    ax.set_title(
        f"Final (post-Phase-II)\n"
        f"frac(|ρ|<0.1)={frac_str}  (true null={null_prob_true})"
    )
    ax.axvline(0, color="red", lw=0.8, ls="--", label="ρ=0 (null)")
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-3, 3)

    fig.suptitle(
        f"Diagnostic: oracle warm-start + Phase II bootstrap\n"
        f"L={L}, T={T}, seed={seed}, null_prob={null_prob_true}, "
        f"phase1={n_phase1_steps} (oracle), phase2={n_phase2_steps}",
        fontsize=11,
    )
    fig.tight_layout()
    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", dpi=150)
    plt.close(fig)
    results["diag_posterior_vs_truth.png"] = buf1.getvalue()

    # ---- Plot 2: Survival function ----
    rho_true_abs = np.abs(ref_samples[:, 0])
    rho_p1_abs = np.abs(U_p1d_np[:, 0][np.isfinite(U_p1d_np[:, 0])])
    rho_inf_abs = np.abs(U_inf_np[:, 0][np.isfinite(U_inf_np[:, 0])])

    rho_true_sorted = np.sort(rho_true_abs)
    surv_true = 1.0 - np.arange(1, len(rho_true_sorted) + 1) / len(rho_true_sorted)

    rho_p1_sorted = np.sort(rho_p1_abs)
    surv_p1 = 1.0 - np.arange(1, len(rho_p1_sorted) + 1) / len(rho_p1_sorted)

    rho_inf_sorted = np.sort(rho_inf_abs)
    surv_inf = 1.0 - np.arange(1, len(rho_inf_sorted) + 1) / len(rho_inf_sorted)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rho_true_sorted, surv_true,
            label="true π", color="tab:blue", lw=1.5)
    ax.plot(rho_p1_sorted, surv_p1,
            label="post-Phase-I (oracle)", color="tab:green", lw=1.5, ls="--")
    ax.plot(rho_inf_sorted, surv_inf,
            label="final (post-Phase-II)", color="tab:orange", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("|ρ|")
    ax.set_ylabel("P(|ρ| > t)")
    ax.set_title(
        f"Oracle warm-start + Phase II bootstrap (null_prob={null_prob_true})\n"
        f"post-P1 frac={frac_p1_str}, final frac={frac_str}"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=1e-4, top=1.0)
    fig.tight_layout()
    buf2 = io.BytesIO()
    fig.savefig(buf2, format="png", dpi=150)
    plt.close(fig)
    results["diag_rho_survival.png"] = buf2.getvalue()

    # ---- Training log ----
    training_log = "\n".join(log_lines) + "\n"
    results["diag_training_log.txt"] = training_log.encode("utf-8")

    # ---- GPU utilization log ----
    gpu_header = ("timestamp, utilization.gpu [%], utilization.memory [%], "
                  "memory.used [MiB], memory.total [MiB], temperature.gpu")
    gpu_utilization_log = gpu_header + "\n" + "\n".join(gpu_log_lines) + "\n"
    results["diag_gpu_utilization.txt"] = gpu_utilization_log.encode("utf-8")

    log(f"Generated {len(results)} result files.")
    return results


@app.local_entrypoint()
def main():
    import os
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    result_files = run_training.remote()

    for name, data in result_files.items():
        path = os.path.join(results_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved {path} ({len(data)} bytes)")

    print("All done!")
