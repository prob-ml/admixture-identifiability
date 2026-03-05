#!/usr/bin/env python3
"""Oracle Phase-I test: train with the *true* distribution pi.

This mode verifies that the flow-matching architecture can learn the
correct aggregate posterior when given data from the true shape
distribution.  No Phase II bootstrap is run — the model sees ground-truth
(X, U) pairs throughout training.

Usage:
    modal run gaubump_scripts/run_modal_oracle.py
"""

from __future__ import annotations

import modal

app = modal.App("gaubump-oracle-phaseI")

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
def run_oracle_training():
    """Train with true pi and generate diagnostic plots."""
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
        # Clamp logsigma to prevent exp overflow/underflow (NaN when var→0).
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
        # Clamp the logsigma components (odd indices) to prevent downstream NaN.
        # The output z is flattened as [rho0, ls0, rho1, ls1, ...].
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
    # Training configuration — oracle Phase I
    # ================================================================
    L = 3
    T = 20
    null_prob_true = 0.95      # 95% null mass — easy case
    n_phase1_steps = 150_000
    n_init_samples = 256
    n_source = 256             # batch size for diagnostic inference
    lr = 1e-3
    hidden_dims = [256, 256, 256]
    seed = 42

    key = jax.random.PRNGKey(seed)

    # Same GMM components, but null_prob boosted to 0.95
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
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_phase1_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    )
    opt_state = optimizer.init(model)

    np_true, w_true, m_true, s_true = stack_mixture_params(pi_true)

    # ================================================================
    # JIT warmup
    # ================================================================
    log("=== JIT warmup ===")
    t_warmup = time_mod.time()

    key, k1, k2, k3 = jax.random.split(key, 4)
    _X1, _U1 = sample_XU_mixture(
        k1, np_true, w_true, m_true, s_true, L, T, n_init_samples)
    _, _, _loss = update_step(model, opt_state, optimizer, _U1, _X1, k2)
    _Uinf = sample_flow_batch(model, _X1, k3)
    jax.block_until_ready((_X1, _U1, _loss, _Uinf))

    dt_warmup = time_mod.time() - t_warmup
    log(f"JIT warmup complete in {dt_warmup:.1f}s")

    # ================================================================
    # Oracle Phase I — train with true pi
    # ================================================================
    log(f"=== Oracle Phase I ({n_phase1_steps} steps, "
        f"null_prob={null_prob_true}) ===")
    log_interval = max(1, n_phase1_steps // 20)
    t_phase1 = time_mod.time()

    for step in range(n_phase1_steps):
        key, k_data, k_step = jax.random.split(key, 3)
        # Train with ground-truth (X, U) from pi_true
        X, U = sample_XU_mixture(
            k_data, np_true, w_true, m_true, s_true, L, T, n_init_samples)
        model, opt_state, loss = update_step(
            model, opt_state, optimizer, U, X, k_step)
        if step % log_interval == 0 or step == n_phase1_steps - 1:
            # Periodic diagnostic: infer U from a batch of true X
            key, k_diag_x, k_diag_inf = jax.random.split(key, 3)
            X_diag_batch, _ = sample_XU_mixture(
                k_diag_x, np_true, w_true, m_true, s_true,
                L, T, n_source)
            U_diag_batch = jax.lax.stop_gradient(
                sample_flow_batch(model, X_diag_batch, k_diag_inf))
            U_diag_flat = U_diag_batch.reshape(-1, 2)
            frac_small = float(
                jnp.mean(jnp.abs(U_diag_flat[:, 0]) < 0.1))
            log(
                f"  step {step:6d}  loss={float(loss):.6f}  "
                f"frac(|rho|<0.1)={frac_small:.3f}"
            )

    jax.block_until_ready(loss)
    dt_p1 = time_mod.time() - t_phase1
    log(f"Oracle Phase I complete in {dt_p1:.1f}s "
        f"({dt_p1 / n_phase1_steps * 1000:.1f} ms/step)")

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

    key, k_ref = jax.random.split(key)
    _, ref_U = sample_XU_mixture(
        k_ref, np_true, w_true, m_true, s_true, L=0, T=0, n_samples=10000)
    ref_samples = np.array(ref_U.reshape(-1, 2))

    results: dict[str, bytes] = {}

    # ---- Plot 1: Scatter of aggregate posterior vs ground truth ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
    ax.scatter(U_inf_np[:, 0], U_inf_np[:, 1],
               s=1, alpha=0.3, c="tab:orange", label="inferred U")
    ax.set_xlabel("ρ")
    ax.set_ylabel("log σ")
    frac_str = f"{frac_near_zero:.1%}"
    ax.set_title(
        f"Aggregate posterior (inferred U)\n"
        f"frac(|ρ|<0.1)={frac_str}  (true null={null_prob_true})"
    )
    ax.axvline(0, color="red", lw=0.8, ls="--", label="ρ=0 (null)")
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-3, 3)

    fig.suptitle(
        f"Oracle Phase I — L={L}, T={T}, seed={seed}, "
        f"steps={n_phase1_steps}, null_prob={null_prob_true}",
        fontsize=11,
    )
    fig.tight_layout()
    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", dpi=150)
    plt.close(fig)
    results["oracle_posterior_vs_truth.png"] = buf1.getvalue()

    # ---- Plot 2: Empirical survival function of |rho| ----
    rho_true = np.abs(ref_samples[:, 0])
    rho_inf = np.abs(U_inf_np[:, 0])
    rho_inf = rho_inf[np.isfinite(rho_inf)]

    rho_true_sorted = np.sort(rho_true)
    surv_true = 1.0 - np.arange(1, len(rho_true_sorted) + 1) / len(rho_true_sorted)

    rho_inf_sorted = np.sort(rho_inf)
    surv_inf = 1.0 - np.arange(1, len(rho_inf_sorted) + 1) / len(rho_inf_sorted)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rho_true_sorted, surv_true,
            label="true π", color="tab:blue", lw=1.5)
    ax.plot(rho_inf_sorted, surv_inf,
            label="aggregate posterior", color="tab:orange", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("|ρ|")
    ax.set_ylabel("P(|ρ| > t)")
    ax.set_title(
        f"Empirical survival function of |ρ|\n"
        f"null_prob={null_prob_true}, "
        f"frac(|ρ|<0.1)={frac_str}"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=1e-4, top=1.0)
    fig.tight_layout()
    buf2 = io.BytesIO()
    fig.savefig(buf2, format="png", dpi=150)
    plt.close(fig)
    results["oracle_rho_survival.png"] = buf2.getvalue()

    # ---- Training log ----
    training_log = "\n".join(log_lines) + "\n"
    results["oracle_training_log.txt"] = training_log.encode("utf-8")

    # ---- GPU utilization log ----
    gpu_header = ("timestamp, utilization.gpu [%], utilization.memory [%], "
                  "memory.used [MiB], memory.total [MiB], temperature.gpu")
    gpu_utilization_log = gpu_header + "\n" + "\n".join(gpu_log_lines) + "\n"
    results["oracle_gpu_utilization.txt"] = gpu_utilization_log.encode("utf-8")

    log(f"Generated {len(results)} result files.")
    return results


@app.local_entrypoint()
def main():
    import os
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    result_files = run_oracle_training.remote()

    for name, data in result_files.items():
        path = os.path.join(results_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved {path} ({len(data)} bytes)")

    print("All done!")
