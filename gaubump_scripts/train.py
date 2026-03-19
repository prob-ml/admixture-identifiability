#!/usr/bin/env python3
"""End-to-end training script for the Gaussian-bump flow matching model.

Usage:
    python train.py [--L 3] [--T 20] [--n_phase1 2000] [--n_phase2 200] ...

Phase I  — warm-up on samples from an initial guess pihat.
Phase II — iterative refinement using true X samples.
"""

from __future__ import annotations

import argparse
import jax
import jax.numpy as jnp

from model import (
    ShapeDistribution,
    GaussianComponent,
    EmpiricalDistribution,
    sample_XU,
    sample_pi,
    stack_mixture_params,
    sample_XU_mixture,
    sample_XU_empirical,
)
from flow_matching import (
    VelocityMLP,
    make_optimizer,
    update_step,
    sample_flow_batch,
)


# ---------------------------------------------------------------------------
# Default ground-truth distribution
# ---------------------------------------------------------------------------

def make_default_pi_true(null_prob: float = 0.6) -> ShapeDistribution:
    """A mixture: atom at (0, ·) with weight *null_prob*, plus two Gaussians."""
    return ShapeDistribution(
        null_prob=null_prob,
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


def make_default_pihat_init(null_prob: float = 0.5) -> ShapeDistribution:
    """An initial guess — deliberately different from the truth."""
    return ShapeDistribution(
        null_prob=null_prob,
        components=[
            GaussianComponent(
                weight=1.0,
                mean=jnp.array([1.0, 0.0]),
                std=jnp.array([1.0, 1.0]),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    L: int = 3,
    T: int = 20,
    n_phase1_steps: int = 2000,
    n_phase2_steps: int = 200,
    n_init_samples: int = 256,
    n_source: int = 128,
    n_eachstep_samples: int = 256,
    lr: float = 1e-3,
    hidden_dims: list[int] | None = None,
    seed: int = 0,
    null_prob_true: float = 0.6,
    null_prob_init: float = 0.5,
):
    if hidden_dims is None:
        hidden_dims = [256, 256, 256]

    key = jax.random.PRNGKey(seed)

    pi_true = make_default_pi_true(null_prob=null_prob_true)
    pihat = make_default_pihat_init(null_prob=null_prob_init)

    # Pre-stack mixture params for JIT-compiled sampling
    np_hat, w_hat, m_hat, s_hat = stack_mixture_params(pihat)
    np_true, w_true, m_true, s_true = stack_mixture_params(pi_true)

    # Initialise network and optimiser
    key, model_key = jax.random.split(key)
    model = VelocityMLP(T=T, L=L, hidden_dims=hidden_dims, key=model_key)
    optimizer = make_optimizer(lr)
    opt_state = optimizer.init(model)

    # ------------------------------------------------------------------
    # JIT warmup — compile all kernels before the training loops
    # ------------------------------------------------------------------
    print("=== JIT warmup ===")
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    _X1, _U1 = sample_XU_mixture(k1, np_hat, w_hat, m_hat, s_hat, L, T, n_init_samples)
    _X2, _ = sample_XU_mixture(k2, np_true, w_true, m_true, s_true, L, T, n_source)
    _, _, _loss = update_step(model, opt_state, optimizer, _U1, _X1, k3)
    _Uinf = sample_flow_batch(model, _X2, k4)
    _X3, _ = sample_XU_empirical(k5, _Uinf.reshape(-1, 2), L, T, n_eachstep_samples)
    jax.block_until_ready((_X1, _X2, _X3, _loss, _Uinf))
    print("  warmup complete")

    # ------------------------------------------------------------------
    # Phase I — warm-up
    # ------------------------------------------------------------------
    print(f"=== Phase I ({n_phase1_steps} steps) ===")
    for step in range(n_phase1_steps):
        key, k_data, k_step = jax.random.split(key, 3)
        X, U = sample_XU_mixture(k_data, np_hat, w_hat, m_hat, s_hat,
                                 L, T, n_init_samples)
        model, opt_state, loss = update_step(model, opt_state, optimizer, U, X, k_step)
        if step % max(1, n_phase1_steps // 20) == 0 or step == n_phase1_steps - 1:
            print(f"  step {step:5d}  loss={float(loss):.6f}")

    # ------------------------------------------------------------------
    # Phase II — iterative refinement
    # ------------------------------------------------------------------
    print(f"\n=== Phase II ({n_phase2_steps} iterations) ===")
    for step in range(n_phase2_steps):
        key, k_true, k_infer, k_data, k_step = jax.random.split(key, 5)

        # 1. Draw X from the *true* model (JIT-compiled)
        X_true, _ = sample_XU_mixture(k_true, np_true, w_true, m_true, s_true,
                                      L, T, n_source)

        # 2. Infer U with the current flow model (stop-gradient)
        U_inferred = jax.lax.stop_gradient(
            sample_flow_batch(model, X_true, k_infer)
        )  # (n_source, T+1, 2)

        # Replace any remaining NaN/Inf with null marks (rho=-10, logsigma=0)
        null_mark = jnp.array([-10.0, 0.0])
        U_inferred = jnp.where(
            jnp.isfinite(U_inferred), U_inferred, null_mark)

        # 3. Train one step on fresh samples from empirical pihat (JIT-compiled)
        U_flat = U_inferred.reshape(-1, 2)  # (n_source*(T+1), 2)
        X_new, U_new = sample_XU_empirical(k_data, U_flat, L, T, n_eachstep_samples)
        model, opt_state, loss = update_step(model, opt_state, optimizer, U_new, X_new, k_step)

        if step % max(1, n_phase2_steps // 20) == 0 or step == n_phase2_steps - 1:
            # Quick diagnostic: mean |rho| in inferred U
            mean_rho = float(jnp.mean(jnp.abs(U_flat[:, 0])))
            frac_null = float(jnp.mean(jax.nn.softplus(U_flat[:, 0]) < 0.1))
            print(
                f"  iter {step:5d}  loss={float(loss):.6f}  "
                f"mean|rho|={mean_rho:.3f}  frac(softplus(rho)<0.1)={frac_null:.3f}"
            )

    # ------------------------------------------------------------------
    # Final diagnostic inference
    # ------------------------------------------------------------------
    print("\n=== Final diagnostic inference ===")
    key, k_diag_true, k_diag_infer = jax.random.split(key, 3)
    n_diag = max(n_source, 256)
    X_diag, U_diag_true = sample_XU_mixture(
        k_diag_true, np_true, w_true, m_true, s_true, L, T, n_diag)
    U_diag_inferred = jax.lax.stop_gradient(
        sample_flow_batch(model, X_diag, k_diag_infer)
    )
    U_diag_flat = U_diag_inferred.reshape(-1, 2)
    frac_near_zero = float(jnp.mean(jax.nn.softplus(U_diag_flat[:, 0]) < 0.1))
    print(f"  n_diag={n_diag}, total U pairs={U_diag_flat.shape[0]}")
    print(f"  frac(softplus(rho)<0.1)={frac_near_zero:.3f}  (true null_prob={null_prob_true})")

    print("\nDone.")
    return {
        "model": model,
        "pi_true": pi_true,
        "U_diag_inferred": U_diag_inferred,  # (n_diag, T+1, 2)
        "U_diag_true": U_diag_true,          # (n_diag, T+1, 2)
        "X_diag": X_diag,                    # (n_diag, T-2L+1)
        "null_prob_true": null_prob_true,
        "L": L,
        "T": T,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Gaussian-bump flow matching model")
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--n_phase1", type=int, default=2000)
    parser.add_argument("--n_phase2", type=int, default=200)
    parser.add_argument("--n_init_samples", type=int, default=256)
    parser.add_argument("--n_source", type=int, default=128)
    parser.add_argument("--n_eachstep_samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null_prob_true", type=float, default=0.6)
    parser.add_argument("--null_prob_init", type=float, default=0.5)
    args = parser.parse_args()

    result = train(
        L=args.L,
        T=args.T,
        n_phase1_steps=args.n_phase1,
        n_phase2_steps=args.n_phase2,
        n_init_samples=args.n_init_samples,
        n_source=args.n_source,
        n_eachstep_samples=args.n_eachstep_samples,
        lr=args.lr,
        seed=args.seed,
        null_prob_true=args.null_prob_true,
        null_prob_init=args.null_prob_init,
    )
    return result


if __name__ == "__main__":
    main()
