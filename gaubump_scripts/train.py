#!/usr/bin/env python3
"""End-to-end training script for the Gaussian-bump flow matching model.

Uses the binary-V two-model architecture:
  1. V-classifier: predicts null/active gate V from X.
  2. U-flow: flow matching conditioned on (X, V), masked to active positions.

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
    sample_XVU_mixture,
    sample_XVU_empirical,
)
from flow_matching import (
    VClassifierMLP,
    VelocityMLP,
    make_optimizer,
    update_step,
    update_step_v,
    sample_flow_batch,
    eval_v_accuracy,
)


# ---------------------------------------------------------------------------
# Default ground-truth distribution
# ---------------------------------------------------------------------------

def make_default_pi_true(null_prob: float = 0.6) -> ShapeDistribution:
    """A mixture: null with weight *null_prob*, plus two Gaussians."""
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

    # Initialise networks and optimisers
    key, model_key, vclass_key = jax.random.split(key, 3)
    model = VelocityMLP(T=T, L=L, hidden_dims=hidden_dims, key=model_key)
    v_model = VClassifierMLP(T=T, L=L, hidden_dims=hidden_dims, key=vclass_key)
    optimizer = make_optimizer(lr)
    opt_state = optimizer.init(model)
    v_optimizer = make_optimizer(lr)
    v_opt_state = v_optimizer.init(v_model)

    # ------------------------------------------------------------------
    # JIT warmup — compile all kernels before the training loops
    # ------------------------------------------------------------------
    print("=== JIT warmup ===")
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    _X1, _V1, _U1 = sample_XVU_mixture(
        k1, np_hat, w_hat, m_hat, s_hat, L, T, n_init_samples)
    _X2, _V2, _ = sample_XVU_mixture(
        k2, np_true, w_true, m_true, s_true, L, T, n_source)
    _, _, _loss = update_step(model, opt_state, optimizer, _U1, _X1, _V1, k3)
    v_model, v_opt_state, _vloss = update_step_v(
        v_model, v_opt_state, v_optimizer, _X1, _V1)
    # Predict V, then sample U conditioned on it
    _V_logits = jax.vmap(v_model)(_X2)
    _V_pred = (_V_logits > 0.0).astype(jnp.float32)
    _Uinf = sample_flow_batch(model, _X2, _V_pred, k4)
    _X3, _V3, _U3 = sample_XVU_empirical(
        k5, _V_pred.reshape(-1), _Uinf.reshape(-1, 2), L, T, n_eachstep_samples)
    jax.block_until_ready((_X1, _X2, _X3, _loss, _Uinf))
    print("  warmup complete")

    # ------------------------------------------------------------------
    # Phase I — warm-up
    # ------------------------------------------------------------------
    print(f"=== Phase I ({n_phase1_steps} steps) ===")
    for step in range(n_phase1_steps):
        key, k_data, k_step = jax.random.split(key, 3)
        X, V, U = sample_XVU_mixture(
            k_data, np_hat, w_hat, m_hat, s_hat, L, T, n_init_samples)
        model, opt_state, loss = update_step(
            model, opt_state, optimizer, U, X, V, k_step)
        v_model, v_opt_state, v_loss = update_step_v(
            v_model, v_opt_state, v_optimizer, X, V)
        if step % max(1, n_phase1_steps // 20) == 0 or step == n_phase1_steps - 1:
            print(f"  step {step:5d}  u_loss={float(loss):.6f}"
                  f"  v_loss={float(v_loss):.6f}")

    # ------------------------------------------------------------------
    # Phase II — iterative refinement
    # ------------------------------------------------------------------
    print(f"\n=== Phase II ({n_phase2_steps} iterations) ===")
    for step in range(n_phase2_steps):
        key, k_true, k_infer, k_data, k_step = jax.random.split(key, 5)

        # 1. Draw X from the *true* model (JIT-compiled)
        X_true, V_true_unused, _ = sample_XVU_mixture(
            k_true, np_true, w_true, m_true, s_true, L, T, n_source)

        # 2. Predict V with classifier
        V_logits = jax.vmap(v_model)(X_true)
        V_pred = (V_logits > 0.0).astype(jnp.float32)

        # 3. Infer U with the current flow model (stop-gradient)
        U_inferred = jax.lax.stop_gradient(
            sample_flow_batch(model, X_true, V_pred, k_infer)
        )  # (n_source, T+1, 2)

        # Replace any remaining NaN/Inf with zero
        U_inferred = jnp.where(
            jnp.isfinite(U_inferred), U_inferred, 0.0)

        # 4. Train one step on fresh samples from empirical distribution
        V_flat = V_pred.reshape(-1)
        U_flat = U_inferred.reshape(-1, 2)
        X_new, V_new, U_new = sample_XVU_empirical(
            k_data, V_flat, U_flat, L, T, n_eachstep_samples)
        k_u, k_v = jax.random.split(k_step)
        model, opt_state, loss = update_step(
            model, opt_state, optimizer, U_new, X_new, V_new, k_u)
        v_model, v_opt_state, v_loss = update_step_v(
            v_model, v_opt_state, v_optimizer, X_new, V_new)

        if step % max(1, n_phase2_steps // 20) == 0 or step == n_phase2_steps - 1:
            frac_null = float(jnp.mean(V_flat < 0.5))
            print(
                f"  iter {step:5d}  u_loss={float(loss):.6f}"
                f"  v_loss={float(v_loss):.6f}"
                f"  frac(V=0)={frac_null:.3f}"
            )

    # ------------------------------------------------------------------
    # Final diagnostic inference
    # ------------------------------------------------------------------
    print("\n=== Final diagnostic inference ===")
    key, k_diag_true, k_diag_infer = jax.random.split(key, 3)
    n_diag = max(n_source, 256)
    X_diag, V_diag_true, U_diag_true = sample_XVU_mixture(
        k_diag_true, np_true, w_true, m_true, s_true, L, T, n_diag)
    V_diag_logits = jax.vmap(v_model)(X_diag)
    V_diag_pred = (V_diag_logits > 0.0).astype(jnp.float32)
    U_diag_inferred = jax.lax.stop_gradient(
        sample_flow_batch(model, X_diag, V_diag_pred, k_diag_infer)
    )
    V_diag_flat = V_diag_pred.reshape(-1)
    frac_null = float(jnp.mean(V_diag_flat < 0.5))
    v_acc = float(eval_v_accuracy(v_model, X_diag, V_diag_true))
    print(f"  n_diag={n_diag}, total marks={V_diag_flat.shape[0]}")
    print(f"  frac(V=0) inferred = {frac_null:.3f}  (true null_prob={null_prob_true})")
    print(f"  V-classifier accuracy = {v_acc:.3f}")

    print("\nDone.")
    return {
        "model": model,
        "v_model": v_model,
        "pi_true": pi_true,
        "U_diag_inferred": U_diag_inferred,
        "V_diag_pred": V_diag_pred,
        "U_diag_true": U_diag_true,
        "V_diag_true": V_diag_true,
        "X_diag": X_diag,
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
