"""Rectified flow matching network and utilities.

The velocity network is an MLP (Equinox module) that maps
    (noisy_U, X, s) -> velocity
where
    noisy_U in R^{(T+1) x 2}   — interpolated latent at flow time s
    X       in R^{T-2L+1}      — conditioning observation
    s       in R               — flow time in [0, 1]
    velocity in R^{(T+1) x 2}  — predicted velocity field
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial


# ---------------------------------------------------------------------------
# Velocity network
# ---------------------------------------------------------------------------

class VelocityMLP(eqx.Module):
    """MLP velocity network for rectified flow matching."""
    layers: list
    u_dim: int = eqx.field(static=True)  # (T+1)*2
    x_dim: int = eqx.field(static=True)  # T-2L+1

    def __init__(self, T: int, L: int, hidden_dims: list[int], *, key: jax.Array):
        u_dim = (T + 1) * 2
        x_dim = T - 2 * L + 1
        in_dim = u_dim + x_dim + 1  # noisy_U flattened + X + s

        self.u_dim = u_dim
        self.x_dim = x_dim

        dims = [in_dim] + hidden_dims + [u_dim]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))

    def __call__(self, noisy_U_flat: jnp.ndarray, X: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            noisy_U_flat: (u_dim,) flattened noisy latent.
            X:            (x_dim,) observation.
            s:            scalar flow time.

        Returns:
            velocity: (u_dim,) predicted velocity.
        """
        s = jnp.atleast_1d(s)
        inp = jnp.concatenate([noisy_U_flat, X, s])
        x = inp
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        return x


# ---------------------------------------------------------------------------
# Flow matching loss (single sample)
# ---------------------------------------------------------------------------

def flow_matching_loss_single(model: VelocityMLP, U_flat: jnp.ndarray,
                              X: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Rectified flow matching loss for one (X, U) pair.

    1. Sample s ~ Uniform(0,1) and eps ~ N(0, I).
    2. z_s = (1-s)*eps + s*U.
    3. target velocity = U - eps.
    4. loss = ||v_theta(z_s, X, s) - target||^2.
    """
    k1, k2 = jax.random.split(key)
    s = jax.random.uniform(k1, shape=())
    eps = jax.random.normal(k2, shape=U_flat.shape)

    z_s = (1.0 - s) * eps + s * U_flat
    target = U_flat - eps

    pred = model(z_s, X, s)
    return jnp.mean((pred - target) ** 2)


# ---------------------------------------------------------------------------
# Batched loss
# ---------------------------------------------------------------------------

@eqx.filter_value_and_grad
def flow_matching_loss_batch(model: VelocityMLP, U: jnp.ndarray,
                             X: jnp.ndarray, keys: jax.Array) -> jnp.ndarray:
    """Mean rectified flow matching loss over a batch.

    Args:
        model: velocity network.
        U:     (batch, T+1, 2) latent marks.
        X:     (batch, T-2L+1) observations.
        keys:  (batch, 2) PRNG keys.

    Returns:
        Scalar mean loss.
    """
    batch = U.shape[0]
    U_flat = U.reshape(batch, -1)  # (batch, u_dim)

    losses = jax.vmap(
        lambda u, x, k: flow_matching_loss_single(model, u, x, k)
    )(U_flat, X, keys)

    return jnp.mean(losses)


# ---------------------------------------------------------------------------
# ODE sampling (Euler)
# ---------------------------------------------------------------------------

def sample_flow(model: VelocityMLP, X: jnp.ndarray, key: jax.Array,
                n_steps: int = 20) -> jnp.ndarray:
    """Sample U by integrating the learned ODE from noise.

    Uses ``jax.lax.fori_loop`` so the loop body is compiled once
    rather than being unrolled *n_steps* times.

    Args:
        model:   trained velocity network.
        X:       (x_dim,) observation vector.
        key:     PRNG key for initial noise.
        n_steps: number of Euler steps.

    Returns:
        U_flat: (u_dim,) sampled latent (flattened).
    """
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


def sample_flow_batch(model: VelocityMLP, X_batch: jnp.ndarray,
                      key: jax.Array, n_steps: int = 20) -> jnp.ndarray:
    """Sample U for a batch of observations.

    Args:
        model:   trained velocity network.
        X_batch: (batch, x_dim) observation vectors.
        key:     PRNG key.
        n_steps: number of Euler steps.

    Returns:
        U: (batch, T+1, 2) sampled latents.
    """
    batch = X_batch.shape[0]
    keys = jax.random.split(key, batch)
    T_plus_1 = model.u_dim // 2

    U_flat = jax.vmap(
        lambda x, k: sample_flow(model, x, k, n_steps)
    )(X_batch, keys)  # (batch, u_dim)

    return U_flat.reshape(batch, T_plus_1, 2)


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def make_optimizer(lr: float = 1e-3):
    """Create an Adam optimiser."""
    return optax.adam(lr)


# The train step is fully JIT-compiled: loss computation, gradient,
# and parameter update all run as a single fused GPU kernel.  The
# returned ``loss`` is a device array — no host sync happens until
# the caller explicitly reads it (e.g. via ``float(loss)``).
@eqx.filter_jit
def update_step(model: VelocityMLP, opt_state, optimizer, U: jnp.ndarray,
                X: jnp.ndarray, key: jax.Array):
    """Single gradient-descent update.

    Returns:
        (new_model, new_opt_state, loss)
    """
    batch = X.shape[0]
    keys = jax.random.split(key, batch)

    loss, grads = flow_matching_loss_batch(model, U, X, keys)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss
