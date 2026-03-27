"""Rectified flow matching network and utilities.

**Two-model architecture:**

1. **VClassifierMLP** — predicts the binary null/active gate V from X.
   Trained with binary cross-entropy.

2. **VelocityMLP** — maps (noisy_U, X, V, s) → velocity.
   Flow matching loss is *masked* so that only active (V=1) positions
   contribute, preventing wasted capacity on trivially-zero null marks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial


# ---------------------------------------------------------------------------
# V-classifier network
# ---------------------------------------------------------------------------

class VClassifierMLP(eqx.Module):
    """MLP classifier: X → V logits (one per position)."""
    layers: list
    x_dim: int = eqx.field(static=True)  # T-2L+1
    v_dim: int = eqx.field(static=True)  # T+1

    def __init__(self, T: int, L: int, hidden_dims: list[int], *, key: jax.Array):
        x_dim = T - 2 * L + 1
        v_dim = T + 1
        self.x_dim = x_dim
        self.v_dim = v_dim

        dims = [x_dim] + hidden_dims + [v_dim]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            X: (x_dim,) observation.

        Returns:
            logits: (v_dim,) unnormalised log-odds for each position.
        """
        x = X
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        logits = self.layers[-1](x)
        return logits


# ---------------------------------------------------------------------------
# V-classifier loss
# ---------------------------------------------------------------------------

def v_classifier_loss_single(v_model: VClassifierMLP, X: jnp.ndarray,
                             V: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy loss for one (X, V) pair."""
    logits = v_model(X)  # (v_dim,)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, V))


@eqx.filter_value_and_grad
def v_classifier_loss_batch(v_model: VClassifierMLP, X: jnp.ndarray,
                            V: jnp.ndarray) -> jnp.ndarray:
    """Mean BCE loss over a batch.

    Args:
        v_model: classifier.
        X:       (batch, x_dim).
        V:       (batch, T+1) binary.
    """
    losses = jax.vmap(
        lambda x, v: v_classifier_loss_single(v_model, x, v)
    )(X, V)
    return jnp.mean(losses)


@eqx.filter_jit
def update_step_v(v_model: VClassifierMLP, opt_state, optimizer,
                  X: jnp.ndarray, V: jnp.ndarray):
    """Single gradient-descent update for the V-classifier."""
    loss, grads = v_classifier_loss_batch(v_model, X, V)
    updates, new_opt_state = optimizer.update(grads, opt_state, v_model)
    new_model = eqx.apply_updates(v_model, updates)
    return new_model, new_opt_state, loss


@eqx.filter_jit
def eval_v_loss(v_model: VClassifierMLP, X: jnp.ndarray,
                V: jnp.ndarray) -> jnp.ndarray:
    """Evaluate V-classifier loss WITHOUT gradients."""
    losses = jax.vmap(
        lambda x, v: v_classifier_loss_single(v_model, x, v)
    )(X, V)
    return jnp.mean(losses)


@eqx.filter_jit
def eval_v_accuracy(v_model: VClassifierMLP, X: jnp.ndarray,
                    V: jnp.ndarray) -> jnp.ndarray:
    """Compute classification accuracy on a batch."""
    logits = jax.vmap(v_model)(X)  # (batch, v_dim)
    preds = (logits > 0.0).astype(jnp.float32)
    return jnp.mean(preds == V)


# ---------------------------------------------------------------------------
# Velocity network (conditioned on V)
# ---------------------------------------------------------------------------

class VelocityMLP(eqx.Module):
    """MLP velocity network for rectified flow matching, conditioned on V."""
    layers: list
    u_dim: int = eqx.field(static=True)  # (T+1)*2
    x_dim: int = eqx.field(static=True)  # T-2L+1
    v_dim: int = eqx.field(static=True)  # T+1

    def __init__(self, T: int, L: int, hidden_dims: list[int], *, key: jax.Array):
        u_dim = (T + 1) * 2
        x_dim = T - 2 * L + 1
        v_dim = T + 1
        in_dim = u_dim + x_dim + v_dim + 1  # noisy_U + X + V + s

        self.u_dim = u_dim
        self.x_dim = x_dim
        self.v_dim = v_dim

        dims = [in_dim] + hidden_dims + [u_dim]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))

    def __call__(self, noisy_U_flat: jnp.ndarray, X: jnp.ndarray,
                 V: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            noisy_U_flat: (u_dim,) flattened noisy latent.
            X:            (x_dim,) observation.
            V:            (v_dim,) binary gate.
            s:            scalar flow time.

        Returns:
            velocity: (u_dim,) predicted velocity.
        """
        s = jnp.atleast_1d(s)
        inp = jnp.concatenate([noisy_U_flat, X, V, s])
        x = inp
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        return x


# ---------------------------------------------------------------------------
# Flow matching loss (single sample, masked)
# ---------------------------------------------------------------------------

def flow_matching_loss_single(model: VelocityMLP, U_flat: jnp.ndarray,
                              X: jnp.ndarray, V: jnp.ndarray,
                              key: jax.Array) -> jnp.ndarray:
    """Rectified flow matching loss for one (X, V, U) triple.

    Only active (V=1) positions contribute to the loss.

    1. Sample s ~ Uniform(0,1) and eps ~ N(0, I).
    2. z_s = (1-s)*eps + s*U.
    3. target velocity = U - eps.
    4. loss = masked_mean(||v_theta(z_s, X, V, s) - target||^2).
    """
    k1, k2 = jax.random.split(key)
    s = jax.random.uniform(k1, shape=())
    eps = jax.random.normal(k2, shape=U_flat.shape)

    z_s = (1.0 - s) * eps + s * U_flat
    target = U_flat - eps

    pred = model(z_s, X, V, s)
    sq_err = (pred - target) ** 2  # (u_dim,)

    # Mask: expand V (T+1,) to (u_dim,) by repeating each entry twice
    # (once for rho, once for logsigma)
    V_mask = jnp.repeat(V, 2)  # (u_dim,)
    masked_err = sq_err * V_mask
    n_active = jnp.maximum(V_mask.sum(), 1.0)
    return masked_err.sum() / n_active


# ---------------------------------------------------------------------------
# Batched loss
# ---------------------------------------------------------------------------

@eqx.filter_value_and_grad
def flow_matching_loss_batch(model: VelocityMLP, U: jnp.ndarray,
                             X: jnp.ndarray, V: jnp.ndarray,
                             keys: jax.Array) -> jnp.ndarray:
    """Mean masked flow matching loss over a batch.

    Args:
        model: velocity network.
        U:     (batch, T+1, 2) latent marks.
        X:     (batch, T-2L+1) observations.
        V:     (batch, T+1) binary gates.
        keys:  (batch, 2) PRNG keys.

    Returns:
        Scalar mean loss.
    """
    batch = U.shape[0]
    U_flat = U.reshape(batch, -1)  # (batch, u_dim)

    losses = jax.vmap(
        lambda u, x, v, k: flow_matching_loss_single(model, u, x, v, k)
    )(U_flat, X, V, keys)

    return jnp.mean(losses)


# ---------------------------------------------------------------------------
# ODE sampling (Euler)
# ---------------------------------------------------------------------------

def sample_flow(model: VelocityMLP, X: jnp.ndarray, V: jnp.ndarray,
                key: jax.Array, n_steps: int = 20) -> jnp.ndarray:
    """Sample U by integrating the learned ODE from noise.

    Null positions (V=0) are zeroed after integration.

    Args:
        model:   trained velocity network.
        X:       (x_dim,) observation vector.
        V:       (v_dim,) binary gate.
        key:     PRNG key for initial noise.
        n_steps: number of Euler steps.

    Returns:
        U_flat: (u_dim,) sampled latent (flattened).
    """
    z = jax.random.normal(key, shape=(model.u_dim,))
    dt = 1.0 / n_steps

    def euler_step(i, z):
        s = i * dt
        v = model(z, X, V, jnp.array(s))
        return z + dt * v

    z = jax.lax.fori_loop(0, n_steps, euler_step, z)
    # Clamp the logsigma components (odd indices) to prevent downstream NaN.
    logsigma_mask = jnp.tile(jnp.array([0.0, 1.0]), model.u_dim // 2)
    z = jnp.where(logsigma_mask, jnp.clip(z, -4.0, 4.0), z)
    # Zero out null positions
    V_mask = jnp.repeat(V, 2)
    z = z * V_mask
    return z


def sample_flow_batch(model: VelocityMLP, X_batch: jnp.ndarray,
                      V_batch: jnp.ndarray, key: jax.Array,
                      n_steps: int = 20) -> jnp.ndarray:
    """Sample U for a batch of observations.

    Args:
        model:   trained velocity network.
        X_batch: (batch, x_dim) observation vectors.
        V_batch: (batch, v_dim) binary gates.
        key:     PRNG key.
        n_steps: number of Euler steps.

    Returns:
        U: (batch, T+1, 2) sampled latents.
    """
    batch = X_batch.shape[0]
    keys = jax.random.split(key, batch)
    T_plus_1 = model.u_dim // 2

    U_flat = jax.vmap(
        lambda x, v, k: sample_flow(model, x, v, k, n_steps)
    )(X_batch, V_batch, keys)  # (batch, u_dim)

    return U_flat.reshape(batch, T_plus_1, 2)


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def make_optimizer(lr: float = 1e-3):
    """Create an Adam optimiser."""
    return optax.adam(lr)


@eqx.filter_jit
def update_step(model: VelocityMLP, opt_state, optimizer, U: jnp.ndarray,
                X: jnp.ndarray, V: jnp.ndarray, key: jax.Array):
    """Single gradient-descent update for the U-flow.

    Returns:
        (new_model, new_opt_state, loss)
    """
    batch = X.shape[0]
    keys = jax.random.split(key, batch)

    loss, grads = flow_matching_loss_batch(model, U, X, V, keys)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


@eqx.filter_jit
def eval_flow_loss(model: VelocityMLP, U: jnp.ndarray, X: jnp.ndarray,
                   V: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Evaluate masked flow matching loss WITHOUT gradients."""
    batch = U.shape[0]
    U_flat = U.reshape(batch, -1)
    keys = jax.random.split(key, batch)
    losses = jax.vmap(
        lambda u, x, v, k: flow_matching_loss_single(model, u, x, v, k)
    )(U_flat, X, V, keys)
    return jnp.mean(losses)
