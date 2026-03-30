"""Rectified flow matching network and utilities.

**Two-flow architecture:**

1. **VFlowMLP** — learns the joint distribution of the binary gate V
   via rectified flow matching.  At inference the ODE output is rounded
   to {0, 1}.

2. **VelocityMLP** — maps (noisy_U, X, V, s) → velocity for U.
   Both the interpolated state *z_s*, the target velocity, and the
   predicted velocity are *masked* at V=0 positions so that the
   network never sees or produces non-zero values at null positions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial


# ---------------------------------------------------------------------------
# V-flow network
# ---------------------------------------------------------------------------

class VFlowMLP(eqx.Module):
    """MLP for flow matching on V (binary gate vector).

    Learns the joint distribution over all V positions via rectified
    flow matching.  At inference the ODE output is rounded to {0, 1}.
    """
    layers: list
    v_dim: int = eqx.field(static=True)   # T+1
    x_dim: int = eqx.field(static=True)   # T-2L+1

    def __init__(self, T: int, L: int, hidden_dims: list[int], *, key: jax.Array):
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

    def __call__(self, noisy_V: jnp.ndarray, X: jnp.ndarray,
                 s: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            noisy_V: (v_dim,) noisy latent gate vector.
            X:       (x_dim,) observation.
            s:       scalar flow time.

        Returns:
            velocity: (v_dim,) predicted velocity for V.
        """
        s = jnp.atleast_1d(s)
        inp = jnp.concatenate([noisy_V, X, s])
        x = inp
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# V-flow loss
# ---------------------------------------------------------------------------

def v_flow_loss_single(v_model: VFlowMLP, V: jnp.ndarray,
                       X: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Rectified flow matching loss for one (X, V) pair."""
    k1, k2 = jax.random.split(key)
    s = jax.random.uniform(k1, shape=())
    eps = jax.random.normal(k2, shape=V.shape)
    z_s = (1.0 - s) * eps + s * V
    target = V - eps
    pred = v_model(z_s, X, s)
    return jnp.mean((pred - target) ** 2)


@eqx.filter_value_and_grad
def v_flow_loss_batch(v_model: VFlowMLP, V: jnp.ndarray,
                      X: jnp.ndarray,
                      keys: jax.Array) -> jnp.ndarray:
    """Mean V-flow loss over a batch.

    Args:
        v_model: V-flow network.
        V:       (batch, T+1) binary targets.
        X:       (batch, x_dim) observations.
        keys:    (batch, 2) PRNG keys.
    """
    losses = jax.vmap(
        lambda v, x, k: v_flow_loss_single(v_model, v, x, k)
    )(V, X, keys)
    return jnp.mean(losses)


@eqx.filter_jit
def update_step_v(v_model: VFlowMLP, opt_state, optimizer,
                  X: jnp.ndarray, V: jnp.ndarray, key: jax.Array):
    """Single gradient-descent update for the V-flow."""
    batch = X.shape[0]
    keys = jax.random.split(key, batch)
    loss, grads = v_flow_loss_batch(v_model, V, X, keys)
    updates, new_opt_state = optimizer.update(grads, opt_state, v_model)
    new_model = eqx.apply_updates(v_model, updates)
    return new_model, new_opt_state, loss


@eqx.filter_jit
def eval_v_loss(v_model: VFlowMLP, V: jnp.ndarray,
                X: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Evaluate V-flow loss WITHOUT gradients."""
    batch = V.shape[0]
    keys = jax.random.split(key, batch)
    losses = jax.vmap(
        lambda v, x, k: v_flow_loss_single(v_model, v, x, k)
    )(V, X, keys)
    return jnp.mean(losses)


# ---------------------------------------------------------------------------
# V-flow sampling (ODE → round to {0, 1})
# ---------------------------------------------------------------------------

def sample_v_flow(v_model: VFlowMLP, X: jnp.ndarray,
                  key: jax.Array, n_steps: int = 20) -> jnp.ndarray:
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
def sample_v_flow_batch(v_model: VFlowMLP, X_batch: jnp.ndarray,
                        key: jax.Array,
                        n_steps: int = 20) -> jnp.ndarray:
    """Sample V for a batch of observations.

    Returns:
        V: (batch, T+1) binary gate vectors.
    """
    batch = X_batch.shape[0]
    keys = jax.random.split(key, batch)
    return jax.vmap(
        lambda x, k: sample_v_flow(v_model, x, k, n_steps)
    )(X_batch, keys)


@eqx.filter_jit
def eval_v_accuracy(v_model: VFlowMLP, X: jnp.ndarray,
                    V: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Sample V from V-flow and compute accuracy vs true V."""
    V_pred = sample_v_flow_batch(v_model, X, key)
    return jnp.mean(V_pred == V)


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

    The interpolated state z_s, target velocity, and predicted velocity
    are all masked at V=0 positions — not just the loss.

    1. Sample s ~ Uniform(0,1) and eps ~ N(0, I).
    2. z_s = ((1-s)*eps + s*U) * V_mask
    3. target = (U - eps) * V_mask
    4. pred = v_theta(z_s, X, V, s) * V_mask
    5. loss = mean_active(||pred - target||^2).
    """
    k1, k2 = jax.random.split(key)
    s = jax.random.uniform(k1, shape=())
    eps = jax.random.normal(k2, shape=U_flat.shape)

    # Mask: expand V (T+1,) to (u_dim,) by repeating each entry twice
    V_mask = jnp.repeat(V, 2)  # (u_dim,)

    # Mask interpolated state and target
    z_s = ((1.0 - s) * eps + s * U_flat) * V_mask
    target = (U_flat - eps) * V_mask

    pred = model(z_s, X, V, s)
    pred = pred * V_mask  # mask predicted velocity

    sq_err = (pred - target) ** 2
    n_active = jnp.maximum(V_mask.sum(), 1.0)
    return sq_err.sum() / n_active


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

    Both the initial noise and the velocity at each Euler step are
    masked at V=0 positions, so null positions stay exactly at zero
    throughout the integration.

    Args:
        model:   trained velocity network.
        X:       (x_dim,) observation vector.
        V:       (v_dim,) binary gate.
        key:     PRNG key for initial noise.
        n_steps: number of Euler steps.

    Returns:
        U_flat: (u_dim,) sampled latent (flattened).
    """
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
    # Clamp the logsigma components (odd indices) to prevent downstream NaN.
    logsigma_mask = jnp.tile(jnp.array([0.0, 1.0]), model.u_dim // 2)
    z = jnp.where(logsigma_mask, jnp.clip(z, -4.0, 4.0), z)
    z = z * V_mask  # final mask
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
