"""Generative model for the Gaussian-bump additive process.

Mark space S = R^2, with each mark (rho, logsigma) defining a shape
    f(x) = rho * exp(-x^2 / (2 * exp(2*logsigma)))
on the integer grid {-L, ..., L}.

Given a shape distribution pi, a draw of the observation X in R^{T-2L+1}
is produced by sampling U_t ~ pi for t in {0,...,T} and summing the
shifted shapes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Shape evaluation
# ---------------------------------------------------------------------------

def shape_fn(rho: jnp.ndarray, logsigma: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Gaussian bump shape at integer positions *xs*.

    f(x) = rho * exp(-x^2 / (2 * exp(2*logsigma)))

    Args:
        rho:      scalar or array of amplitudes
        logsigma: scalar or array of log-scale parameters
        xs:       1-D array of integer positions, e.g. jnp.arange(-L, L+1)

    Returns:
        Array of shape (*rho.shape, len(xs)).
    """
    rho = jnp.asarray(rho)
    logsigma = jnp.asarray(logsigma)
    xs = jnp.asarray(xs, dtype=jnp.float32)
    var = jnp.exp(2.0 * logsigma)  # sigma^2
    # rho[..., None] * exp(...)  broadcasts over xs
    return rho[..., None] * jnp.exp(-xs**2 / (2.0 * var[..., None]))


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

class GaussianComponent(NamedTuple):
    """A single 2-D Gaussian component for the shape distribution."""
    weight: float
    mean: jnp.ndarray    # shape (2,)
    std: jnp.ndarray     # shape (2,)  (axis-aligned)


class ShapeDistribution(NamedTuple):
    """Mixture distribution on R^2 = (rho, logsigma).

    With probability *null_prob* we emit (0, logsigma) where logsigma is
    drawn from a standard normal (the value is irrelevant since rho=0).
    Otherwise we draw from a mixture of axis-aligned Gaussians given by
    *components*.
    """
    null_prob: float
    components: list  # list[GaussianComponent]


class EmpiricalDistribution(NamedTuple):
    """Empirical distribution: sample uniformly (with replacement) from *samples*."""
    samples: jnp.ndarray  # shape (N, 2)


def sample_pi(key: jax.Array, pi, n: int) -> jnp.ndarray:
    """Draw *n* i.i.d. marks from a shape distribution.

    Args:
        key: PRNG key.
        pi:  a ShapeDistribution or EmpiricalDistribution.
        n:   number of samples.

    Returns:
        marks: array of shape (n, 2).
    """
    if isinstance(pi, EmpiricalDistribution):
        return _sample_empirical(key, pi, n)
    return _sample_mixture(key, pi, n)


def _sample_empirical(key: jax.Array, pi: EmpiricalDistribution, n: int) -> jnp.ndarray:
    idxs = jax.random.randint(key, shape=(n,), minval=0, maxval=pi.samples.shape[0])
    return pi.samples[idxs]


def _sample_mixture(key: jax.Array, pi: ShapeDistribution, n: int) -> jnp.ndarray:
    k_null, k_comp, k_which, k_gauss = jax.random.split(key, 4)

    # Decide which samples are null (rho=0)
    is_null = jax.random.bernoulli(k_null, p=pi.null_prob, shape=(n,))

    # For non-null: pick a component
    n_comp = len(pi.components)
    weights = jnp.array([c.weight for c in pi.components])
    weights = weights / weights.sum()

    comp_idx = jax.random.choice(k_which, n_comp, shape=(n,), p=weights)

    # Stack component parameters
    means = jnp.stack([c.mean for c in pi.components])   # (n_comp, 2)
    stds = jnp.stack([c.std for c in pi.components])     # (n_comp, 2)

    chosen_mean = means[comp_idx]  # (n, 2)
    chosen_std = stds[comp_idx]    # (n, 2)

    z = jax.random.normal(k_gauss, shape=(n, 2))
    non_null_samples = chosen_mean + chosen_std * z

    # Null samples: rho=0, logsigma irrelevant (draw from N(0,1))
    null_logsigma = jax.random.normal(k_null, shape=(n,))
    null_samples = jnp.stack([jnp.zeros(n), null_logsigma], axis=-1)

    # Combine
    marks = jnp.where(is_null[:, None], null_samples, non_null_samples)
    return marks


# ---------------------------------------------------------------------------
# Observation computation
# ---------------------------------------------------------------------------

def compute_X(U: jnp.ndarray, L: int) -> jnp.ndarray:
    """Compute the observation vector X from a latent sequence U.

    Args:
        U: array of shape (T+1, 2), marks for t=0,...,T.
        L: support window.

    Returns:
        X: array of shape (T-2L+1,).
    """
    T_plus_1 = U.shape[0]
    T = T_plus_1 - 1
    n_obs = T - 2 * L + 1

    rho = U[:, 0]         # (T+1,)
    logsigma = U[:, 1]    # (T+1,)

    # Integer grid for the shape support
    xs = jnp.arange(-L, L + 1, dtype=jnp.float32)  # (2L+1,)

    # Vectorised: shapes[tau, dx] = rho_tau * exp(-(dx)^2 / 2*sigma_tau^2)
    # where dx ranges over -L..L
    var = jnp.exp(2.0 * logsigma)  # (T+1,)
    shapes = rho[:, None] * jnp.exp(-xs[None, :] ** 2 / (2.0 * var[:, None]))
    # shapes: (T+1, 2L+1)

    # X_t = sum_{tau} shapes[tau, t - tau] for t in {L,...,T-L}.
    # Observation index i = t - L, so i in {0,...,T-2L}.
    # For fixed tau, dx in {-L,...,L}: t = tau+dx, i = tau+dx-L.

    X = jnp.zeros(n_obs)
    for tau in range(T_plus_1):
        for dx_idx in range(2 * L + 1):
            dx = dx_idx - L
            t = tau + dx
            i = t - L
            if 0 <= i < n_obs:
                X = X.at[i].add(shapes[tau, dx_idx])
    return X


def compute_X_batched(U: jnp.ndarray, L: int) -> jnp.ndarray:
    """Vectorised observation computation for a batch of latent sequences.

    Uses explicit matrix construction instead of Python loops so the
    computation can be fully traced by JAX.

    Args:
        U: array of shape (batch, T+1, 2).
        L: support window.

    Returns:
        X: array of shape (batch, T-2L+1).
    """
    T_plus_1 = U.shape[1]
    T = T_plus_1 - 1
    n_obs = T - 2 * L + 1

    rho = U[:, :, 0]         # (batch, T+1)
    logsigma = U[:, :, 1]    # (batch, T+1)

    xs = jnp.arange(-L, L + 1, dtype=jnp.float32)  # (2L+1,)
    var = jnp.exp(2.0 * logsigma)  # (batch, T+1)

    # shapes: (batch, T+1, 2L+1)
    shapes = rho[:, :, None] * jnp.exp(
        -xs[None, None, :] ** 2 / (2.0 * var[:, :, None])
    )

    # Build a dense mapping matrix M of shape (T+1, 2L+1, n_obs)
    # M[tau, dx_idx, i] = 1 if tau + (dx_idx - L) - L == i, else 0
    tau_grid = jnp.arange(T_plus_1)[:, None]         # (T+1, 1)
    dx_grid = jnp.arange(2 * L + 1)[None, :]         # (1, 2L+1)
    obs_idx = tau_grid + (dx_grid - L) - L             # (T+1, 2L+1)

    i_grid = jnp.arange(n_obs)[None, None, :]         # (1, 1, n_obs)
    M = (obs_idx[:, :, None] == i_grid).astype(jnp.float32)  # (T+1, 2L+1, n_obs)

    # shapes: (batch, T+1, 2L+1)
    # Flatten tau and dx_idx dims for the matmul
    M_flat = M.reshape(T_plus_1 * (2 * L + 1), n_obs)       # (K, n_obs)
    shapes_flat = shapes.reshape(U.shape[0], -1)              # (batch, K)

    X = shapes_flat @ M_flat  # (batch, n_obs)
    return X


# ---------------------------------------------------------------------------
# Joint sampling
# ---------------------------------------------------------------------------

def sample_XU(key: jax.Array, pi, L: int, T: int, n_samples: int):
    """Draw n_samples independent (X, U) pairs.

    Args:
        key:       PRNG key.
        pi:        shape distribution (ShapeDistribution or EmpiricalDistribution).
        L:         support window.
        T:         visible window.
        n_samples: batch size.

    Returns:
        X: array of shape (n_samples, T-2L+1).
        U: array of shape (n_samples, T+1, 2).
    """
    keys = jax.random.split(key, n_samples + 1)
    sample_keys = keys[1:]

    # Draw marks for each sample: each sample needs (T+1) marks
    all_U = []
    for i in range(n_samples):
        marks = sample_pi(sample_keys[i], pi, T + 1)  # (T+1, 2)
        all_U.append(marks)
    U = jnp.stack(all_U)  # (n_samples, T+1, 2)

    X = compute_X_batched(U, L)  # (n_samples, T-2L+1)
    return X, U
