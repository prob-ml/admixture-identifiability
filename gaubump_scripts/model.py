"""Generative model for the Gaussian-bump additive process.

Mark space S = R^2, with each mark (rho, logsigma) defining a shape
    f(x) = V * rho * exp(-x^2 / (2 * exp(2*logsigma)))
on the integer grid {-L, ..., L}.

A binary latent **V** gates each mark: V=0 means the mark is null
(contributing nothing to X) and V=1 means it is active with amplitude
rho used directly.  Null marks always have rho=0, logsigma=0 by
convention.

Given a shape distribution pi, a draw of the observation X in R^{T-2L+1}
is produced by sampling (V_t, U_t) ~ pi for t in {0,...,T} and summing
the shifted shapes gated by V.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Shape evaluation
# ---------------------------------------------------------------------------

def shape_fn(V: jnp.ndarray, rho: jnp.ndarray, logsigma: jnp.ndarray,
             xs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Gaussian bump shape at integer positions *xs*.

    f(x) = V * rho * exp(-x^2 / (2 * exp(2*logsigma)))

    Args:
        V:        scalar or array, binary gate (0=null, 1=active).
        rho:      scalar or array of amplitudes.
        logsigma: scalar or array of log-scale parameters.
        xs:       1-D array of integer positions, e.g. jnp.arange(-L, L+1)

    Returns:
        Array of shape (*rho.shape, len(xs)).
    """
    V = jnp.asarray(V, dtype=jnp.float32)
    rho = jnp.asarray(rho)
    logsigma = jnp.asarray(logsigma)
    xs = jnp.asarray(xs, dtype=jnp.float32)
    # Clamp logsigma to prevent exp overflow/underflow (NaN when var→0).
    logsigma = jnp.clip(logsigma, -4.0, 4.0)
    var = jnp.exp(2.0 * logsigma)  # sigma^2
    return (V[..., None] * rho[..., None]
            * jnp.exp(-xs**2 / (2.0 * var[..., None])))


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

    With probability *null_prob* we emit V=0 with (rho, logsigma) = (0, 0).
    Otherwise we emit V=1 and draw (rho, logsigma) from a mixture of
    axis-aligned Gaussians given by *components*.
    """
    null_prob: float
    components: list  # list[GaussianComponent]


class EmpiricalDistribution(NamedTuple):
    """Empirical distribution: sample uniformly (with replacement).

    *V_samples* is a (N,) binary array and *U_samples* is (N, 2).
    """
    V_samples: jnp.ndarray  # shape (N,)
    U_samples: jnp.ndarray  # shape (N, 2)


def sample_pi(key: jax.Array, pi, n: int):
    """Draw *n* i.i.d. (V, mark) tuples from a shape distribution.

    Args:
        key: PRNG key.
        pi:  a ShapeDistribution or EmpiricalDistribution.
        n:   number of samples.

    Returns:
        V:     array of shape (n,), binary gate.
        marks: array of shape (n, 2).
    """
    if isinstance(pi, EmpiricalDistribution):
        return _sample_empirical(key, pi, n)
    return _sample_mixture(key, pi, n)


def _sample_empirical(key: jax.Array, pi: EmpiricalDistribution, n: int):
    idxs = jax.random.randint(key, shape=(n,), minval=0, maxval=pi.V_samples.shape[0])
    return pi.V_samples[idxs], pi.U_samples[idxs]


def _sample_mixture(key: jax.Array, pi: ShapeDistribution, n: int):
    k_null, k_comp, k_which, k_gauss = jax.random.split(key, 4)

    # V: 1 = active, 0 = null
    is_null = jax.random.bernoulli(k_null, p=pi.null_prob, shape=(n,))
    V = (1 - is_null).astype(jnp.float32)

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

    # Null samples: (0, 0) — V gates these out
    null_samples = jnp.zeros((n, 2))

    # Combine
    marks = jnp.where(V[:, None] > 0.5, non_null_samples, null_samples)
    return V, marks


# ---------------------------------------------------------------------------
# Observation computation
# ---------------------------------------------------------------------------

def compute_X(V: jnp.ndarray, U: jnp.ndarray, L: int) -> jnp.ndarray:
    """Compute the observation vector X from a latent sequence (V, U).

    Args:
        V: array of shape (T+1,), binary gate.
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

    # Vectorised: shapes[tau, dx] = V_tau * rho_tau * exp(-(dx)^2 / 2*sigma_tau^2)
    logsigma = jnp.clip(logsigma, -4.0, 4.0)
    var = jnp.exp(2.0 * logsigma)  # (T+1,)
    shapes = V[:, None] * rho[:, None] * jnp.exp(-xs[None, :] ** 2 / (2.0 * var[:, None]))
    # shapes: (T+1, 2L+1)

    X = jnp.zeros(n_obs)
    for tau in range(T_plus_1):
        for dx_idx in range(2 * L + 1):
            dx = dx_idx - L
            t = tau + dx
            i = t - L
            if 0 <= i < n_obs:
                X = X.at[i].add(shapes[tau, dx_idx])
    return X


def compute_X_batched(V: jnp.ndarray, U: jnp.ndarray, L: int) -> jnp.ndarray:
    """Vectorised observation computation for a batch of latent sequences.

    Uses explicit matrix construction instead of Python loops so the
    computation can be fully traced by JAX.

    Args:
        V: array of shape (batch, T+1), binary gate.
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
    logsigma = jnp.clip(logsigma, -4.0, 4.0)
    var = jnp.exp(2.0 * logsigma)  # (batch, T+1)

    # shapes: (batch, T+1, 2L+1) — gated by V
    shapes = V[:, :, None] * rho[:, :, None] * jnp.exp(
        -xs[None, None, :] ** 2 / (2.0 * var[:, :, None])
    )

    # Build a dense mapping matrix M of shape (T+1, 2L+1, n_obs)
    tau_grid = jnp.arange(T_plus_1)[:, None]         # (T+1, 1)
    dx_grid = jnp.arange(2 * L + 1)[None, :]         # (1, 2L+1)
    obs_idx = tau_grid + (dx_grid - L) - L             # (T+1, 2L+1)

    i_grid = jnp.arange(n_obs)[None, None, :]         # (1, 1, n_obs)
    M = (obs_idx[:, :, None] == i_grid).astype(jnp.float32)  # (T+1, 2L+1, n_obs)

    M_flat = M.reshape(T_plus_1 * (2 * L + 1), n_obs)       # (K, n_obs)
    shapes_flat = shapes.reshape(U.shape[0], -1)              # (batch, K)

    X = shapes_flat @ M_flat  # (batch, n_obs)
    return X


# ---------------------------------------------------------------------------
# Joint sampling
# ---------------------------------------------------------------------------

def sample_XU(key: jax.Array, pi, L: int, T: int, n_samples: int):
    """Draw n_samples independent (X, V, U) triples.

    Args:
        key:       PRNG key.
        pi:        shape distribution (ShapeDistribution or EmpiricalDistribution).
        L:         support window.
        T:         visible window.
        n_samples: batch size.

    Returns:
        X: array of shape (n_samples, T-2L+1).
        V: array of shape (n_samples, T+1), binary gate.
        U: array of shape (n_samples, T+1, 2).
    """
    total_marks = n_samples * (T + 1)
    V_flat, marks = sample_pi(key, pi, total_marks)
    V = V_flat.reshape(n_samples, T + 1)
    U = marks.reshape(n_samples, T + 1, 2)

    X = compute_X_batched(V, U, L)
    return X, V, U


# ---------------------------------------------------------------------------
# JIT-compiled data generation
# ---------------------------------------------------------------------------

def stack_mixture_params(pi: ShapeDistribution):
    """Pre-stack mixture parameters into JAX arrays for JIT-compiled sampling.

    Returns:
        (null_prob, weights, means, stds) where weights, means, stds are
        JAX arrays of shapes (n_comp,), (n_comp, 2), (n_comp, 2).
    """
    weights = jnp.array([c.weight for c in pi.components])
    weights = weights / weights.sum()
    means = jnp.stack([c.mean for c in pi.components])
    stds = jnp.stack([c.std for c in pi.components])
    return pi.null_prob, weights, means, stds


@partial(jax.jit, static_argnums=(5, 6, 7))
def sample_XVU_mixture(key, null_prob, weights, means, stds, L, T, n_samples):
    """JIT-compiled (X, V, U) sampling from a Gaussian mixture shape distribution.

    Use :func:`stack_mixture_params` to pre-compute the array arguments
    from a :class:`ShapeDistribution`.  *L*, *T*, and *n_samples* are
    static (they determine array shapes).

    Returns:
        X: (n_samples, T-2L+1)
        V: (n_samples, T+1)  binary
        U: (n_samples, T+1, 2)
    """
    total = n_samples * (T + 1)
    k_null, k_which, k_gauss = jax.random.split(key, 3)

    is_null = jax.random.bernoulli(k_null, p=null_prob, shape=(total,))
    V = (1 - is_null).astype(jnp.float32)

    comp_idx = jax.random.choice(
        k_which, weights.shape[0], shape=(total,), p=weights)
    chosen_mean = means[comp_idx]
    chosen_std = stds[comp_idx]
    z = jax.random.normal(k_gauss, shape=(total, 2))
    non_null = chosen_mean + chosen_std * z

    null_marks = jnp.zeros((total, 2))
    marks = jnp.where(V[:, None] > 0.5, non_null, null_marks)

    V = V.reshape(n_samples, T + 1)
    U = marks.reshape(n_samples, T + 1, 2)
    X = compute_X_batched(V, U, L)
    return X, V, U


@partial(jax.jit, static_argnums=(3, 4, 5))
def sample_XVU_empirical(key, V_samples, U_samples, L, T, n_samples):
    """JIT-compiled (X, V, U) sampling from an empirical distribution.

    *V_samples* is (N,) binary and *U_samples* is (N, 2).  Marks are drawn
    with replacement.  *L*, *T*, and *n_samples* are static.

    Returns:
        X: (n_samples, T-2L+1)
        V: (n_samples, T+1)  binary
        U: (n_samples, T+1, 2)
    """
    total = n_samples * (T + 1)
    idxs = jax.random.randint(
        key, shape=(total,), minval=0, maxval=V_samples.shape[0])
    V = V_samples[idxs].reshape(n_samples, T + 1)
    U = U_samples[idxs].reshape(n_samples, T + 1, 2)
    X = compute_X_batched(V, U, L)
    return X, V, U
