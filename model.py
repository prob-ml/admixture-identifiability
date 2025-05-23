import itertools
import math
from typing import List, Iterator
import dataclasses
import numpy as np
import scipy as sp
import scipy.stats

import torch
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def contribution_sparsity_pattern_E0(N: List[int]):
    """
    Consider the space of pmfs on a nonnegative integer vector of length len(shape), with maximum
    values given by shape.  The native dimension of this space is the product of the shape - 1.
    However, we are only interested in PMFs that, with probability 1, guarantee that the vector
    U ~ PMF satisfies

    U[0]*0 + U[1]*1 + ... + U[N-1]*(N-1) = sum(U)*(N-1)/2

    This function returns a list of the values of U that satisfy this condition.
    """

    n = len(N)
    m = (n - 1) / 2

    Us=[]
    idxs=[]

    # iterate over the Cartesian product of ranges
    for i,U in enumerate(itertools.product(*(range(N[i]) for i in range(n)))):
        S = sum(U)
        W = sum(i * U[i] for i in range(n))
        # Check W == S*m
        if W == S * m:
            Us.append(U)
            idxs.append(i)

    return torch.tensor(Us, dtype=torch.int),torch.tensor(idxs, dtype=torch.int)

def fit_loss_fn(v, observation_pmf, contrib_shape, contrib_pattern, n):
    proposed_contribution_pmf = parameters_to_contribution_pmf(v, contrib_shape, contrib_pattern)
    proposed_observation_pmf = contribution_pmf_to_observation_pmf(
        proposed_contribution_pmf, n,
    )

    mask = observation_pmf != 0
    loss = -torch.sum(observation_pmf[mask] * torch.log(proposed_observation_pmf[mask]))
    return loss

def fit_contribution_pmf(observation_pmf,contrib_shape,
                            true_contrib_pmf=None,n_iter=1000, log_every=None,
                            initial_guess = None, fix_p0=True):
    # get the sparsity pattern
    contrib_pattern,_ = contribution_sparsity_pattern_E0(contrib_shape)

    # make a random proposal
    if initial_guess is not None:
        # check that initial guess sums to 1
        if not torch.allclose(
            initial_guess.sum(),
            torch.tensor(1.0, dtype=initial_guess.dtype, device=initial_guess.device)):
            raise ValueError("initial_guess must sum to 1")
        initial_guess = initial_guess / initial_guess.sum()
        params = torch.nn.Parameter(initial_guess.clone())
    else:
        guess = torch.rand(contrib_pattern.shape[0], dtype = observation_pmf.dtype, device = observation_pmf.device)
        guess = guess / guess.sum()
        params = torch.nn.Parameter(guess)

    # get number of times we have to sample from contrib for each observation
    n = len(observation_pmf.shape) + len(contrib_shape) -1

    # sanity check shapes
    proposed_contribution_pmf = parameters_to_contribution_pmf(params, contrib_shape, contrib_pattern)
    proposed_observation_pmf = contribution_pmf_to_observation_pmf(
        proposed_contribution_pmf, n,
    )
    if tuple(proposed_contribution_pmf.shape) != tuple(contrib_shape):
        raise ValueError(f"proposed_contribution_pmf shape {tuple(proposed_contribution_pmf.shape)} "
                         f"does not match contrib_shape {contrib_shape}")
    if proposed_observation_pmf.shape != observation_pmf.shape:
        raise ValueError(f"proposed_observation_pmf shape {proposed_observation_pmf.shape} "
                         f"does not match observation_pmf shape {observation_pmf.shape}")

    if fix_p0 and initial_guess is None:
        raise ValueError("fix_p0 is True, but initial_guess is None. "
                         "Please provide an initial guess that has p0")

    # run gradient descent
    losses=[]
    secret_losses=[]

    for i in range(n_iter):
        if params.grad is not None:
            params.grad.zero_()
        loss = fit_loss_fn(params, observation_pmf, contrib_shape, contrib_pattern, n)
        losses.append(loss.item())
        loss.backward()

        proposed_contribution_pmf = parameters_to_contribution_pmf(params, contrib_shape, contrib_pattern)
        if true_contrib_pmf is not None:
            mask = true_contrib_pmf!=0
            secret_losses.append(
                (proposed_contribution_pmf[mask] - true_contrib_pmf[mask]).abs().sum().item())

        newval = params*params.grad

        if fix_p0:
            newval[0] = initial_guess[0]
            newval[1:] = (1-initial_guess[0]) * newval[1:] / newval[1:].sum()
        else:
            newval = newval / newval.sum()

        params.data = newval

        if (log_every is not None) and i % log_every == 0:
            logger.info(
                f"Iteration {i}: Loss = {losses[-1]}, "
                f"Secret Loss = {secret_losses[-1] if true_contrib_pmf is not None else 'N/A'}"
            )

    return params, np.array(losses), np.array(secret_losses)

def sample_observation_pmf(observation_pmf, n_samples, rng):
    """
    Sample from the observation PMF.

    Args:
    ----
      observation_pmf: Tensor of shape (M₁, M₂, …, M_N)
      n_samples: int, number of samples to draw
      rng: random number generator

    Returns:
    -------
      resampled_observation_pmf: Tensor of shape (M₁, M₂, …, M_N)
    """
    # Ensure the PMF is a valid probability mass function
    if not torch.allclose(observation_pmf.sum(), torch.tensor(1.0, dtype=observation_pmf.dtype, device=observation_pmf.device)):
        raise ValueError("The observation PMF must sum to 1.")

    if n_samples is None:
        return observation_pmf

    # Flatten the observation PMF
    flat_obs_pmf = observation_pmf.flatten()

    # Sample indices according to the PMF
    sampled_idxs = torch.multinomial(
        flat_obs_pmf, num_samples=n_samples, replacement=True, generator=rng)

    # Count occurrences of each index
    sampled_counts = torch.bincount(
        sampled_idxs, minlength=len(flat_obs_pmf))

    # Normalize to get the resampled PMF
    resampled_observation_pmf = sampled_counts.float() / n_samples

    # Reshape and cast back
    resampled_observation_pmf = resampled_observation_pmf.reshape(observation_pmf.shape)
    resampled_observation_pmf = resampled_observation_pmf.to(dtype=observation_pmf.dtype)

    return resampled_observation_pmf


def parameters_to_contribution_pmf(v: torch.Tensor,
                                   max_vals: List[int],
                                   sparsity_pattern: torch.LongTensor) -> torch.Tensor:
    """
    Args:
    ----
      v:                  shape (K,), real-valued “scores” for each pattern
      sparsity_pattern:   shape (K, N), each row is a length‑N index into the pmf

    Returns:
    -------
      pmf: tensor of shape (M1, M2, …, MN), a valid probability mass
           function, where Mj = max(sparsity_pattern[:,j]) + 1
    """
    K, N = sparsity_pattern.shape

    dims = max_vals   # [M1, M2, ..., MN]

    # 2) initialize pmf to 0
    pmf = torch.zeros(dims, dtype=v.dtype, device=v.device)

    # 3) build an N‑tuple of length‑K index tensors for advanced indexing
    #    e.g. if N==3, idx = (sparsity_pattern[:,0],
    #                         sparsity_pattern[:,1],
    #                         sparsity_pattern[:,2])
    idx = tuple(sparsity_pattern[:, dim] for dim in range(N))

    # 4) scatter your scores into the pmf
    pmf[idx] = v

    # 5) done
    return pmf


def convolve_n_pmfs(pmfs: List[torch.Tensor]) -> torch.Tensor:
    result = convolve_pmfs(pmfs[0], pmfs[1])
    for pmf in pmfs[2:]:
        result = convolve_pmfs(result, pmf)
    return result


def build_mask_by_padding(
    full_shape: List[int],      # e.g. list(sigma.shape)
    patch_shape: List[int],     # e.g. list(pi.shape)
    shift: torch.Tensor         # 1D tensor of length N giving the offset
) -> torch.Tensor:
    '''
    In essence, this:

        # slices = [
        #     slice(int(shift[k]), int(shift[k]) + patch_shape[k])
        #     for k in range(N)
        # ]
        # mask = torch.zeros(full_shape, dtype=torch.bool)
        # mask[tuple(slices)] = True

    '''
    device = shift.device

    # turn shapes into tensors for arithmetic
    full  = torch.tensor(full_shape, dtype=torch.int64, device=device)
    patch = torch.tensor(patch_shape, dtype=torch.int64, device=device)

    # compute left/right pads per axis
    lefts  = shift
    rights = full - lefts - patch

    # reverse so we get [L_{N-1}, R_{N-1}, ..., L_0, R_0]
    N       = full.size(0)
    rev_idx = torch.arange(N - 1, -1, -1, device=device)
    lefts_r = lefts[rev_idx]
    rights_r= rights[rev_idx]

    # TorchScript needs a static-typed list here
    pads: List[int] = torch.stack([lefts_r, rights_r], dim=1).flatten().tolist()

    # build a tiny all-True block and pad it out to full_shape
    patch_block = torch.ones(patch_shape, dtype=torch.bool, device=device)
    mask = F.pad(patch_block, pads)

    return mask

def convolve_pmfs(
    pi:  torch.Tensor,  # shape = (M₁+1, …, M_N+1)
    tau: torch.Tensor   # shape = (L₁+1, …, L_N+1)
) -> torch.Tensor:
    """
    Compute the PMF of X+Y when X∼π and Y∼τ are independent N‑dim integer vectors.
    Dense arrays only, no FFTs.

    Args
    ----
    pi  : Tensor of shape (M1+1, …, M_N+1)
    tau : Tensor of shape (L1+1, …, L_N+1)

    Returns
    -------
    sigma : Tensor of shape (M1+L1+1, …, M_N+L_N+1)
        sigma[x] = sum_{y+z = x} pi[y] * tau[z].
    """
    if pi.ndim != tau.ndim:
        raise ValueError("π and τ must have the same dimensionality")
    N = pi.ndim

    # 1) Allocate the result array
    out_shape = [pi.shape[k] + tau.shape[k] - 1 for k in range(N)]
    sigma = torch.zeros(out_shape, dtype=pi.dtype, device=pi.device)

    # 2) Find where tau is nonzero (skip all the zero‐mass shifts)
    #    idx has shape (n_nonzero, N), probs has shape (n_nonzero,)
    nni = tau!=0
    idx = torch.nonzero(nni)
    probs = tau[nni]

    # 3) For each z with P[Y=z] = p_z, shift π by z and accumulate
    #
    #    sigma[ z + i ] += p_z * pi[i] for each integer vector i
    #    We do that by slicing into the big sigma tensor.
    for shift, p_z in zip(idx, probs):
        # build slices:  slice(shift[k], shift[k] + pi.shape[k])  for each axis k
        slices = [
            slice(int(shift[k]), int(shift[k]) + pi.shape[k])
            for k in range(N)
        ]

        sigma[slices] += pi * p_z

    return sigma

# @torch.jit.script
def contribution_pmf_to_observation_pmf(
    prior: torch.Tensor,  # shape = (M₁, …, M_N)
    n: int,  # length of stochastic process
) -> torch.Tensor:
    """
    Compute the full log PMF of the observation space of the following process:

    U_i iid ~ prior for i = 1, …, n
    X_i = sum_j U_{j,i-j} for i = N, …, n-N, with convention U_{j,i-j} = 0 unless 0 leq i-j < N

    Args
    ----
    prior : Tensor of shape (M₁, …, M_N)
        The prior distribution of the U_i.
    n    : int
        Length of the stochastic process.

    Returns
    -------
    pmf : Tensor of shape (L,L,...,L) where L = M₁ + M₂ + ... + M_N - 1
        The PMF of the observation space.
        pmf[x] = P[X=x] for x in the observation space.
        pmf will have (n-N+1) axes

    For example, let's say prior is a 2D tensor with shape (3, 4), and n = 5.
    Then we consider drawing 5 2d vectors, independently using pi, and summing
    them up as follows

    U0 = (U0_0, U0_1)
    U1 =       (U1_0, U1_1)
    U2 =             (U2_0, U2_1)
    U3 =                   (U3_0, U3_1)
    U4 =                         (U4_0, U4_1)       +
    -------------------------------------------------------
    X  =       (X0,   X1,   X2,   X3)

    Here M₁=3 and M₂=4 and N=2 and n=5.  We have that

        0 <= Ui_0 < 3
        0 <= Ui_1 < 4

    So

        0 <= Xi < 6 = 2+4-1

    The number of axes in X is 4, because n-N+1=5-2+1=4.  The number of columns in the summation
    is 6, because n+N-1=5+2-1=6.  Buffer is on either side is 1, because N-1=2-1=1.

    Next let's try n=5 and N=3.  We get

    U0 = (U0_0, U0_1, U0_2)
    U1 =       (U1_0, U1_1, U1_2)
    U2 =             (U2_0, U2_1, U2_2)
    U3 =                   (U3_0, U3_1, U3_2)
    U4 =                         (U4_0, U4_1, U4_2)       +
    -------------------------------------------------------
    X  =             (X0,   X1,   X2)

    Number of columns in the summation is 7, because n+N-1=5+3-1=7.  Buffer on either side
    is 2, because N-1=3-1=2.
    """

    N = prior.ndim
    if n < N:
        raise ValueError(f"Length n={n} must be at least the vector dimension N={N}")
    T = n - N + 1  # number of X coordinates
    summation_columns = n + N - 1  # number of columns in the summation

    # buffer_summing_columns = tuple(range(N-1)) + tuple(range(n, n+N-1))
    buffer_summing_columns = [i for i in range(N - 1)] + [i for i in range(n, n + N - 1)]

    # make n versions of the prior
    # in the middle they will all be the same
    # but on the edge we must marginalize some of the axes
    pmfs=[]

    for i in range(n):
        # construct reshaped prior with shape (1,1,...,1,M₁,M₂,...,M_N,1,...,1)
        # where the 1s are in the right places
        # shp = [1] * summation_columns
        # shp[i:i+N] = list(prior.shape)
        shp: List[int] = [1] * i + list(prior.shape) + [1] * (summation_columns - i-len(prior.shape))
        prior_reshaped = prior.reshape(shp)

        if N>1:
            # and squeeze on either side
            prior_reshaped = torch.sum(prior_reshaped, buffer_summing_columns)

        pmfs.append(prior_reshaped)

    # convolve together
    convolved = convolve_n_pmfs(pmfs)

    return convolved

@dataclasses.dataclass
class SimulationResult:
    params: torch.Tensor
    contrib_shape: List[int]
    losses: np.ndarray
    secret_losses: np.ndarray
    observation_pmf: torch.Tensor
    corrupted_observation_pmf: torch.Tensor
    secret_v: torch.Tensor
    initial_guess_v: torch.Tensor
    n_samples: int
    n_actual_samples: int
    n_iter: int
    n: int

    def to(self, device: torch.device):
        self.params = self.params.to(device)
        self.observation_pmf = self.observation_pmf.to(device)
        self.corrupted_observation_pmf = self.corrupted_observation_pmf.to(device)
        self.secret_v = self.secret_v.to(device)
        self.initial_guess_v = self.initial_guess_v.to(device)

    def __post_init__(self):
        self.params = self.params.detach()
        self.p0 = self.secret_v[0].item()

    def __repr__(self):
        return f"SimulationResult(contrib_shape={self.contrib_shape}, " \
               f"final loss={self.losses[-1]:.3f}, final secret loss={self.secret_losses[-1]:.1e}, " \
               f"n_samples={self.n_samples}, n_actual_samples={self.n_actual_samples})"

def n_sources_pmf(n,p0):
    "Return PMF on the total number of sources in each observation"
    # Binom(n, self.p0)
    return sp.stats.binom.pmf(np.r_[0:n],n,1-p0)


def run_simulation(n, contrib_shape, device, p0, n_samples, rng, n_iter, log_every=None):
    # setup problem
    contrib_pattern,contrib_pattern_idxs = contribution_sparsity_pattern_E0(contrib_shape)
    contrib_pattern = contrib_pattern.to(device)
    n_params = len(contrib_pattern)
    secret_v = torch.rand(n_params, dtype=torch.float64, device=device, generator=rng)
    secret_v[0] = p0
    secret_v[1:] = (1-p0)*secret_v[1:]/secret_v[1:].sum()
    contribution_pmf = parameters_to_contribution_pmf(
        secret_v, contrib_shape, contrib_pattern)
    observation_pmf = contribution_pmf_to_observation_pmf(contribution_pmf, n)

    # n_samples is supposed to represent number of *nontrivial* samples
    if n_samples is not None:
        ptrivial = observation_pmf.view(-1)[0].item()
        n_actual_samples = int(n_samples / (1 - ptrivial))
    else:
        n_actual_samples = None

    corrupted_observation_pmf = sample_observation_pmf(observation_pmf, n_actual_samples, rng)

    # make initial guess
    initial_guess_v = torch.ones((n_params,), dtype=torch.float64, device=device)
    initial_guess_v[0] = p0
    initial_guess_v[1:] = (1-initial_guess_v[0])*initial_guess_v[1:]/initial_guess_v[1:].sum()

    # run EM
    params, losses, secret_losses = fit_contribution_pmf(
        corrupted_observation_pmf,
        contrib_shape,
        true_contrib_pmf=contribution_pmf,
        initial_guess=initial_guess_v,
        n_iter=n_iter,
        log_every=log_every,
        fix_p0=True,
    )

    return SimulationResult(
        params=params,
        contrib_shape=contrib_shape,
        losses=losses,
        secret_losses=secret_losses,
        observation_pmf=observation_pmf,
        corrupted_observation_pmf=corrupted_observation_pmf,
        secret_v=secret_v,
        initial_guess_v=initial_guess_v,
        n_samples=n_samples,
        n_actual_samples=n_actual_samples,
        n_iter=n_iter,
        n= n,
    )

