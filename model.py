import torch
import itertools
import math
from typing import List, Iterator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_latent_U(U_collection: torch.Tensor):
    """
    Print the U collection in a readable format.
    """
    result = torch.zeros(U_collection.shape[0], U_collection.shape[0] + U_collection.shape[1] -1,dtype=torch.int)
    for i in range(U_collection.shape[0]):
        result[i,i:i+U_collection.shape[1]] = U_collection[i, :]
    return result

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
    proposed_contribution_logpmf = parameters_to_contribution_logpmf(v, contrib_shape, contrib_pattern)
    proposed_observation_logpmf = contribution_logpmf_to_observation_logpmf(
        proposed_contribution_logpmf, n
    )

    # for every location where the proposed pmf if neginf, the observation had better be 0
    if not torch.all(observation_pmf[torch.isneginf(proposed_observation_logpmf)] ==0):
        print("proposed_observation_logpmf", proposed_observation_logpmf.flatten())
        print("observation_pmf", observation_pmf.flatten())
        raise ValueError("Mismatch in -inf patterns between proposed and observed logpmfs")

    mask = observation_pmf != 0
    loss = -torch.sum(observation_pmf[mask] * proposed_observation_logpmf[mask])
    return loss

    # return torch.mean((observation_pmf - torch.exp(proposed_observation_logpmf))**2)

def fit_contribution_logpmf(observation_pmf,contrib_shape,
                            true_contrib_logpmf=None,n_iter=1000, log_every=50,
                            initial_guess = None, lr=1, optmethod ='adam'):
    # get the sparsity pattern
    contrib_pattern,_ = contribution_sparsity_pattern_E0(contrib_shape)

    # if we have truth, exp it
    if true_contrib_logpmf is not None:
        true_contrib_pmf = torch.exp(true_contrib_logpmf)

    # make a random proposal
    if initial_guess is not None:
        params = torch.nn.Parameter(initial_guess.clone())
    else:
        params = torch.nn.Parameter(torch.rand(contrib_pattern.shape[0]))

    # get number of times we have to sample from contrib for each observation
    n = len(observation_pmf.shape) + len(contrib_shape) -1

    # sanity check shapes
    proposed_contribution_logpmf = parameters_to_contribution_logpmf(params, contrib_shape, contrib_pattern)
    proposed_observation_logpmf = contribution_logpmf_to_observation_logpmf(
        proposed_contribution_logpmf, n,
    )
    if tuple(proposed_contribution_logpmf.shape) != tuple(contrib_shape):
        raise ValueError(f"proposed_contribution_logpmf shape {tuple(proposed_contribution_logpmf.shape)} "
                         f"does not match contrib_shape {contrib_shape}")
    if proposed_observation_logpmf.shape != observation_pmf.shape:
        raise ValueError(f"proposed_observation_pmf shape {proposed_observation_logpmf.shape} "
                         f"does not match observation_pmf shape {observation_pmf.shape}")

    # run gradient descent
    losses=[]
    secret_losses=[]
    if optmethod == 'adam':
        optimizer = torch.optim.Adam([params], lr=lr)
    elif optmethod == 'lbfgs':
        optimizer = torch.optim.LBFGS([params], lr=lr, line_search_fn='strong_wolfe')
    else:
        raise ValueError(f"Unknown optimizer {optmethod}")

    def closure():
        optimizer.zero_grad()
        loss = fit_loss_fn(params, observation_pmf, contrib_shape, contrib_pattern, n)
        if torch.isnan(loss):
            raise ValueError("Loss is NaN. Aborting loop.")
        loss.backward()
        return loss

    for i in range(n_iter):
        if optmethod == 'lbfgs':
            loss = optimizer.step(closure)
        else:
            loss = closure()
            optimizer.step()

        if i % log_every == 0:
            losses.append(loss.item())

            if true_contrib_logpmf is not None:
                proposed_contribution_logpmf = parameters_to_contribution_logpmf(
                    params, contrib_shape, contrib_pattern)
                proposed_contribution_pmf = torch.exp(proposed_contribution_logpmf)
                secret_losses.append(torch.sum((proposed_contribution_pmf - true_contrib_pmf) ** 2).item())

            logger.info(
                f"Iteration {i}: Loss = {losses[-1]}, "
                f"Secret Loss = {secret_losses[-1] if true_contrib_logpmf is not None else 'N/A'}"
            )

            
            
    return params, losses, secret_losses

def parameters_to_contribution_logpmf(v: torch.Tensor,
                                   max_vals: List[int],
                                   sparsity_pattern: torch.LongTensor) -> torch.Tensor:
    """
    Args:
    ----
      v:                  shape (K,), real-valued “scores” for each pattern
      sparsity_pattern:   shape (K, N), each row is a length‑N index into the pmf

    Returns:
    -------
      logpmf: tensor of shape (M1, M2, …, MN), a valid log probability mass
           function, where Mj = max(sparsity_pattern[:,j]) + 1
    """
    K, N = sparsity_pattern.shape

    # 1) find the output shape: M_j = max over column j, plus one
    empirical_max_vals, _ = torch.max(sparsity_pattern, dim=0)
    
    # confirm that empirical_max_vals are all less than or equal to max_vals
    if len(max_vals) != len(empirical_max_vals) or not torch.all(empirical_max_vals <= torch.tensor(max_vals, device=empirical_max_vals.device)):
        raise ValueError("max_vals must be greater than or equal to empirical_max_vals")
    
    dims = max_vals   # [M1, M2, ..., MN]

    # 2) initialize log‑pmf to –∞
    logpmf = torch.full(dims, -math.inf, dtype=v.dtype, device=v.device)

    # 3) build an N‑tuple of length‑K index tensors for advanced indexing
    #    e.g. if N==3, idx = (sparsity_pattern[:,0],
    #                         sparsity_pattern[:,1],
    #                         sparsity_pattern[:,2])
    idx = tuple(sparsity_pattern[:, dim] for dim in range(N))

    # 4) scatter your scores (prepended with zero) into the log‑pmf
    logpmf[idx] = torch.cat([torch.tensor([0.0], device=v.device, dtype=v.dtype), v])

    # 5) done
    return logpmf - torch.logsumexp(logpmf.view(-1), dim=0)


def contribution_logpmf_to_observation_logpmf(
    logprior: torch.Tensor,  # shape = (M₁, …, M_N)
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

    N = logprior.ndim
    if n < N:
        raise ValueError(f"Length n={n} must be at least the vector dimension N={N}")
    T = n - N + 1  # number of X coordinates
    summation_columns = n + N - 1  # number of columns in the summation

    buffer_summing_columns = tuple(range(N-1)) + tuple(range(n, n+N-1))

    # make n versions of the prior
    # in the middle they will all be the same
    # but on the edge we must marginalize some of the axes
    logpmfs=[]

    for i in range(n):
        # construct reshaped prior with shape (1,1,...,1,M₁,M₂,...,M_N,1,...,1)
        # where the 1s are in the right places
        shp = [1] * summation_columns
        shp[i:i+N] = list(logprior.shape)
        logprior_reshaped = logprior.reshape(shp)

        if N>1:
            # we want to marginalize out the first axes and last axes
            # but in logsumexp land with gradients, this can create headaches
            # logsumexp([a,b]) does terrible things when a and b are -inf
            # to avoid this, we first compute the values in a way that is safe for -infs
            # in the backward pass because it returns finite values
            logprior_reshaped_inexact = torch.logsumexp(
                torch.clamp(logprior_reshaped,min=-1e20), dim=buffer_summing_columns, keepdim=True)

            # we then determine where the -infs should be in the final product
            mask = torch.isneginf(torch.logsumexp(
                logprior_reshaped, dim=buffer_summing_columns, keepdim=True))

            # put back -infs where they ought to go
            neginf = torch.full_like(logprior_reshaped_inexact, -math.inf)
            logprior_reshaped = torch.where(mask,neginf,logprior_reshaped_inexact)

            # and squeeze on either side
            logprior_reshaped = logprior_reshaped.squeeze(dim=buffer_summing_columns)

        logpmfs.append(logprior_reshaped)

    # convolve together
    convolved = convolve_n_logpmfs(logpmfs)

    return convolved

def convolve_n_logpmfs(logpmfs):
    result = convolve_logpmfs(logpmfs[0], logpmfs[1])
    for logpmf in logpmfs[2:]:
        result = convolve_logpmfs(result, logpmf)
    return result

def convolve_logpmfs(
    logpi:  torch.Tensor,  # shape = (M₁+1, …, M_N+1)
    logtau: torch.Tensor   # shape = (L₁+1, …, L_N+1)
) -> torch.Tensor:
    """
    Compute the PMF of X+Y when X∼π and Y∼τ are independent N‑dim integer vectors.
    Dense arrays only, no FFTs.

    Args
    ----
    logpi  : Tensor of shape (M1+1, …, M_N+1)
    logtau : Tensor of shape (L1+1, …, L_N+1)

    Returns
    -------
    sigma : Tensor of shape (M1+L1+1, …, M_N+L_N+1)
        sigma[x] = sum_{y+z = x} pi[y] * tau[z].
    """
    if logpi.ndim != logtau.ndim:
        raise ValueError("π and τ must have the same dimensionality")
    N = logpi.ndim

    # 1) Allocate the result array
    out_shape = [logpi.shape[k] + logtau.shape[k] - 1 for k in range(N)]
    logsigma = torch.ones(out_shape, dtype=logpi.dtype)*(-math.inf)

    # 2a) Find where tau is nonzero (skip all the zero‐mass shifts)
    #    idx has shape (n_nonzero, N), probs has shape (n_nonzero,)
    idx = torch.nonzero(~torch.isneginf(logtau), as_tuple=False)
    logprobs = logtau[tuple(idx.t().unbind())]

    # 2b) Find where pi is nonzero (skip all the zero‐mass shifts)
    logpi_submask = ~torch.isneginf(logpi.view(-1))
    logpi_submasked = logpi.view(-1)[logpi_submask]

    # 3) For each z with P[Y=z] = p_z, shift π by z and accumulate
    #
    #    sigma[ z + i ] += p_z * pi[i] for each integer vector i
    #    We do that by slicing into the big sigma tensor.
    #
    #    in logspace, this is log(sigma[ z + i ]) = log(exp(logsigma[ z + i ]) + exp(logp_z+logpi[i]))
    for shift, logp_z in zip(idx, logprobs):
        # build slices:  slice(shift[k], shift[k] + pi.shape[k])  for each axis k
        slices: List[slice] = [
            slice(int(shift[k]), int(shift[k]) + logpi.shape[k])
            for k in range(N)
        ]
        mask = torch.zeros(logsigma.shape, dtype=torch.bool)
        mask[tuple(slices)] = True
        # (note that mask.sum() is always equal to the number of elements in logpi)

        # if life were simple, we would just do 
        #              logsigma[mask] = logaddexp(logsigma[mask],logp_z + logpi)
        # but we have to deal with -infs and grad issues, so we do 
        #              logsigma[mask][submask] = logaddexp(logsigma[mask][submask],logp_z + logpi_submasked)
        # but we have to deal with the fact that in-place doesn't work, so we have to do this in two
        # steps of masked_scatter

        # get new values
        logsigma_mask = logsigma[mask]
        new_values = torch.logaddexp(logsigma_mask[logpi_submask], logp_z + logpi_submasked)

        # scatter them back in, somewhat tediously
        transformed_values = logsigma_mask.masked_scatter(logpi_submask, new_values)
        logsigma = logsigma.masked_scatter(mask, transformed_values)

    return logsigma

