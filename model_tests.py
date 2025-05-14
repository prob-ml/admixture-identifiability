# %% imports
%load_ext autoreload
%autoreload 2
import model
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

model.logger.setLevel("INFO")

#####################################
# %% check that contribution_pmf_to_observation_pmf
# seems to be about right

n = 7

pmf = torch.rand(2, 3, 2, 2)
# pmf = torch.zeros(1, 3, 1, 2)
# pmf[0, 0, 0, 0] = .95
# pmf[0, 2, 0, 0] = .05

N=pmf.ndim
pmf = pmf / pmf.sum()

# get theoretical observation pmf
observation_logpmf = model.contribution_logpmf_to_observation_logpmf(torch.log(pmf), n)

# make a mazillion samples
mc_samples = 5000

observed_sample_counts = torch.zeros_like(observation_logpmf)
for i in range(mc_samples):
    U_samples = torch.multinomial(pmf.flatten(), n, replacement=True)
    U_samples_as_vectors = torch.stack(
        torch.unravel_index(U_samples, pmf.shape),
        dim=1,
    )
    X = torch.zeros(n+N-1, dtype=torch.int)
    for j in range(n):
        X[j:j+N] += U_samples_as_vectors[j, :]
    
    X_seen = X[N-1:1-N]

    observed_sample_counts[tuple(X_seen)] += 1

# should be close to 0
print((observed_sample_counts/mc_samples - torch.exp(observation_logpmf)).abs().max())



#####################################
# %% example gradienting

# make some ground truth
n = 5
contrib_shape = (2, )*3
N=len(contrib_shape)
contrib_pattern,contrib_pattern_idxs = model.contribution_sparsity_pattern_E0(contrib_shape)
n_params = len(contrib_pattern)-1
secret_v = torch.randn(n_params)
contribution_logpmf = model.parameters_to_contribution_logpmf(
    secret_v, contrib_shape, contrib_pattern)

# first let us check that we can convolve pmfs safely
guess_v = torch.randn(n_params)

guess_v1 = guess_v.clone().requires_grad_()
guess_contribution_logpmf = model.parameters_to_contribution_logpmf(
    guess_v1, contrib_shape, contrib_pattern)
convolved1 = model.convolve_n_logpmfs([
    guess_contribution_logpmf[:,    :,    :,    None, None, None, None],
    guess_contribution_logpmf[None, :,    :,    :,    None, None, None],
    guess_contribution_logpmf[None, None, :,    :,    :,    None, None],
    guess_contribution_logpmf[None, None, None, :,    :,    :,    None],
    guess_contribution_logpmf[None, None, None, None, :,    :,    :   ],
])
convolved1 = torch.logsumexp(convolved1, dim=(0,1,5,6))

guess_v2 = guess_v.clone().requires_grad_()
guess_contribution_logpmf = model.parameters_to_contribution_logpmf(
    guess_v2, contrib_shape, contrib_pattern)
convolved2 = model.contribution_logpmf_to_observation_logpmf(guess_contribution_logpmf, n)

mask = ~torch.isneginf(convolved1)
loss1 = torch.sum(convolved1[mask])
loss1.backward()
grad1 = guess_v1.grad

mask = ~torch.isneginf(convolved2)
loss2 = torch.sum(convolved2[mask])
loss2.backward()
grad2 = guess_v2.grad

# Confirm that loss1 is close to loss2
assert torch.allclose(loss1, loss2, atol=1e-6), f"loss1 ({loss1}) and loss2 ({loss2}) are not close"

# Confirm that grad1 is close to grad2
assert torch.allclose(grad1, grad2, atol=1e-6), "grad1 and grad2 are not close"



#####################################
# %% example training

# make some ground truth
n = 10
contrib_shape = (3, )*5
N=len(contrib_shape)
contrib_pattern,contrib_pattern_idxs = model.contribution_sparsity_pattern_E0(contrib_shape)
n_params = len(contrib_pattern)-1

# secret_v = torch.rand(len(contrib_pattern))
# contribution_logpmf = model.parameters_to_contribution_logpmf(
#     secret_v, contrib_shape, contrib_pattern)

patterns=[9,15]
secret_v= torch.full((n_params,), -20.0)
print('patterns in truth')
print('.   [off] --> logit is 0.0')
print(f'.   [default] --> logit is {secret_v[0].item()}')
for p in patterns:
    secret_v[p]=0
    print('.  ',contrib_pattern[p+1], '--> logit is', secret_v[p])
contribution_logpmf = model.parameters_to_contribution_logpmf(
    secret_v, contrib_shape, contrib_pattern)
contribution_pmf = torch.exp(contribution_logpmf)

# make initial guess
guess_patterns=[16,2,18,8,12,13,14,10,3]
initial_guess_v= torch.full((n_params,), -20.0)
print('\npatterns in initial guess')
print('.   [off] --> logit is 0.0')
print(f'.   [default] --> logit is {initial_guess_v[0].item()}')
for p in guess_patterns:
    initial_guess_v[p]=0
    print('.  ',contrib_pattern[p+1], '--> logit is', initial_guess_v[p])

# make initial guess
# rng = np.random.default_rng(0)
# initial_guess_v = torch.from_numpy(
#     rng.normal(
#         loc=0.0,
#         scale=1.0,
#         size=(n_params,)
# ))

# use ground truth to make observables
observation_logpmf = model.contribution_logpmf_to_observation_logpmf(contribution_logpmf, n)
observation_pmf = torch.exp(observation_logpmf)
print("\nobservation_pmf sum",torch.sum(observation_pmf.view(-1), dim=0))
print("observation_logpmf shape", observation_logpmf.shape)
print("observation_logpmf smallest value", observation_logpmf.min())

# check loss of the truth
loss = model.fit_loss_fn(secret_v,observation_pmf,contrib_shape,contrib_pattern,n)
print("loss of truth", loss.item())

# try to recover contribution_logpmf from observation_logpmf
print('\ntraining')
params, losses, secret_losses = model.fit_contribution_logpmf(
    observation_pmf,
    contrib_shape,
    true_contrib_logpmf=contribution_logpmf,
    initial_guess=initial_guess_v,
    # optmethod="adam",
    # n_iter=1000,
    # log_every=50,
    optmethod='lbfgs',
    n_iter=10,
    log_every=1,
)

guess_contribution_logpmf = model.parameters_to_contribution_logpmf(
    params, contrib_shape, contrib_pattern)
guess_contribution_pmf = torch.exp(guess_contribution_logpmf)
guess_observation_logpmf = model.contribution_logpmf_to_observation_logpmf(guess_contribution_logpmf, n)
guess_observation_pmf = torch.exp(guess_observation_logpmf)

# plot losses
plt.gcf().set_size_inches(2,2)
plt.plot(losses[len(losses)//2:])

# %%
# determine gradient at the final guess
params_clone = torch.nn.Parameter(params.clone())
guess_contribution_logpmf = model.parameters_to_contribution_logpmf(
    params_clone, contrib_shape, contrib_pattern)
guess_observation_logpmf = model.contribution_logpmf_to_observation_logpmf(guess_contribution_logpmf, n)
mask = ~torch.isneginf(guess_observation_logpmf)
loss = -torch.sum(observation_pmf[mask] * guess_observation_logpmf[mask])
loss.backward()
print(params_clone.grad)
print('max mag',params_clone.grad.abs().max())

# %% plot loss in convex combination between final guess for contrib_pmf truth about contrib_pmf
mask = observation_pmf != 0
convex_comb_losses = []
alphas = np.r_[0:1:20j]
for i,alpha in enumerate(alphas):
    # when alpha = 0, we use truth
    # when alpha = 1, we use guess
    convex_pmf= alpha * guess_contribution_pmf + (1-alpha) * contribution_pmf
    convex_observation_logpmf = model.contribution_logpmf_to_observation_logpmf(torch.log(convex_pmf), n)
    loss = -torch.sum(observation_pmf[mask] * convex_observation_logpmf[mask])
    convex_comb_losses.append(loss.item())
plt.gcf().set_size_inches(4,2)
plt.plot(alphas,convex_comb_losses)
plt.ylabel("negative\nlog likelihood")
plt.xlabel("one-dimensional family of\nguesses for waveform distribution")
plt.tight_layout()

# check whether our guess is at edge of the convex hull
print('least likely atom',params.min())

# say a bit about the most likely patterns
npp = np.argsort(params.detach().numpy())
print('most likely patterns')
print('.   [off] --> logit is 0.0')
for p in npp[-11:][::-1]:
    print('.  ',contrib_pattern[p+1], f'--> logit is {params[p].item():+.5f}',f"(p# = {p}, true logit is {secret_v[p].item()})")

# %%
# say a bit about the most likely patterns
npp = np.argsort(params.detach().numpy())
pvals = params.detach().numpy()[npp[-9:][::-1]]
pvals = np.exp(pvals); pvals = pvals/pvals.sum()
local_contribs = contrib_pattern.detach().numpy()[npp[-9:][::-1]+1]
for p, contrib in zip(pvals, local_contribs):
    print('   ',contrib, f'--> probability is {p:.3f}')


# %%
# plot loss in convex combination between final params and secret_v
mask = observation_pmf != 0
convex_comb_losses = []
alphas = np.r_[-.2:1.2:200j]
for i,alpha in enumerate(alphas):
    # when alpha = 0, we use truth
    # when alpha = 1, we use guess
    convex_params= alpha * params + (1-alpha) * secret_v
    loss = model.fit_loss_fn(convex_params, observation_pmf, contrib_shape, contrib_pattern, n)
    convex_comb_losses.append(loss.item())
plt.plot(alphas,convex_comb_losses)
plt.ylabel("negative log likelihood")
plt.xlabel("one-dimensional family of guesses for waveform distribution")

# mask = ~torch.isneginf(contribution_logpmf)
# print(torch.stack([
#     torch.exp(contribution_logpmf[mask].view(-1)),
#     torch.exp(guess_contribution_logpmf[mask].view(-1))
# ]).T)

# %%
plt.plot(
    torch.exp(contribution_logpmf[mask].view(-1)).detach().numpy(),
    torch.exp(guess_contribution_logpmf[mask].view(-1)).detach().numpy(),
    'x',
)
plt.xlabel("truth")
plt.ylabel("guess")
# %%
