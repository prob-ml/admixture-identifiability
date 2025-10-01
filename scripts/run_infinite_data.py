# %% imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import emalg as model
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

model.logger.setLevel("INFO")
# %% setup problem framework

# make the framework of the problem
n = 7 # n= T+1 = number of observations
L = 1 # a contribution is 2*L+1 wide
MAX_VAL = 3 # each contribution entry is in {0,...,MAX_VAL-1}
contrib_shape = (MAX_VAL,)*(2*L+1)
N=len(contrib_shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print(device)

rng = torch.Generator(device=device)

# %% run simulations with various p0 and n_samples

n_iter = 500

simresults = {}

p0_options = [0.95, 0.9, 0.8, 0.6]

for p0 in p0_options:
    rng.manual_seed(123)
    for i,p0guess in enumerate(p0_options):
        print(p0,i)
        simresults[p0, i]=model.run_simulation(
            n,
            contrib_shape,
            device,
            p0,
            None, # infinite samples
            rng,
            n_iter,
            p0_guess=p0guess,
        ).strip_giant_pmfs().to('cpu')

# %% save

os.makedirs('cached', exist_ok=True)
with open('cached/infinite_data_sim.pkl', 'wb') as f:
    pickle.dump({
        'n_iter': n_iter,
        'n_samples': None,
        'p0_options': p0_options,
        'results': simresults,
    }, f)

print("Saved pickle file size: {:.2f} KB".format(os.path.getsize('cached/infinite_data_sim.pkl') / 1024))