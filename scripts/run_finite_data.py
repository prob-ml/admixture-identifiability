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

# setup devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = torch.Generator(device=device)

# %% run simulations with various p0 and n_samples

p0_options = [0.95, 0.9, 0.8, 0.6]

n_iter = 100
n_sample_options = [None,1000,100,10]

simresults = {}

if __name__ == "__main__":
    for p0 in p0_options:
        for n_samples in n_sample_options:
            print(p0, n_samples)
            rng.manual_seed(123)
            for i,p0guess in enumerate(p0_options):
                simresults[p0, n_samples, i]=model.run_simulation(
                    n,
                    contrib_shape,
                    device,
                    p0,
                    n_samples,
                    rng,
                    n_iter,
                    p0_guess=p0guess,
                ).strip_giant_pmfs().to('cpu')


    with open('cached/finite_data_sim.pkl', 'wb') as f:
        pickle.dump({
            'n_iter': n_iter,
            'n_sample_options': n_sample_options,
            'p0_options': p0_options,
            'results': simresults,
        }, f)

    print("Saved pickle file size: {:.2f} KB".format(os.path.getsize('cached/finite_data_sim.pkl') / 1024))