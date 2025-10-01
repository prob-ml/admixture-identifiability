import matplotlib.pyplot as plt
import numpy as np
import pickle

import emalg as model

# %% load results
with open('cached/finite_data_sim.pkl', 'rb') as f:
    dct = pickle.load(f)

simresults = dct['results']
markers = ['o', 's', '^', 'D', 'x']
p0_options = dct['p0_options']
n_iter = dct['n_iter']
n_sample_options = dct['n_sample_options']

# %% print blending pmfs

p0_options = [0.95, 0.9, 0.8, 0.6]

from run_finite_data import n

plt.gcf().set_size_inches(3,2.7)
for i,p0 in enumerate(p0_options):
    plt.plot(
        model.n_sources_pmf(n,p0),label=f'$p_0$={p0}',
        marker=markers[i], markersize=5,
        alpha=.5,
    )
plt.xlabel('Number of sources\nin each observation')
plt.ylabel('Probability')
# plt.legend(framealpha=1.0)
plt.tight_layout()
plt.yticks([0, .25, .5, .75, 1])
plt.grid(True, alpha=0.3)
plt.savefig('plots/n_sources.png', dpi=300)


# %% plot loss at different n_samples

plt.figure()
plt.gcf().set_size_inches(5,2.7)
for j,y in enumerate(p0_options):
    for k in range(len(p0_options)):
        losses = []
        for n_samples in n_sample_options:
            losses.append(simresults[y,n_samples,k].secret_losses[-1])
        losses = np.array(losses)

        plt.plot(losses,
                    marker=markers[j],
                    markevery=1,
                    color=f"C{j}",
                    label=f"p0={y}" if k==0 else None,
                    alpha=.5,
        )

assert n_sample_options==[None,1000,100,10]

plt.xticks([0,1,2,3],['∞', '1000', '100', '10'])
# plt.gca().set_yscale('log')
plt.ylim(-.01,.8)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Number of samples')
plt.ylabel('Loss (TV distance)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('plots/finite_data_consequences.png', dpi=300)