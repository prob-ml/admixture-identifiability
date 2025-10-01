import matplotlib.pyplot as plt
import numpy as np
import pickle

# %% load results
with open('cached/infinite_data_sim.pkl', 'rb') as f:
    dct = pickle.load(f)

simresults = dct['results']
markers = ['o', 's', '^', 'D', 'x']
p0_options = dct['p0_options']
n_iter = dct['n_iter']

# %% plot loss convergence

upto = 30

plt.gcf().set_size_inches(5,2.7)
for j,y in enumerate(p0_options):
    for k in range(len(p0_options)):
        plt.plot(np.array(simresults[y,k].secret_losses)[:upto],
                    marker=markers[j],
                    markevery=3,
                    color=f"C{j}",
                    label=f"$p_0={y}$" if k==0 else None,
                    alpha=.5,
        )
plt.ylim(-.01,.2)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Iteration')
plt.ylabel('Loss (TV distance)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('plots/infinite_data_loss_convergence.png', dpi=300)

# %% plot p0 convergence

upto=10

plt.figure()
plt.gcf().set_size_inches(3,2.7)
for j,y in enumerate(p0_options):
    for k in range(len(p0_options)):
        plt.plot(np.array(simresults[y,k].p0_guesses)[:upto],
                    marker=markers[j],
                    markevery=1,
                    color=f"C{j}",
                    label=f"p0={y}" if k==0 else None,
                    alpha=.5,
        )

plt.xlabel('Iteration')
plt.ylabel('$p_0$ estimate')
# plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('plots/infinite_data_p0_convergence.png', dpi=300)
