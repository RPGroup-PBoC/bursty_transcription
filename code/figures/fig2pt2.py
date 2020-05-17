#%%
import pickle
from git import Repo #for directory convenience

import numpy as np
import scipy.stats as st
import pandas as pd
import arviz as az

import matplotlib.pyplot as plt
import bebi103.viz

import srep.viz
from srep.utils import load_FISH_by_promoter

#%%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

colors = srep.viz.color_selector('constit')
srep.viz.plotting_style()

pklfile = open(f"{repo_rootdir}/data/mcmc_samples/constit_post_inf.pkl", 'rb')
all_samples = pickle.load(pklfile)
pklfile.close()

df_unreg, df_energies, = load_FISH_by_promoter(("unreg", "energies"))
df_energies.sort_values('Energy (kT)', inplace=True)

#%%
# ###########################################################################
# 95% HPD for all 18 constitutive promoters
# ###########################################################################
fig, ax = plt.subplots(1, 2, figsize=(8,4.5), sharey=True)
# setup colors, linestyles, and marker styles for all 18 promoters
cwheel_set = (
    'red', 'blue', 'purple',
    'green', 'brown', 'black',
    'yellow', 'magenta', 'white'
    )
cwheel = 2 * colors
# cwheel = 2 * cwheel_set
lwheel_nested = (9*('-',), 9*('--',))
lwheel = [item for sublist in lwheel_nested for item in sublist]
mwheel_nested = (9*('o',), 9*('^',))
mwheel = [item for sublist in mwheel_nested for item in sublist]

# # loop thru df, not all_samples keys, so we get deterministic order!
for i, promoter in enumerate(df_energies.Name):
    alpha_samples = all_samples[promoter].posterior.alpha.values.flatten()
    b_samples = all_samples[promoter].posterior.b.values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        b_samples, alpha_samples, levels=0.95, smooth=0.025
    )
    ax[0].loglog(
        x_contour[0],
        y_contour[0],
        lwheel[i], #linestyle
        label=promoter,
        linewidth=1.0,
        color=cwheel[i],
        )
ax[0].set_xlim(right=1.2e1)
ax[0].set_ylim(top=1e1)
ax[0].set_ylabel(r'$k_i$ (bursts per mRNA lifetime)')
ax[0].set_xlabel(r'$b$ (transcripts per burst)')

# ###########################################################################
# LOG(BURST RATE) vs ENERGIES from ENERGY MATRIX
# ###########################################################################
for i, promoter in enumerate(df_energies.Name):
    samples = all_samples[promoter].posterior.alpha.values.flatten()
    ptile_low, ptile_med, ptile_upr = np.percentile(samples, (2.5, 50, 97.5))
    err_lower = ptile_med - ptile_low
    err_upper = ptile_upr - ptile_med
    ax[1].errorbar(
        df_energies[df_energies.Name == promoter]["Energy (kT)"],
        ptile_med,
        yerr=np.array((err_lower, err_upper)).reshape((2,1)),
        # fmt='None',
        marker=mwheel[i],
        markersize=4.0,
        label=promoter,
        color=cwheel[i]
        )
# add a guideline for the eye for the predicted log(k_i) ~ - binding E
guide_x = np.linspace(-5.5,-2)
guide_y = np.exp(-guide_x)/50
ax[1].plot(guide_x, guide_y, 'k--')

ax[1].set_xlabel(r'Binding energy $(k_BT)$')
ax[1].set_yscale("log")
ax[1].legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))

plt.savefig(
    f"{repo_rootdir}/figures/fig2/fig2pt2.pdf", bbox_inches='tight'
    )

# %%
