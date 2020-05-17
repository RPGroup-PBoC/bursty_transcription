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

# ###########################################################################
# POSTERIOR FOR UV5
# ###########################################################################
# what was formerly a 2x2 subplots is now 2 separate 1x2 subplots
fig, ax = plt.subplots(1, 2, figsize=(8,4))
alpha_samples = all_samples['UV5'].posterior.alpha.values.flatten()
b_samples = all_samples['UV5'].posterior.b.values.flatten()
# thin samples before plotting, excessive # of glyphs otherwise
ax[0].plot(
    b_samples[::3],
    alpha_samples[::3],
    'k.',
    markersize=3,
    alpha=0.2,
    label='UV5 posterior samples')
ax[0].set_ylabel(r'$k_i$ (bursts per mRNA lifetime)')
ax[0].set_xlabel(r'$b$ (transcripts per burst)')
ax[0].legend()

# ###########################################################################
# POSTERIOR PREDICTIVE FOR UV5
# ###########################################################################
df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]
n_samples = (
    all_samples["UV5"].posterior_predictive.dims["chain"]
    * all_samples["UV5"].posterior_predictive.dims["draw"]
    )
ptiles = (95, 75, 50, 25)

# first compute params in analytical gamma-Poisson posterior.
# ignore prior b/c data completely(!) overwhelms it.
alpha = df_UV5["mRNA_cell"].sum()
beta = len(df_UV5)
# approx gamma posterior as normal b/c alpha is so huge
poiss_post_draws = st.norm.rvs(alpha/beta, np.sqrt(alpha)/beta, size=400)
poiss_ppc_draws = np.empty((len(df_UV5), len(poiss_post_draws)))
for i, draw in enumerate(poiss_post_draws):
    poiss_ppc_draws[:,i] = st.poisson.rvs(draw, size=len(df_UV5))
# now plot PPC from Poisson samples
srep.viz.predictive_ecdf(
    poiss_ppc_draws,
    color='purple',
    percentiles=ptiles,
    discrete=True,
    ax=ax[1],
    pred_label='Model 1 (Poisson) PPC',
    )

# next neg binom model + data overlaid
srep.viz.predictive_ecdf(
    all_samples['UV5'].posterior_predictive["mRNA_counts_ppc"].values.reshape(
        (n_samples, len(df_UV5))
    ),
    data=df_UV5["mRNA_cell"],
    percentiles=ptiles,
    discrete=True,
    # diff=True,
    ax=ax[1],
    pred_label='Model 5 (N. Binom) PPC',
    data_label='UV5 data, Jones et. al.',
    data_color='red',
    data_size=1 #linewidth
    )
# first plot UV5 data
# uv5_counts = df_unreg[df_unreg['experiment'] == "UV5"]["mRNA_cell"]
# x_data, y_data = bebi103.viz._ecdf_vals(uv5_counts, staircase=True)
# ax[1].plot(
#     x_data, y_data, label="UV5 data", color=colors[0]
#     )
# next add Poisson
# y_poiss = st.poisson.cdf(x_data, 18.72)
# ax[1].plot(x_data, y_poiss, label="Model 1 (Poisson) PPC", color='k')
# then add negbinom
# # UV5 MAP: alpha = 5.35, b = 3.5, note scipy's weird parametrization
# y_nbinom = st.nbinom.cdf(x_data, 5.35, 1/4.5)
# ax[1].plot(x_data, y_nbinom, '--', label="Model 5 (N Binom) PPC", color='k')
ax[1].legend(loc='lower right', fontsize='small')
ax[1].set_xlabel('mRNA counts per cell')
ax[1].set_ylabel('ECDF')
ax[1].set_xlim(right=60)

plt.savefig(f"{repo_rootdir}/figures/fig2/fig2pt1.pdf")

# %%
