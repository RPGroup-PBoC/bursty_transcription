import pickle
from git import Repo #for directory convenience

import numpy as np
import scipy.stats as st
import pandas as pd
import arviz as az

import matplotlib.pyplot as plt
import bebi103.viz

import srep.viz
from srep.data_loader import load_FISH_by_promoter


repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

colors = srep.viz.color_selector('constit')
srep.viz.plotting_style()

pklfile = open(f"{repo_rootdir}/data/stan_samples/constit_post_inf.pkl", 'rb')
all_samples = pickle.load(pklfile)
pklfile.close()

df_unreg, df_energies, = load_FISH_by_promoter(("unreg", "energies"))

fig, ax = plt.subplots(2, 2, figsize=(9,9))

# ###########################################################################
# POSTERIOR PREDICTIVE FOR UV5
# ###########################################################################
# JB's util only works w/ Bokeh, so plotting PPC in matplotlib will take
# some extra effort. Here's a quick and dirty data + MAP plot as placeholder
# first plot UV5 data
uv5_counts = df_unreg[df_unreg['experiment'] == "UV5"]["mRNA_cell"]
x_data, y_data = bebi103.viz._ecdf_vals(uv5_counts, staircase=True)
ax[0,1].plot(x_data, y_data, label="UV5", color=colors['UV5'])
# next add Poisson
y_poiss = st.poisson.cdf(x_data, 18.72)
ax[0,1].plot(x_data, y_poiss, label="Poisson", color='k')
# then add negbinom
# UV5 MAP: alpha = 5.35, b = 3.5, note scipy's weird parametrization
y_nbinom = st.nbinom.cdf(x_data, 5.35, 1/4.5)
ax[0,1].plot(x_data, y_nbinom, '--', label="Neg Binom", color='k')
ax[0,1].legend(loc='lower right', fontsize='small')
ax[0,1].set_xlabel('mRNA counts per cell')
ax[0,1].set_ylabel('CDF')


# ###########################################################################
# 95% HPD for all 18 constitutive promoters
# ###########################################################################
# # loop thru df, not all_samples keys, so we get deterministic order!
for promoter in df_energies.Name:
    alpha_samples = all_samples[promoter].posterior.alpha.values.flatten()
    b_samples = all_samples[promoter].posterior.b.values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        b_samples, alpha_samples, levels=0.95, smooth=0.025
    )
    ax[1,0].loglog(x_contour[0],
        y_contour[0],
        label=promoter,
        linewidth=0.6,
        color=colors[promoter])
ax[1,0].set_xlim(right=1.2e1)
ax[1,0].set_ylim(top=1e1)
ax[1,0].set_ylabel(r'$\alpha$ (bursts per mRNA lifetime)')
ax[1,0].set_xlabel(r'$b$ (transcripts per burst)')

# ###########################################################################
# LOG(BURST RATE) vs ENERGIES from ENERGY MATRIX
# ###########################################################################
for promoter in df_energies.Name:
    samples = all_samples[promoter].posterior.alpha.values.flatten()
    ptile_low, ptile_med, ptile_upr = np.percentile(samples, (2.5, 50, 97.5))
    err_lower = ptile_med - ptile_low
    err_upper = ptile_upr - ptile_med
    ax[1,1].errorbar(
        df_energies[df_energies.Name == promoter]["Energy (kT)"],
        ptile_med,
        yerr=np.array((err_lower, err_upper)).reshape((2,1)),
        fmt='.',
        label=promoter,
        color=colors[promoter])
ax[1,1].set_ylabel(r'$\alpha$ (bursts per mRNA lifetime)')
ax[1,1].set_xlabel(r'Binding energy $(k_BT)$')
ax[1,1].set_yscale("log")
ax[1,1].legend(loc='upper right', ncol=2, fontsize='small')

plt.savefig(f"{repo_rootdir}/figures/fig2/fig2_base.pdf")