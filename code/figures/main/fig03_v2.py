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
#%%

# Import necessary data

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



# %%

# Generate composition figure to put all plot sections in a 
# single mpl file

# Initialize figure
fig = plt.figure(constrained_layout=False, figsize=(6, 6))

# Set gridspec for UV5 samples corner plot
gs_a = fig.add_gridspec(
    nrows=5, 
    ncols=5, 
    left=0.05, 
    right=0.45,
    top=1, 
    bottom=0.55,
    wspace=0,
    hspace=0
)

ax_a1 = fig.add_subplot(gs_a[0, :-1])
ax_a2 = fig.add_subplot(gs_a[1:, :-1])
ax_a3 = fig.add_subplot(gs_a[1:, -1])
# Set shared axes
ax_a1.get_shared_x_axes().join(ax_a2, ax_a1)
ax_a3.get_shared_y_axes().join(ax_a2, ax_a3)
# turn axis off for marginal plots
ax_a1.get_xaxis().set_visible(False)
ax_a1.get_yaxis().set_visible(False)
ax_a3.get_xaxis().set_visible(False)
ax_a3.get_yaxis().set_visible(False)



# Set girdspec for UV5 posterior predictive checks
gs_b = fig.add_gridspec(
    nrows=1,
    ncols=1,
    left=0.55,
    right=0.98,
    top=1,
    bottom=0.55,
)
# Add axis 
ax_b = fig.add_subplot(gs_b[0, 0])

# Set girdspec for unregulated promotres analysis
gs_cd = fig.add_gridspec(
    nrows=1,
    ncols=2,
    left=0.05,
    right=0.98,
    top=0.45,
    bottom=0.05,
    wspace=0.08
)
# Add axis 
ax_c = fig.add_subplot(gs_cd[0, 0])
ax_d = fig.add_subplot(gs_cd[0, 1])
# Set shared axes
ax_d.get_shared_y_axes().join(ax_c, ax_d)
# Turn off axis
ax_d.get_yaxis().set_visible(False)

# Label plots
plt.gcf().text(-0.1, 1, "(A)", fontsize=14)
plt.gcf().text(0.45, 1, "(B)", fontsize=14)
plt.gcf().text(-0.1, 0.48, "(C)", fontsize=14)
plt.gcf().text(0.48, 0.48, "(D)", fontsize=14)

# Plotting

# (A)
# Set UV5 parameters samples
alpha_samples = all_samples['UV5'].posterior.alpha.values.flatten()
b_samples = all_samples['UV5'].posterior.b.values.flatten()

ax_a2.plot(
    b_samples[::3],
    alpha_samples[::3],
    'k.',
    markersize=3,
    alpha=0.2,
    label='UV5 posterior samples')
ax_a2.set_ylabel(r'$k_i$ (bursts per mRNA lifetime)')
ax_a2.set_xlabel(r'$b$ (transcripts per burst)')

# Add grid lines
[ax_a1.axvline(x, color="white", linewidth=0.5) 
for x in [3.25, 3.5, 3.75]]

# Plot marginal distributions
ax_a1.hist(b_samples,
          50,
          density=1,
          histtype="step",
          color="black",
)
# Add gridlines
[ax_a3.axhline(x, color="white", linewidth=0.5) 
for x in [5, 5.5, 6]]

ax_a3.hist(alpha_samples,
          50,
          density=1,
          histtype="step",
          color="black",
          orientation="horizontal",
)
# set_xticks([3.25, 3.5, 3.75], minor=True)
# ax_a1.yaxis.grid(True, which='minor')

# (B)
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
    color='green',
    percentiles=ptiles,
    discrete=True,
    ax=ax_b,
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
    ax=ax_b,
    pred_label='Model 5 (N. Binom) PPC',
    data_label='UV5 data, Jones et. al.',
    data_color='orange',
    data_size=1 #linewidth
    )

ax_b.legend(loc='lower right', fontsize='small')
ax_b.set_xlabel('mRNA counts per cell')
ax_b.set_ylabel('ECDF')
ax_b.set_xlim(right=60)

# (C)
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
    ax_c.loglog(
        x_contour[0],
        y_contour[0],
        lwheel[i], #linestyle
        label=promoter,
        linewidth=1.0,
        color=cwheel[i],
        )
ax_c.set_xlim(right=1.2e1)
ax_c.set_ylim(top=1e1)
ax_c.set_ylabel(r'$k_i$ (bursts per mRNA lifetime)')
ax_c.set_xlabel(r'$b$ (transcripts per burst)')


# (D)
# Add gridlines
[ax_d.axhline(x, color="white", linewidth=0.5) 
for x in [0.1, 1]]

for i, promoter in enumerate(df_energies.Name):
    samples = all_samples[promoter].posterior.alpha.values.flatten()
    ptile_low, ptile_med, ptile_upr = np.percentile(samples, (2.5, 50, 97.5))
    err_lower = ptile_med - ptile_low
    err_upper = ptile_upr - ptile_med
    ax_d.errorbar(
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
ax_d.plot(guide_x, guide_y, 'k--', label='predicted \n scaling')
# Add text for scaling
ax_d.text(
    0.6, 
    0.6, 
    r"$\log(k_i) \sim \Delta\epsilon_r$",
    transform=ax_d.transAxes,
    rotation=-45,
)

ax_d.set_xlabel(r'Binding energy $(k_BT)$')
ax_d.set_yscale("log")

plt.savefig(
    f"{repo_rootdir}/figures/main/fig03.pdf", bbox_inches='tight'
    )