#%%
import pickle
from git import Repo #for directory convenience

import numpy as np
import scipy.stats as st
import pandas as pd
import arviz as az

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as path_effects
import seaborn as sns
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

# Initialize figure
fig = plt.figure(constrained_layout=False, figsize=(6, 12))

# Set gridspec
gs = fig.add_gridspec(
    nrows=18, 
    ncols=3, 
    # left=0.05, 
    # right=0.45,
    # top=1, 
    # bottom=0.55,
    wspace=0.15,
    hspace=0.15,
)

# Initialize list to save ax objects
ax = list()
# Loop through rows and columns
for i in range(6):
    for j in range(3):
        # Generate axis for ECDF
        ax_ecdf = fig.add_subplot(gs[(i * 3):(i * 3 + 2), j])
        ax_diff = fig.add_subplot(gs[(i * 3 + 2), j])
        # Join axis
        ax_diff.get_shared_x_axes().join(ax_ecdf, ax_diff)
        # Turn off axis labels
        if j != 0:
            ax_ecdf.get_yaxis().set_ticklabels([])
            ax_diff.get_yaxis().set_ticklabels([])

        ax_ecdf.get_xaxis().set_ticklabels([])
        if i != 5:
            ax_diff.get_xaxis().set_ticklabels([])
        # Set axis label
        if j == 0:
            ax_ecdf.set_ylabel("ECDF")
            ax_diff.set_ylabel("ECDF diff.")
        if i == 5:
            ax_diff.set_xlabel("mRNA/cell")
        # Join axis to first plot
        if (i != 0) | (j != 0):
            ax_ecdf.get_shared_x_axes().join(ax_ecdf, ax[0][0])
            ax_ecdf.get_shared_y_axes().join(ax_ecdf, ax[0][0])
            ax_diff.get_shared_x_axes().join(ax_diff, ax[0][1])
            ax_diff.get_shared_y_axes().join(ax_diff, ax[0][1])
        # Add to list
        ax.append([ax_ecdf, ax_diff])

# Align y axis label
fig.align_ylabels(ax)

# Find unique promoters
promoters = df_energies["Name"].unique()

# Loop through unique promoters and plot ECDF
for i, p in enumerate(promoters):
    # Plot ECDF
    srep.viz.ppc_ecdf_pair(
        all_samples[p],
        'mRNA_counts_ppc',
        df_unreg[df_unreg['experiment'] == p],
        ax=ax[i],
        data_color='black',
        color='betancourt',
        data_label=p
    )

    # Comupute mean mRNA
    mean_mRNA = np.round(df_unreg[
        df_unreg['experiment'] == p
        ]['mRNA_cell'].mean(), 1)
    ax[i][0].text(
        0.5, 
        0.5, 
        f"{p} \n $\\left\\langle m \\right\\rangle = ${mean_mRNA}",
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=8,
        transform=ax[i][0].transAxes,
    )

plt.savefig(
    f"{repo_rootdir}/figures/si/figS0X_ppc.pdf", bbox_inches='tight'
)
# %%


# %%
