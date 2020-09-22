# %%
import enum
import re
import dill
from git import Repo #for directory convenience

import numpy as np
import pandas as pd

import emcee
import arviz as az

import matplotlib.pyplot as plt
import seaborn as sns
import bebi103.viz

import srep

srep.viz.plotting_style()
pboc_colors = srep.viz.color_selector('pboc')

# %%
fig, ax = plt.subplots(4, 3, figsize=(8.5, 10), sharex=False, sharey=False)

# # Modify tick font size
# for a in ax:
#     a.tick_params(axis="both", which="major", labelsize=8)

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

# Select PBoC color palette
colors = srep.viz.color_selector('pboc')
# Set PBoC plotting style
srep.viz.plotting_style()


# load in the pickled samples
pklfile = open(
    f"{repo_rootdir}/data/mcmc_samples/repression_pooled_expts.pkl", 'rb'
    )
model, sampler, ppc_uv5, ppc_rep = dill.load(pklfile)
pklfile.close()
inf_dat = az.convert_to_inference_data(
    sampler, var_names=model.var_labels
    )

data_uv5, data_rep = srep.utils.condense_data(model.expts)

n_dim = np.shape(model.var_labels)

# Define operators
op_array = ["Oid", "O1", "O2"]

# Define aTc concentrations
aTc_array = ["0p5ngmL", "1ngmL", "2ngmL", "10ngmL"]

# Set global colors for aTc concentrations
aTc_colors = ('blue', 'betancourt', 'green', 'orange')
aTc_col_dict = dict(zip(aTc_array , aTc_colors))

# organize all the options upfront
all_expts = (
    ("Oid_2ngmL", "Oid_1ngmL"),
    ("O1_1ngmL", "O1_2ngmL", "O1_10ngmL"),
    ("O2_0p5ngmL", "O2_1ngmL", "O2_2ngmL", "O2_10ngmL")
)

# Loop through operators concentrations
for op_idx, op in enumerate(op_array):
    # List experiments available for operator
    op_exp = all_expts[op_idx]

    # Loop through aTc concentrations
    for aTc_idx, aTc in enumerate(aTc_array):
        # Define aTc concentration color
        col = aTc_col_dict[aTc]
        color = srep.viz.bebi103_colors()[col]

        # Define experiment
        expt = f"{op}_{aTc}"

        # Add operator top of colums
        if aTc_idx == 0:
            label = f"operator {op}"
            ax[aTc_idx, op_idx].set_title(label, bbox=dict(facecolor="#ffedce"))
                # Add aTc concentration to right plots
        if op_idx == 2:
            # Generate twin axis
            axtwin = ax[aTc_idx, op_idx].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Fix label
            label = expt.split("_")[1]
            label = label.replace("ngmL", " ng/mL")
            label = label.replace("0p5", "0.5")

            # Set label
            axtwin.set_ylabel(
                f"[aTc] {label}",
                bbox=dict(facecolor="#ffedce"),
            )
            # Remove residual ticks from the original left axis
            ax[aTc_idx, op_idx].tick_params(color="w", width=0)
        
        # Add ylabel to left plots
        # if op_idx == 0:
        #     ax[aTc_idx, op_idx].set_ylabel("probability")

        # Check if experiment exists, if not, skip experiment
        if expt not in op_exp:
            ax[aTc_idx, op_idx].set_facecolor("#D3D3D3")
            ax[aTc_idx, op_idx].tick_params(axis='x', colors='white')
            ax[aTc_idx, op_idx].tick_params(axis='y', colors='white')
            continue

        # Find experiment index
        expt_idx = model.expts.index(expt)
        # Extract PPC samples and unpack them to raw format
        ppc_samples = srep.utils.uncondense_ppc(ppc_rep[expt_idx])
        # Define bins in histogram
        bins = np.arange(0, ppc_samples.max() + 1)
        # Initialize matrix to save histograms
        hist_mat = np.zeros([ppc_samples.shape[0], len(bins) - 1])
        # Loop through each ppc sample and compute histogram 
        for s_idx, s in enumerate(ppc_samples):
            hist_mat[s_idx] = np.histogram(s, bins=bins, density=True)[0]

        # Find percentiles to be plot
        lower_tile = np.percentile(hist_mat, 2.5, axis=0)
        upper_tile = np.percentile(hist_mat, 97.5, axis=0)
        mid_tile = np.percentile(hist_mat, 50, axis=0)

        # Extract data
        expt_data = srep.utils.uncondense_valuescounts(data_rep[expt_idx])
        # Compute histogram for data
        hist_data = np.histogram(expt_data, bins=bins, density=True)[0]

        # Plot predicted histogram with percentiles
        # 95% percentile 
        ax[aTc_idx, op_idx].fill_between(
            bins[:-1], 
            lower_tile, 
            upper_tile, 
            step="post", 
            edgecolor=color[0],
            color=color[0]
        )
        # median
        ax[aTc_idx, op_idx].step(
            bins[:-1],
            mid_tile,
            where="post",
            color=color[-1]
        )
        # add data on top
        ax[aTc_idx, op_idx].step(
            bins[:-1],
            hist_data,
            where="post",
            color="black",
            linewidth=1.25
        )
        # Set x-label
        ax[aTc_idx, op_idx].set_xlabel("mRNA / cell")
        ax[aTc_idx, op_idx].set_ylabel("probability")

        # Set axis limit
        upper_limit = np.where(hist_data > 5E-3)[0][-1]
        ax[aTc_idx, op_idx].set_xlim(0, upper_limit)
        
# Adjust spacing between plots
plt.subplots_adjust(hspace=0.3, wspace=0.4)


plt.savefig(
    f"{repo_rootdir}/figures/si/fig0X_histograms.pdf", bbox_inches='tight'
)
# %%
