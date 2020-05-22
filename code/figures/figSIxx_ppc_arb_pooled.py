#%%
# This script computes ppc and generates plots for multiple
# pooled repressed datasets + a constitutive dataset
import re
import dill
from git import Repo #for directory convenience

import numpy as np
import pandas as pd

import emcee
import arviz as az

import matplotlib.pyplot as plt
import bebi103.viz

import srep

srep.viz.plotting_style()
pboc_colors = srep.viz.color_selector('pboc')

#%%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir


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

#%%

fig, axes = plt.subplots(2,2, figsize=(9,9))

# ##########################################################################
# PLOTTING PPC FOR ALL DATASETS FROM POOLED MODEL
# ##########################################################################

# organize all the options upfront
plotting_draws = 75
all_expts = (
    ("_"), # a blank b/c 1st index is skipped
    ("Oid_2ngmL", "Oid_1ngmL"),
    ("O1_1ngmL", "O1_2ngmL", "O1_10ngmL"),
    ("O2_0p5ngmL", "O2_1ngmL", "O2_2ngmL", "O2_10ngmL")
    )
ppc_colors = ('blue', 'purple', 'red', 'yellow')
data_colors = ('black', 'black', 'black', 'black')
# ppc_labels = ('0.5 ng/mL aTc', '1 ng/mL aTc', '2 ng/mL aTc', '10 ng/mL aTc')
uv5_colors = {'ppc':'green', 'data':'black'}
# convert labels to hex colors
ppc_colors = [pboc_colors[label] for label in ppc_colors]
data_colors = [pboc_colors[label] for label in data_colors]
ppc_alpha = 0.3
ppc_lw = 0.2
data_lw = 0.6
# then (nearly?) all the rest below will not need changing

for k, ax in enumerate(fig.axes):
    if k == 0:
        continue # upper left panel is handled separately
    ppc_labels = all_expts[k]

    # now we start plotting. UV5 1st
    for i in range(plotting_draws):
        ax.plot(
            *srep.viz.ecdf(ppc_uv5[i]),
            color=pboc_colors[uv5_colors['ppc']],
            alpha=ppc_alpha,
            lw=ppc_lw
            )
    # plot UV5 data later below to fix legend order

    # now loop over repressed datasets & plot ppc + observed data
    for j, expt in enumerate(ppc_labels):
        expt_idx = model.expts.index(expt)
        for i in range(plotting_draws):
            ax.plot(
                *srep.viz.ecdf(ppc_rep[expt_idx][i]),
                color=ppc_colors[j],
                alpha=ppc_alpha,
                lw=ppc_lw
                )
    # separate loop to plot data so it won't get buried by any ppc's
    for j, expt in enumerate(ppc_labels):
        i = model.expts.index(expt)
        ax.plot(*srep.viz.ecdf(data_rep[i]), color=data_colors[j], lw=data_lw)

    # plot UV5 last to get legend in order
    ax.plot(
        *srep.viz.ecdf(data_uv5),
        color=pboc_colors[uv5_colors['data']],
        lw=data_lw,
        label='Jones et al FISH data')

    # off-screen markers to trick legend into doing what I want
    for i, _ in enumerate(ppc_labels):
        ax.plot(
            (-10,-15),(0,0), '-',
            color=ppc_colors[i],
            label=ppc_labels[i],
            lw=data_lw
        )
    ax.plot(
        (-10,-15),(0,0), '-',
        color=pboc_colors[uv5_colors['ppc']],
        label='UV5',
        lw=data_lw
        )

    # finishing touches
    ax.set_xlabel('mRNA counts per cell')
    ax.set_ylabel('ECDF')
    ax.set_xlim(-2.5, 55)
    # ax.set_title('Posterior predictive samples vs experimental data')
    ax.legend(loc='lower right', fontsize='small')

plt.savefig(f"{repo_rootdir}/figures/figSIxx/ppc_many_pooled.pdf")

# %%
