#%%
# This script computes ppc and generates plots for multiple
# pooled repressed datasets + a constitutive dataset

import dill
from git import Repo #for directory convenience

import numpy as np
import pandas as pd

import emcee
import arviz as az

import matplotlib.pyplot as plt
import bokeh
import bokeh.io
import bebi103.viz

# from srep.data_loader import load_FISH_by_promoter
# from srep.viz import ecdf
# from srep.viz import plotting_style
import srep

bokeh.io.output_notebook()
srep.viz.plotting_style()
pboc_colors = srep.viz.color_selector('pboc')

#%%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir


# load in the pickled samples
pklfile = open(
    f"{repo_rootdir}/data/mcmc_samples/O2_cross_pooled_test.pkl", 'rb'
    )
model, sampler = dill.load(pklfile)
pklfile.close()

data_uv5, data_rep = srep.utils.condense_data(model.expts)

n_dim = np.shape(model.var_labels)

#%%
emcee_output = az.convert_to_inference_data(
    sampler, var_names=model.var_labels
    )
bokeh.io.show(
    bebi103.viz.corner(
        emcee_output, pars=model.var_labels[:6], plot_ecdf=True
        )
    )

#%%

# ppc plots: organize all the options upfront
plotting_draws = 75
plotting_expts = ("Oid_1ngmL", "O1_1ngmL", "O2_1ngmL")
ppc_colors = ('blue', 'purple', 'red', 'yellow')
data_colors = ('black', 'black', 'black', 'black')
ppc_labels = ('0.5 ng/mL aTc', '1 ng/mL aTc', '2 ng/mL aTc', '10 ng/mL aTc')
uv5_colors = {'ppc':'green', 'data':'black'}
# convert labels to hex colors
ppc_colors = [pboc_colors[label] for label in ppc_colors]
data_colors = [pboc_colors[label] for label in data_colors]
ppc_alpha = 0.3
ppc_lw = 0.2
data_lw = 0.6

# (nearly?) all the rest below will not need changing
ppc_uv5 = srep.models.post_pred_bursty_rep(
    sampler, n_pred=sum(data_uv5[1]), n_post=plotting_draws,
    kon_idx='nbinom', koff_idx='nbinom'
    )
ppc_rep = []
# for i in range(len(model.expts)):
for expt in plotting_expts:
    # first look up indices
    i = model.expts.index(expt)
    kon_idx, koff_idx = model.expt_idx_to_rate_idx(i)
    ppc_rep.append(
        srep.models.post_pred_bursty_rep(
            sampler, n_pred=sum(data_rep[i][1]), n_post=plotting_draws,
            kon_idx=kon_idx, koff_idx=koff_idx))

fig, ax = plt.subplots(1,1)

for i in range(plotting_draws):
    ax.plot(
        *srep.viz.ecdf(ppc_uv5[i]),
        color=pboc_colors[uv5_colors['ppc']],
        alpha=ppc_alpha,
        lw=ppc_lw
        )
# plot UV5 data later below to fix legend order

# now loop over repressed datasets & plot ppc + observed data
for j, _ in enumerate(plotting_expts):
    for i in range(plotting_draws):
        ax.plot(
            *srep.viz.ecdf(ppc_rep[j][i]),
            color=ppc_colors[j],
            alpha=ppc_alpha,
            lw=ppc_lw
            )
# separate loop to plot data so it won't get buried by any ppc's
for j, expt in enumerate(plotting_expts):
    i = model.expts.index(expt)
    ax.plot(*srep.viz.ecdf(data_rep[i]), color=data_colors[j], lw=data_lw)

# plot UV5 last to get legend in order
ax.plot(
    *srep.viz.ecdf(data_uv5),
    color=pboc_colors[uv5_colors['data']],
    lw=data_lw,
    label='Jones et al FISH data')

# off-screen markers to trick legend
for i, _ in enumerate(plotting_expts):
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
ax.set_title('Posterior predictive samples vs experimental data')
ax.legend()

#%%
plt.savefig(f"{repo_rootdir}/figures/figSIxx/ppc_cross_pooled.pdf")

# %%
