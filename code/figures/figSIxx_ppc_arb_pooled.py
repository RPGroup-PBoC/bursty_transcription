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
    f"{repo_rootdir}/data/mcmc_samples/many_pooled_test.pkl", 'rb'
    )
model, sampler = dill.load(pklfile)
pklfile.close()
inf_dat = az.convert_to_inference_data(
    sampler, var_names=model.var_labels
    )

data_uv5, data_rep = srep.utils.condense_data(model.expts)

n_dim = np.shape(model.var_labels)

#%%
emcee_output = az.convert_to_inference_data(
    sampler, var_names=model.var_labels
    )
bokeh.io.show(
    bebi103.viz.corner(
        emcee_output, pars=model.var_labels[-5:], plot_ecdf=True
        )
    )

#%%

fig, axes = plt.subplots(2,2, figsize=(9,9))

# ##########################################################################
# PLOTTING ALL PAIRWISE POSTERIORS FROM POOLED DATA MODEL
# ##########################################################################
lstyle = {'O2':'-', 'O1':'-.', 'Oid':'--'}
lcolor = {'0p5ngmL':'green', '1ngmL':'purple', '2ngmL':'blue', '10ngmL':'red'}
for i, expt in enumerate(model.expts):
    # parse op/aTc values
    op, aTc = re.split('_', expt)
    # look up rates
    kRon_label, kRoff_label = model.expt_rates[expt]
    kR_on_samples = inf_dat.posterior[kRon_label].values.flatten()
    kR_off_samples = inf_dat.posterior[kRoff_label].values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        kR_off_samples, kR_on_samples, levels=0.95, smooth=0.025
    )
    axes[0,0].plot(
        x_contour[0],
        y_contour[0],
        linewidth=0.6,
        linestyle=lstyle[op],
        color=lcolor[aTc]
        )
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        kR_off_samples, kR_on_samples, levels=0.5, smooth=0.025
    )
    axes[0,0].plot(
        x_contour[0],
        y_contour[0],
        linewidth=0.6,
        linestyle=lstyle[op],
        color=lcolor[aTc]
        )
        
# off-screen markers to trick legend
for conc in lcolor.keys():
    axes[0,0].plot(
        (100,100),(100,100), '-',
        label=conc,
        color=lcolor[conc],
        # lw=data_lw
    )
for op in lstyle.keys():
    axes[0,0].plot(
        (-1,-1),(100,100), '-',
        label=op,
        color='k',
        linestyle=lstyle[op],
        # lw=data_lw
    )
axes[0,0].set_xlim(-0.7, 0.75)
axes[0,0].set_ylim(-0.6, 1.9)
axes[0,0].set_ylabel(r'$log_{10}(k_R^+/\gamma)$')
axes[0,0].set_xlabel(r'$log_{10}(k_R^-/\gamma)$')
axes[0,0].legend(loc='lower left', fontsize='small', ncol=2)

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
# ppc_labels = plotting_expts
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
    plotting_expts = all_expts[k]
    ppc_labels = plotting_expts

    # next, before plotting we must generate all the posterior predictive samples
    ppc_uv5 = srep.models.post_pred_bursty_rep(
        sampler, n_pred=sum(data_uv5[1]), n_post=plotting_draws,
        kon_idx='nbinom', koff_idx='nbinom'
        )
    ppc_rep = []
    for expt in plotting_expts:
        # first look up indices
        i = model.expts.index(expt)
        kon_idx, koff_idx = model.expt_idx_to_rate_idx(i)
        ppc_rep.append(
            srep.models.post_pred_bursty_rep(
                sampler, n_pred=sum(data_rep[i][1]), n_post=plotting_draws,
                kon_idx=kon_idx, koff_idx=koff_idx))

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

    # off-screen markers to trick legend into doing what I want
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
    # ax.set_title('Posterior predictive samples vs experimental data')
    ax.legend(loc='lower right', fontsize='small')

plt.savefig(f"{repo_rootdir}/figures/figSIxx/ppc_many_pooled.pdf")

# %%
