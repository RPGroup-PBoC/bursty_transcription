#%%
import re
import dill
from git import Repo #for directory convenience

import numpy as np
import pandas as pd
import arviz as az

import matplotlib.pyplot as plt
import bebi103.viz

import srep

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

colors = srep.viz.color_selector('pboc')
srep.viz.plotting_style()

pklfile = open(
    f"{repo_rootdir}/data/mcmc_samples/many_pooled_test.pkl", 'rb'
    )
model, sampler = dill.load(pklfile)
pklfile.close()
inf_dat = az.convert_to_inference_data(
    sampler, var_names=model.var_labels
    )

#%%
fig, ax = plt.subplots(2, 2, figsize=(9,9))

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
    ax[0,1].plot(
        x_contour[0],
        y_contour[0],
        linewidth=0.6,
        linestyle=lstyle[op],
        color=lcolor[aTc]
        )
        
# off-screen markers to trick legend
for conc in lcolor.keys():
    ax[0,1].plot(
        (100,100),(100,100), '-',
        label=conc,
        color=lcolor[conc],
        # lw=data_lw
    )
for op in lstyle.keys():
    ax[0,1].plot(
        (-1,-1),(100,100), '-',
        label=op,
        color='k',
        linestyle=lstyle[op],
        # lw=data_lw
    )
ax[0,1].set_xlim(-0.7, 0.75)
ax[0,1].set_ylim(-0.6, 1.9)
ax[0,1].set_ylabel(r'$log_{10}(k_R^+/\gamma)$')
ax[0,1].set_xlabel(r'$log_{10}(k_R^-/\gamma)$')
ax[0,1].legend(loc='lower left', fontsize='small', ncol=2)

#%%
plt.savefig(f"{repo_rootdir}/figures/fig3/fig3_v3.pdf")

# %%
