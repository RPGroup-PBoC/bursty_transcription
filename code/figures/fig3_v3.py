#%%
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
    f"{repo_rootdir}/data/mcmc_samples/O2_cross_pooled_test.pkl", 'rb'
    )
model, sampler = dill.load(pklfile)
pklfile.close()
inf_dat = az.convert_to_inference_data(
    sampler, var_names=model.var_labels
    )

#%%
fig, ax = plt.subplots(2, 2, figsize=(9,9))

for i, expt in enumerate(model.expts):
    # look up rates
    kRon_label, kRoff_label = model.expt_rates[expt]
    kR_on_samples = inf_dat.posterior[kRon_label].values.flatten()
    kR_off_samples = inf_dat.posterior[kRoff_label].values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        kR_off_samples, kR_on_samples, levels=0.95, smooth=0.025
    )
    ax[0,1].plot(
        x_contour[0], y_contour[0], label=expt, linewidth=0.6,
        )

ax[0,1].set_ylabel(r'$log_{10}(k_R^+/\gamma)$')
ax[0,1].set_xlabel(r'$log_{10}(k_R^-/\gamma)$')
ax[0,1].legend(loc='lower left', fontsize='small')#, ncol=2)

#%%
plt.savefig(f"{repo_rootdir}/figures/fig3/fig3_v3.pdf")

# %%
