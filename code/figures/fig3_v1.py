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
#%%
fig, ax = plt.subplots(2, 2, figsize=(9,9))

# ###########################################################################
# 95% HPD for identifiable expts promoters
# ###########################################################################
# # loop thru df, not all_samples keys, so we get deterministic order!
expt_labels = ("O2_0p5ngmL", "O2_1ngmL", "Oid_1ngmL", "O3_10ngmL")
var_labels = ["k_burst", "b", "kR_on", "kR_off"]
color_keys = ["green", "blue", "red", "purple"]
for i, expt in enumerate(expt_labels):
    # unpickle sampler, then convert to arviz InfDat obj
    pklfile = open(f"{repo_rootdir}/data/mcmc_samples/{expt}_sampler.pkl", 'rb')
    sampler = dill.load(pklfile)
    pklfile.close()

    inf_dat = az.convert_to_inference_data(
        sampler, var_names=var_labels
    )
    kR_on_samples = inf_dat.posterior.kR_on.values.flatten()
    kR_off_samples = inf_dat.posterior.kR_off.values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        kR_off_samples, kR_on_samples, levels=0.95, smooth=0.025
    )
    ax[0,1].plot(x_contour[0],
        y_contour[0],
        label=expt,
        linewidth=0.6,
        color=colors[color_keys[i]])
# ax[0,1].set_xlim(-2,2)
# ax[0,1].set_ylim(-2,2)
ax[0,1].set_ylabel(r'$log_{10}(k_R^+/\gamma)$')
ax[0,1].set_xlabel(r'$log_{10}(k_R^-/\gamma)$')
ax[0,1].legend()#loc='upper right', ncol=2, fontsize='small')

plt.savefig(f"{repo_rootdir}/figures/fig3/fig3b_v1.pdf")

# %%
