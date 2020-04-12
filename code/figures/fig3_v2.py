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
    f"{repo_rootdir}/data/mcmc_samples/1ngmL_sampler.pkl", 'rb'
    )
sampler_1ngmL = dill.load(pklfile)
pklfile.close()
expts_1ng = ("Oid_1ngmL", "O1_1ngmL", "O2_1ngmL")
var_labels_1ng = ["k_burst", "b", "kR_on", "koff_Oid", "koff_O1", "koff_O2"]
inf_dat_1ng = az.convert_to_inference_data(
    sampler_1ngmL, var_names=var_labels_1ng
    )

pklfile = open(
    f"{repo_rootdir}/data/mcmc_samples/O2_pooled_sampler.pkl", 'rb'
    )
sampler_O2 = dill.load(pklfile)
pklfile.close()
expts_O2 = ("O2_0p5ngmL", "O2_1ngmL", "O2_2ngmL", "O2_10ngmL")
var_labels_O2 = [
    "k_burst", "b", "kRon_0p5", "kRon_1", "kRon_2", "kRon_10", "kRoff"
    ]
inf_dat_O2 = az.convert_to_inference_data(
    sampler_O2, var_names=var_labels_O2
    )

#%%
fig, ax = plt.subplots(2, 2, figsize=(9,9))

for i, expt in enumerate(expts_1ng):
    kR_on_samples = inf_dat_1ng.posterior["kR_on"].values.flatten()
    kR_off_samples = inf_dat_1ng.posterior[var_labels_1ng[3+i]].values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        kR_off_samples, kR_on_samples, levels=0.95, smooth=0.025
    )
    ax[0,1].plot(
        x_contour[0], y_contour[0], '--', label=expts_1ng[i], linewidth=0.6,
        )

for i, expt in enumerate(expts_O2):
    kR_on_samples = inf_dat_O2.posterior[var_labels_O2[2+i]].values.flatten()
    kR_off_samples = inf_dat_O2.posterior["kRoff"].values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        kR_off_samples, kR_on_samples, levels=0.95, smooth=0.025
    )
    ax[0,1].plot(
        x_contour[0], y_contour[0], label=expts_O2[i], linewidth=0.6,
        )
ax[0,1].set_ylabel(r'$log_{10}(k_R^+/\gamma)$')
ax[0,1].set_xlabel(r'$log_{10}(k_R^-/\gamma)$')
ax[0,1].legend()#loc='upper right', ncol=2, fontsize='small')

plt.savefig(f"{repo_rootdir}/figures/fig3/fig3_v2.pdf")

# %%
