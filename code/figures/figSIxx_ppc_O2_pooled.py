#%%
# This script computes ppc and generates plots for multiple
# pooled repressed datasets + a constitutive dataset

import dill
from git import Repo #for directory convenience

import numpy as np
from scipy.stats import nbinom as neg_binom
from mpmath import hyp2f1
from scipy.special import gammaln
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

srep.viz.plotting_style()
pboc_colors = srep.viz.color_selector('pboc')

#%%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

expts = ("O2_0p5ngmL", "O2_1ngmL", "O2_2ngmL", "O2_10ngmL")
data_uv5, data_rep = srep.utils.condense_data(expts)

# load in the pickled samples
pklfile = open(f"{repo_rootdir}/data/mcmc_samples/O2_pooled_sampler.pkl", 'rb')
sampler = dill.load(pklfile)
pklfile.close()

n_dim = np.shape(sampler.get_chain())[-1]
# remember these are log_10 of actual params!!
var_labels = ["k_burst", "b", "kRon_0p5", "kRon_1", "kRon_2", "kRon_10", "kRoff"]

#%%
emcee_output = az.convert_to_inference_data(
    sampler, var_names=var_labels
    )
bokeh.io.show(
    bebi103.viz.corner(
        emcee_output, pars=var_labels[2:], plot_ecdf=True
        )
    )
#%%
# ppc plots
plotting_draws = 75
ppc_uv5 = srep.models.post_pred_bursty_rep(
    sampler, n_pred=sum(data_uv5[1]), n_post=plotting_draws,
    kon_ind='nbinom', koff_ind='nbinom'
    )
ppc_rep = []
for i in range(len(expts)):
    ppc_rep.append(
        srep.models.post_pred_bursty_rep(
            sampler, n_pred=sum(data_rep[i][1]), n_post=plotting_draws,
            kon_ind=2+i, koff_ind=-1))

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
for j in range(len(expts)):
    for i in range(plotting_draws):
        ax.plot(
            *srep.viz.ecdf(ppc_rep[j][i]),
            color=ppc_colors[j],
            alpha=ppc_alpha,
            lw=ppc_lw
            )
for j in range(len(expts)):
    ax.plot(*srep.viz.ecdf(data_rep[j]), color=data_colors[j], lw=data_lw)

# plot UV5 last to get legend in order
ax.plot(
    *srep.viz.ecdf(data_uv5),
    color=pboc_colors[uv5_colors['data']],
    lw=data_lw,
    label='Jones et al FISH data')

# off-screen markers to trick legend
for i in range(len(expts)):
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

plt.savefig(f"{repo_rootdir}/figures/figSIxx/ppc_O2_pooled.pdf")

# %%
