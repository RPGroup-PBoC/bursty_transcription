#%%
# This script performs posterior inference for a single simple repression
# dataset (i.e., a single operator at a single aTc concentration).
# Currently messy, needs reorg/refactor

import pickle
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

#%%
def post_pred_bursty_rep(sampler, n_uv5, n_rep):
    """
    Takes as input an emcee EnsembleSampler instance (that has already sampled a posterior) and generates posterior predictive samples from it.
    n_uv5 is how many predictive samples to draw for each posterior sample,
    and similarly for n_rep.
    """
    draws = sampler.get_chain(flat=True)
    if log_sampling == True:
        draws = 10**draws

    def draw_uv5_dataset(draw, n_uv5):
        pp_samples = neg_binom.rvs(draw[0], (1+draw[1])**(-1), size=n_uv5)
        return np.unique(pp_samples, return_counts=True)

    ppc_uv5 = [draw_uv5_dataset(draw, n_uv5) for draw in draws]
    ppc_rep = [srep.models.bursty_rep_rng(draw, n_rep) for draw in draws]
    return ppc_uv5, ppc_rep
#%%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

# first load data using module util
df_unreg, df_reg = srep.data_loader.load_FISH_by_promoter(("unreg", "reg"))
# pull out one specific promoter for convenience for prior pred check & SBC
df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]
op_aTc = "O2_1ngmL"
df_rep = df_reg[df_reg["experiment"] == op_aTc]

# slice data for the sampler
data_uv5 = np.unique(df_UV5['mRNA_cell'], return_counts=True)
data_rep = np.unique(df_rep['mRNA_cell'], return_counts=True)

# load in the pickled samples
pklfile = open(f"{repo_rootdir}/data/mcmc_samples/{op_aTc}_sampler.pkl", 'rb')
all_samples = pickle.load(pklfile)
pklfile.close()

n_dim = np.shape(sampler.get_chain())[-1]
# remember these are log_10 of actual params!!
log_sampling = True
var_labels = ["k_burst", "b", "kR_on", "kR_off"]

#%%
srep.viz.traceplot(sampler, var_labels)

#%%
emcee_output = az.from_emcee(
    sampler, var_names=['k_burst', 'b', 'kR_on', 'kR_off']
    )
bokeh.io.show(bebi103.viz.corner(emcee_output, plot_ecdf=True))
#%%
# ppc plots
ppc_uv5, ppc_rep = post_pred_bursty_rep(
    sampler, sum(data_uv5[1]), sum(data_rep[1]))
# how many post pred datasets should we plot?
plotting_draws = 50
total_draws = len(ppc_rep)

fig, ax = plt.subplots(1,1)
for i in range(0, total_draws, int(total_draws/plotting_draws)):
    ax.plot(*srep.viz.ecdf(ppc_uv5[i]), alpha=0.2, color='green', lw=0.2)
    ax.plot(*srep.viz.ecdf(ppc_rep[i]), alpha=0.2, color='blue', lw=0.2)
ax.plot(*srep.viz.ecdf(data_uv5), color='orange', lw=1)
ax.plot(*srep.viz.ecdf(data_rep), color='red', lw=1)
