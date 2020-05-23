#%%
# This script performs posterior inference for multiple op/aTc pair 
# datasets pooled together. UV5 must be included too to stabilize
# burst size & burst rate inference

import os
os.environ["OMP_NUM_THREADS"] = "1" # avoid probs w/ numpy autoparallel
import re #regex
import warnings
import dill
from multiprocessing import Pool
from git import Repo #for directory convenience

import numpy as np
from scipy.stats import multivariate_normal as multinormal
import pandas as pd

import emcee

import srep

#%%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

expts = (
    "O2_0p5ngmL", "O2_1ngmL", "O2_2ngmL", "O2_10ngmL",
    "Oid_1ngmL", "Oid_2ngmL", "O1_1ngmL", "O1_2ngmL", "O1_10ngmL"
    )
data_uv5, data_rep = srep.utils.condense_data(expts)
var_labels = [
    "k_burst", "b",
    "kRon_0p5", "kRon_1", "kRon_2", "kRon_10",
    "kRoff_Oid", "kRoff_O1", "kRoff_O2"]
prior_mu_sig = {
    # remember these are log_10 of actual params!!
    "k_burst":(0.725, 0.025),
    "b":(0.55, 0.025),
    "kRon_0p5":(-0.45, 0.3),
    "kRon_1":(0.6, 0.3),
    "kRon_2":(1.15, 0.3),
    "kRon_10":(1.5, 0.3),
    "kRoff_Oid":(-0.25, 0.3),
    "kRoff_O1":(0.1, 0.3),
    "kRoff_O2":(0.45, 0.3),
    }
expt_rates = {
    "O2_0p5ngmL":("kRon_0p5", "kRoff_O2"),
    "O2_1ngmL":("kRon_1", "kRoff_O2"),
    "O2_2ngmL":("kRon_2", "kRoff_O2"),
    "O2_10ngmL":("kRon_10", "kRoff_O2"),
    "Oid_1ngmL":("kRon_1", "kRoff_Oid"),
    "Oid_2ngmL":("kRon_2", "kRoff_Oid"),
    "O1_1ngmL":("kRon_1", "kRoff_O1"),
    "O1_2ngmL":("kRon_2", "kRoff_O1"),
    "O1_10ngmL":("kRon_10", "kRoff_O1"),
    }
model = srep.models.pooledInferenceModel(
    expts=expts,
    var_labels=var_labels,
    expt_rates=expt_rates,
    prior_mu_sig=prior_mu_sig
    )

n_dim = len(var_labels)
n_walkers = 35
n_burn = 300
n_steps = 200

# init walkers like prior but w/ narrower spread
p0 = multinormal.rvs(
    mean=model.mu_prior,
    cov=model.cov_prior/4,
    size=n_walkers)

#%%
# run the sampler
with Pool(processes=36) as pool:
# instantiate sampler
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        srep.models.log_posterior,
        args=(data_uv5, data_rep, model),
        pool=pool
    )
    pos, prob, state = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
    _ = sampler.run_mcmc(pos, n_steps, progress=True, thin_by=40);

# with posterior in hand, generate all the posterior predictive samples
ppc_uv5 = srep.models.post_pred_bursty_rep(
    sampler, n_pred=sum(data_uv5[1]),
    kon_idx='nbinom', koff_idx='nbinom'
    )
ppc_rep = []
for expt in expts:
    # first look up indices
    i = model.expts.index(expt)
    kon_idx, koff_idx = model.expt_idx_to_rate_idx(i)
    ppc_rep.append(
        srep.models.post_pred_bursty_rep(
            sampler, n_pred=sum(data_rep[i][1]),
            kon_idx=kon_idx, koff_idx=koff_idx))
del sampler.pool; # otherwise unpickling fails, even though pickling is fine

#%%
outfile = open(f"{repo_rootdir}/data/mcmc_samples/repression_pooled_expts.pkl", 'wb')
dill.dump((model, sampler, ppc_uv5, ppc_rep), outfile)
outfile.close()

print(f"Autocorr time: {sampler.get_autocorr_time()}")

# %%
