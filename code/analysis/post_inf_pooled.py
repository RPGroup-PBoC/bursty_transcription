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

expts = ("O2_0p5ngmL", "O2_1ngmL", "O2_2ngmL", "O2_10ngmL", "Oid_1ngmL", "O1_1ngmL")
data_uv5, data_rep = srep.utils.condense_data(expts)
var_labels = [
    "k_burst", "b",
    "kRon_0p5", "kRon_1", "kRon_2", "kRon_10",
    "kRoff_Oid", "kRoff_O1", "kRoff_O2"]
prior_mu_sig = {
    # remember these are log_10 of actual params!!
    "k_burst":(0.725, 0.025),
    "b":(0.55, 0.025),
    "kRon_0p5":(-0.45, 0.2),
    "kRon_1":(0.6, 0.25),
    "kRon_2":(1.15, 0.2),
    "kRon_10":(1.5, 0.2),
    "kRoff_Oid":(-0.25, 0.2),
    "kRoff_O1":(0.1, 0.2),
    "kRoff_O2":(0.45, 0.2)
    }
expt_rates = {
    "O2_0p5ngmL":("kRon_0p5", "kRoff_O2"),
    "O2_1ngmL":("kRon_1", "kRoff_O2"),
    "O2_2ngmL":("kRon_2", "kRoff_O2"),
    "O2_10ngmL":("kRon_10", "kRoff_O2"),
    "Oid_1ngmL":("kRon_1", "kRoff_Oid"),
    "O1_1ngmL":("kRon_1", "kRoff_O1"),
    }
model = srep.models.pooledInferenceModel(
    expts=expts,
    var_labels=var_labels,
    expt_rates=expt_rates,
    prior_mu_sig=prior_mu_sig
    )

n_dim = len(var_labels)
n_walkers = 35
n_burn = 500
n_steps = 150

# init walkers
p0 = multinormal.rvs(
    mean=model.mu_prior,
    cov=model.cov_prior,
    size=n_walkers)

#%%
# run the sampler
with Pool(processes=35) as pool:
# instantiate sampler
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        srep.models.log_posterior,
        args=(data_uv5, data_rep, model),
        pool=pool
    )
    pos, prob, state = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
    _ = sampler.run_mcmc(pos, n_steps, progress=True, thin_by=35);
del sampler.pool; # otherwise unpickling fails, even though pickling is fine

#%%
outfile = open(f"{repo_rootdir}/data/mcmc_samples/O2_cross_pooled_test.pkl", 'wb')
dill.dump((model, sampler), outfile)
outfile.close()

print(f"Autocorr time: {sampler.get_autocorr_time()}")

# %%
