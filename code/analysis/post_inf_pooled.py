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

expts = ("O2_0p5ngmL", "O2_1ngmL", "O2_2ngmL", "O2_10ngmL")
data_uv5, data_rep = srep.utils.condense_data(expts)
var_labels = ["k_burst", "b", "kRon_0p5", "kRon_1", "kRon_2", "kRon_10", "kRoff"]
prior_mu_sig = {
    # remember these are log_10 of actual params!!
    "k_burst":(0.725, 0.025),
    "b":(0.55, 0.025),
    "kRon_0p5":(-0.45, 0.275),
    "kRon_1":(0.6, 0.25),
    "kRon_2":(1.15, 0.275),
    "kRon_10":(1.5, 0.375),
    "kRoff":(0.45, 0.275)
    }
expt_rates = {
    "O2_0p5ngmL":("kRon_0p5", "kRoff"),
    "O2_1ngmL":("kRon_1", "kRoff"),
    "O2_2ngmL":("kRon_2", "kRoff"),
    "O2_10ngmL":("kRon_10", "kRoff"),
    }
model = srep.models.pooledInferenceModel(
    expts=expts,
    var_labels=var_labels,
    expt_rates=expt_rates,
    prior_mu_sig=prior_mu_sig
    )

n_dim = len(var_labels)
n_walkers = 21
n_burn = 300
n_steps = 250

# init walkers
p0 = multinormal.rvs(
    mean=model.mu_prior,
    cov=model.cov_prior,
    size=n_walkers)

#%%
# run the sampler
with Pool(processes=25) as pool:
# instantiate sampler
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        srep.models.log_posterior,
        args=(data_uv5, data_rep, model),
        pool=pool
    )
    pos, prob, state = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
    _ = sampler.run_mcmc(pos, n_steps, progress=True, thin_by=10);
del sampler.pool; # otherwise unpickling fails, even though pickling is fine

#%%
outfile = open(f"{repo_rootdir}/data/mcmc_samples/O2_pooled_test.pkl", 'wb')
dill.dump(sampler, outfile)
outfile.close()

print(f"Autocorr time: {sampler.get_autocorr_time()}")

# %%
