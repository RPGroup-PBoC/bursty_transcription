#%%
# This script performs posterior inference for multiple operators
# at a single aTc conc (by default, Oid, O1, O2 at 1ng/mL).

import re #regex
import warnings
import dill
from multiprocessing import Pool
from git import Repo #for directory convenience

import numpy as np
from scipy.stats import nbinom as neg_binom
from mpmath import hyp2f1
from scipy.special import gammaln
import pandas as pd

import emcee

import srep


def log_like_repressed(params, data_rep):
    """Conv wrapper for log likelihood for 2-state promoter w/
    transcription bursts and repression.
    
    data_rep: a list of arrays, each of which is n x 2, of form
        data[:, 0] = SORTED unique mRNA counts
        data[:, 1] = frequency of each mRNA count

    Note the data pre-processing here, credit to Manuel for this observation:
    'The likelihood asks for unique mRNA entries and their corresponding 
    counts to speed up the process of computing the probability distribution. 
    Instead of computing the probability of 3 mRNAs n times, it computes it 
    once and multiplies the value by n.'
    This also reduces the size of the data arrays by ~10-fold,
    which reduces the time penalty of emcee's pickling
    to share the data within the multiprocessing Pool.
    """
    k_burst, mean_burst, kR_on, *k_offs = params
    params_local = np.array([k_burst, mean_burst, kR_on, 0])
    target = 0
    for i, expt in enumerate(data_rep):
        max_m = expt[0].max()
        params_local[-1] = k_offs[i]
        # note log_probs contains values for ALL m < max_m,
        # not just those in the data set...
        log_probs = srep.models.log_prob_m_bursty_rep(max_m, *params_local)
        # ...so extract just the ones we want & * by their occurence
        target += np.sum(expt[1] * log_probs[expt[0]])
    return target

def log_like_constitutive(params, data_uv5):
    k_burst = params[0]
    mean_burst = params[1]
    # change vars for scipy's goofy parametrization
    p = (1 + mean_burst)**(-1)
    return np.sum(data_uv5[1] * neg_binom._logpmf(data_uv5[0], k_burst, p))

def log_prior(params):
    k_burst, mean_burst, kR_on, koff_Oid, koff_O1, koff_O2 = params
    # remember these params are log_10 of the actual values!!
    if (0.62 < k_burst < 0.8 and 0.4 < mean_burst < 0.64 and
        0.1 < kR_on < 1.5 and -0.8 < koff_Oid < 0
        and -0.5 < koff_O1 < 0.3 and 0.1 < koff_O2 < 1.2 ):
        return 0.0
    return -np.inf

def log_posterior(params, data_uv5, data_rep):
    """check prior and then farm out data to the respective likelihoods."""
    # Boolean logic to sample in linear or in log scale
    # Credit to Manuel for this
    lp = log_prior(params)
    if lp == -np.inf:
        return -np.inf
    if log_sampling:
        params = 10**params
    return (lp + log_like_constitutive(params, data_uv5)
            + log_like_repressed(params, data_rep))

def condense_data(expts):
    # first load data using module util
    df_unreg, df_reg = srep.data_loader.load_FISH_by_promoter(("unreg", "reg"))
    df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]
    data_uv5 = np.unique(df_UV5['mRNA_cell'], return_counts=True)
    
    rep_data = []
    for expt in expts:
        df = df_reg[df_reg["experiment"] == expt]
        rep_data.append(
            np.unique(df['mRNA_cell'], return_counts=True)
            )
    return data_uv5, rep_data

#%%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

expts = ("Oid_1ngmL", "O1_1ngmL", "O2_1ngmL")
data_uv5, data_rep = condense_data(expts)
#%%
n_dim = 6
n_walkers = 18
n_burn = 1
n_steps = 100

# init walkers
p0 = np.zeros([n_walkers, n_dim])
# remember these are log_10 of actual params!!
log_sampling = True
var_labels = ["k_burst", "b", "kR_on", "koff_Oid", "koff_O1", "koff_O2"]
p0[:,0] = np.random.uniform(0.69,0.71, n_walkers) # k_burst
p0[:,1] = np.random.uniform(0.54,0.55, n_walkers) # mean_burst
p0[:,2] = np.random.uniform(0.5,0.6, n_walkers) # kR_on
p0[:,3] = np.random.uniform(-0.5,-0.4, n_walkers) # koff_Oid
p0[:,4] = np.random.uniform(-0.1,0.1, n_walkers) # koff_O1
p0[:,5] = np.random.uniform(0.3,0.5, n_walkers) # koff_O2
#%%
# run the sampler
with Pool(processes=7) as pool:
# instantiate sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior, args=(data_uv5, data_rep), pool=pool
    )
    pos, prob, state = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
    _ = sampler.run_mcmc(pos, n_steps, progress=True, thin_by=5);
del sampler.pool; # otherwise unpickling fails, even though pickling is fine

#%%
outfile = open(f"{repo_rootdir}/data/mcmc_samples/1ngmL_sampler.pkl", 'wb')
dill.dump(sampler, outfile)
outfile.close()

print(f"Autocorr time: {sampler.get_autocorr_time()}")

# %%
