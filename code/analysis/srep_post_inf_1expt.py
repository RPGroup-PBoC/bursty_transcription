#%%
# This script performs posterior inference for a single simple repression
# dataset (i.e., a single operator at a single aTc concentration).
# Currently messy, needs reorg/refactor

import re #regex
import warnings
import pickle
from multiprocessing import Pool, cpu_count
from git import Repo #for directory convenience

import numpy as np
from scipy.stats import nbinom as neg_binom
from mpmath import hyp2f1
from scipy.special import gammaln
import pandas as pd

import emcee

# from srep.data_loader import load_FISH_by_promoter
# from srep.viz import ecdf
# from srep.viz import plotting_style
import srep


def log_like_repressed(params, data_rep):
    """Conv wrapper for log likelihood for 2-state promoter w/
    transcription bursts and repression.
    
    data : array-like. n x 2
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
    # mRNA, counts = data_rep[0], data_rep[1]
    # return np.sum(counts * log_p_m_bursty_rep(mRNA, *params))
    max_m = data_rep[0].max()
    # note log_probs contains values for ALL m < max_m,
    # not just those in the data set...
    log_probs = srep.models.log_prob_m_bursty_rep(max_m, *params)
    # ...so extract just the ones we want & * by their occurence
    return np.sum(data_rep[1] * log_probs[data_rep[0]])

def log_like_constitutive(params, data_uv5):
    k_burst, mean_burst, _, _ = params
    # k_burst, mean_burst = params
    #  mRNA, counts = data_constit[0], data_constit[1]
    # change vars for scipy's goofy parametrization
    p = (1 + mean_burst)**(-1)
    return np.sum(data_uv5[1] * neg_binom._logpmf(data_uv5[0], k_burst, p))

def log_prior(params):
    k_burst, mean_burst, kR_on, kR_off = params
    # k_burst, mean_burst = params
    # remember these params are log_10 of the actual values!!
    if (2 < k_burst < 8 and 2 < mean_burst < 7 and
        1e-2 < kR_on < 3e2 and 1e-2 < kR_off < 3e2):
        return 0.0
    return -np.inf

def log_posterior(params, data_uv5, data_rep):
    """check prior and then farm out data to the respective likelihoods."""
    # Boolean logic to sample in linear or in log scale
    # Credit to Manuel for this
    if log_sampling:
        params = 10**params
    lp = log_prior(params)
    if lp == -np.inf:
        return -np.inf
    return (lp + log_like_constitutive(params, data_uv5)
            + log_like_repressed(params, data_rep))

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

n_dim = 4
n_walkers = 40
n_burn = 100
n_steps = 100

# slice data for the sampler
data_uv5 = np.unique(df_UV5['mRNA_cell'], return_counts=True)
data_rep = np.unique(df_rep['mRNA_cell'], return_counts=True)

# for fake data testing
# params = (0.73, 0.54, 1.1, .5)
# params = np.power(10, params)
# data_uv5 = np.unique(
#     neg_binom.rvs(
#         params[0], (1 + params[1])**(-1), size=2500
#         ), return_counts=True
#         )
# data_rep = bursty_rep_rng(params, 1000)
# params = np.log10(params)

# init walkers
p0 = np.zeros([n_walkers, n_dim])
# remember these are log_10 of actual params!!
log_sampling = True
var_labels = ["k_burst", "b", "kR_on", "kR_off"]
p0[:,0] = np.random.uniform(0.65,0.75, n_walkers) # k_burst
p0[:,1] = np.random.uniform(0.45,0.65, n_walkers) # mean_burst
p0[:,2] = np.random.uniform(0,1, n_walkers) # kR_on
p0[:,3] = np.random.uniform(-1,1, n_walkers) # kR_off
#%%

# run the sampler
with Pool(processes=7) as pool:
    # instantiate sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior, pool=pool, args=(data_uv5, data_rep),
    )
    pos, prob, state = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
    _ = sampler.run_mcmc(pos, n_steps, progress=True, thin_by=20);

#%%
outfile = open(f"{repo_rootdir}/data/mcmc_samples/{op_aTc}_sampler.pkl", 'wb')
pickle.dump(sampler, outfile)
outfile.close()

print(f"Autocorr time: {sampler.get_autocorr_time()}")