# This script is a test of the recursion computation of 2F1 in the
# log likelihood for a single simple repression dataset (i.e., a single
# operator at a single aTc concentration) to verify its correctness against
# the slow version that uses mpmath for all 2F1 calculations.

import re #regex
import pickle
from multiprocessing import Pool, cpu_count
from git import Repo #for directory convenience

import numpy as np
from scipy.stats import nbinom as neg_binom
from mpmath import hyp2f1
from mpmath import ln as arbprec_ln
from scipy.special import gammaln
import pandas as pd

import emcee
import arviz as az

import matplotlib.pyplot as plt
import bokeh
import bokeh.io
import bebi103.viz

from srep.data_loader import load_FISH_by_promoter

def log_prob_m_bursty_rep(max_m, k_burst, mean_burst, kR_on, kR_off):
    """
    Compute log prob of observing m mRNA for all m s.t. 0 <= m <= m_max,
    given model parameters. The computation uses eq 76 in SI
    of Shahrezai & Swain 2008. The implementation uses mpmath to compute
    2F1 for max_m and max_m+1, then uses the minimal solution of the
    (+00) recursion relation run backwards to efficiently compute the rest.
    """

    # first compute alpha, beta, gamma, the parameters in the 2F1 gen fcn
    # recall the gen fcn is 2F1(alpha, beta, gamma, mean_burst*(z-1))
    rate_sum = k_burst + kR_off + kR_on
    sqrt_discrim = np.sqrt((rate_sum)**2 - 4*k_burst*kR_off)
    alpha = (rate_sum + sqrt_discrim) / 2.0
    beta = (rate_sum - sqrt_discrim) / 2.0
    gamma = kR_on + kR_off

    # now compute a, b, c, z, the parameters in the recursion relation,
    # in terms of alpha, beta, gamma, & mean_burst
    a = alpha
    b = gamma - beta
    c = 1.0 + alpha - beta 
    z = (1.0 + mean_burst)**(-1)

    # next set up recursive calculation of 2F1
    m = np.arange(0, max_m+1)
    # note 1+a-c is strictly > 0 so we needn't worry about signs in gammafcn
    lgamma_numer = gammaln(1+a-c+m)
    lgamma_denom = gammaln(1+a+b-c+m)
    twoFone = np.zeros_like(np.asfarray(m))
    # initialize recursion with starting values...
    twoFone[-1] = hyp2f1(a+m[-1],b,1+a+b-c+m[-1],1-z)
    twoFone[-2] = hyp2f1(a+m[-2],b,1+a+b-c+m[-2],1-z)
    # ...adjusted by gamma prefactors
    twoFone[-2:] *= np.exp(lgamma_numer[-2:] - lgamma_denom[-2:])
    # now run the recursion backwards (i.e., decreasing n)
    # python indexing rules make the indexing here horribly confusing,
    # haven't yet figured out a better/more transparent way
    for i, k in enumerate(np.arange(m[-1]-1, m[0], -1)):
        apk = a+k
        prefac_k = 2*apk - c + (b - apk)*z
        prefac_kplus1 = apk*(z-1)
        denom = c - apk
        twoFone[-(3+i)] = - (prefac_k * twoFone[-(2+i)]
                         + prefac_kplus1 * twoFone[-(1+i)]) / denom
    # when recursion is finished, cancel off prefactor of gammas
    logtwoFone = np.log(twoFone) + lgamma_denom - lgamma_numer

    # now compute prefactors in P(m), combine, done
    gamma_prefac = (gammaln(alpha+m) - gammaln(alpha)
                  + gammaln(beta+m) - gammaln(beta)
                  - gammaln(gamma+m) + gammaln(gamma)
                  - gammaln(1+m))
    bern_p = mean_burst / (1 + mean_burst)
    burst_prefac = m * np.log(bern_p) + alpha * np.log(1 - bern_p)
    return gamma_prefac + burst_prefac + logtwoFone

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
    log_probs = log_prob_m_bursty_rep(max_m, *params)
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
    if (0 < k_burst < 20 and 0 < mean_burst < 20 and
        0 < kR_on < 40 and 0 < kR_off < 20):
        return 0.0
    return -np.inf

def log_posterior(params, data_uv5, data_rep, log_sampling=False):
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

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

# first load data using module util
df_unreg, df_reg = load_FISH_by_promoter(("unreg", "reg"))
# pull out one specific promoter for convenience for prior pred check & SBC
df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]
df_O2_1ngml = df_reg[df_reg["experiment"] == "O2_1ngmL"]

n_dim = 4
n_walkers = 40
n_burn = 10
n_steps = 100

# slice data for the sampler
data_uv5 = np.unique(df_UV5['mRNA_cell'], return_counts=True)
data_rep = np.unique(df_O2_1ngml['mRNA_cell'], return_counts=True)
# init walkers
p0 = np.zeros([n_walkers, n_dim])
p0[:,0] = np.random.uniform(5,6, n_walkers) # k_burst
p0[:,1] = np.random.uniform(3,4, n_walkers) # mean_burst
p0[:,2] = np.random.uniform(0,10, n_walkers) # kR_on
p0[:,3] = np.random.uniform(0,10, n_walkers) # kR_off

# run the sampler
with Pool() as pool:
    # instantiate sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior, pool=pool, args=(data_uv5, data_rep),
    )
    print("starting burn-in...")
    pos, prob, state = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
    print("starting actual sampling...")
    _ = sampler.run_mcmc(pos, n_steps, progress=True, thin_by=40);

# print(f"Autocorr time: {sampler.get_autocorr_time()}")

fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["k_burst", "b", "kR_on", "kR_off"]
for i in range(n_dim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number");
plt.show()

emcee_output = az.from_emcee(
    sampler, var_names=['k_burst', 'b', 'kR_on', 'kR_off']
    )
bokeh.io.show(bebi103.viz.corner(emcee_output, plot_ecdf=True))