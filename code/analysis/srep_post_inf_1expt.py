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
from srep.viz import ecdf
from srep.viz import plotting_style

plotting_style()

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
def bursty_rep_rng(params, n_samp, max_m=100):
    """
    Generate random samples from the bursty rep model. Given a set
    of model parameters, it computes the CDF and then generates rngs
    using the inverse transform sampling method.
    
    Input:
    params - the model params k_burst, mean_burst_size, kR_on, kR_off
    n_samp - how many rng samples to generate
    max_m - a guess of the maximum m the CDF needs to account for.
        If P(m <= max_m) is not sufficiently close to 1, the algorithm will
        multiply max_m by 2 and try again. "Sufficiently close" is computed
        from the requested number of samples, but nevertheless don't put
        excessive confidence in the tails of the generated samples
        distribution. If you're generating more than 1e6 samples, then
        (1) why? and (2) you'll need to think harder about this than I have;
        for my purposes I don't need phenomenal coverage of the tails
        nor very large numbers of samples.
    """
    cdf = np.cumsum(np.exp(log_prob_m_bursty_rep(max_m, *params)))
    rtol = 1e-2 / n_samp
    # check that the range of mRNA we've covered contains ~all the prob mass
    while not np.isclose(cdf[-1], 1, rtol=rtol):
        max_m *= 2
        cdf = np.cumsum(np.exp(log_prob_m_bursty_rep(max_m, *params)))
        if max_m > 3e2:
            warnings.warn(
                "mRNA count limit hit, treat generated rngs with caution"
                )
            break
    # now draw the rngs by "inverting" the CDF
    samples = np.searchsorted(cdf, np.random.rand(n_samp))
    # and condense the data before returning
    return np.unique(samples, return_counts=True)
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
    ppc_rep = [bursty_rep_rng(draw, n_rep) for draw in draws]
    return ppc_uv5, ppc_rep
#%%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

# first load data using module util
df_unreg, df_reg = load_FISH_by_promoter(("unreg", "reg"))
# pull out one specific promoter for convenience for prior pred check & SBC
df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]
df_rep = df_reg[df_reg["experiment"] == "O2_1ngmL"]

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
p0[:,0] = np.random.uniform(0.65,0.75, n_walkers) # k_burst
p0[:,1] = np.random.uniform(0.45,0.65, n_walkers) # mean_burst
p0[:,2] = np.random.uniform(1,2, n_walkers) # kR_on
p0[:,3] = np.random.uniform(-1,2, n_walkers) # kR_off
#%%

# run the sampler
with Pool(processes=7) as pool:
    # instantiate sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior, pool=pool, args=(data_uv5, data_rep),
    )
    pos, prob, state = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
    _ = sampler.run_mcmc(pos, n_steps, progress=True, thin_by=20);

# print(f"Autocorr time: {sampler.get_autocorr_time()}")
#%%
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
for i in range(0, total_draws*.7, int(total_draws/plotting_draws)):
    ax.plot(*ecdf(ppc_uv5[i]), alpha=0.2, color='green', lw=0.2)
    ax.plot(*ecdf(ppc_rep[i]), alpha=0.2, color='blue', lw=0.2)
ax.plot(*ecdf(data_uv5), color='orange', lw=1)
ax.plot(*ecdf(data_rep), color='red', lw=1)

# why/how does the repressed fake data have higher counts than uv5?????

# %%
