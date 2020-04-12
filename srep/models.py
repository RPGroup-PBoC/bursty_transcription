import warnings

import numpy as np
from scipy.stats import nbinom as neg_binom
from mpmath import hyp2f1
from scipy.special import gammaln


def log_prob_m_bursty_rep(max_m, k_burst, mean_burst, kR_on, kR_off):
    """
    log_prob_m_bursty_rep(max_m, k_burst, mean_burst, kR_on, kR_off)

    Computes log prob of observing m mRNA for all m s.t. 0 <= m <= m_max,
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

def bursty_rep_rng(params, n_samp, max_m=100):
    """
    bursty_rep_rng(params, n_samp, max_m=100)

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

def draw_nbinom_dataset(draw, n_samples):
    """
    Generate random samples from a negative binomial model. Assumes that
    draw is a sample from the model parameter space (probably posterior
    but could be prior) and that the 1st element of the draw is
    the burst rate and the 2nd element is the mean burst size.
    """
    pp_samples = neg_binom.rvs(draw[0], (1+draw[1])**(-1), size=n_samples)
    return np.unique(pp_samples, return_counts=True)

def post_pred_bursty_rep(
    sampler,
    n_pred,
    n_post=None,
    kon_ind=None,
    koff_ind=None
    ):
    """
    def post_pred_bursty_rep(
        sampler,
        n_rep,
        kon_ind=None,
        koff_ind=None,
        n_post=None
    )
    Takes as input an emcee EnsembleSampler instance (that has already
    sampled a posterior) and generates posterior predictive samples from it.
    
    - n_post is how many of the posterior draws to generate
    datasets for (chosen evenly spaced along the posterior flatchain)
    - n_pred is how many predictive samples to draw for
    each posterior sample
    - kon_ind & koff_ind flag which indices correspond to the
    rates to use in generating samples (e.g., if the full model
    contains several different rate parameters that we inferred from
    multiple pooled datasets, which do we use here?). Can also generate
    draws from constitutive nbinom dist by setting to 'nbinom'
    """
    if (kon_ind == None) or (koff_ind == None):
        raise ValueError("Must specify which chain indices have kR rates!")
    
    # get posterior draws...
    draws = sampler.get_chain(flat=True)
    # ...but thin if requested
    if n_post is not None:
        draw_slice = np.linspace(0, len(draws)-1, n_post).astype(int)
        draws = draws[draw_slice]

    # sampler runs in log space, rng is in linear space so convert first
    draws = 10**draws

    if (kon_ind == 'nbinom') and (koff_ind == 'nbinom'):
        return [draw_nbinom_dataset(draw, n_pred) for draw in draws]
    else:
        # generate draws from nbinom + simple repression model
        # first slice out only the rates for this expt
        # remember 0 is k_burst and 1 is mean_burst
        var_slice = [0, 1, kon_ind, koff_ind]
        draws = draws[:,var_slice]
        return [bursty_rep_rng(draw, n_pred) for draw in draws]