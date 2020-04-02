data {
  // data set
  int<lower=1> N;
  int mRNA_counts[N];

  // Do posterior predictive checks?
  int ppc;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> b;
}

transformed parameters{
  // Stan parametrizes negbinom w/ beta, not b
  real beta_ = 1.0 / b;
}

model {
  // Priors
  alpha ~ lognormal(-0.5, 2.0);
  b ~ lognormal(0.5, 1.0);

  // Likelihood
  mRNA_counts ~ neg_binomial(alpha, beta_);
}

generated quantities {
  // Post pred check
  int mRNA_counts_ppc[N*ppc];

  if(ppc) {
    for (i in 1:N) {
      mRNA_counts_ppc[i] = neg_binomial_rng(alpha, beta_);
    }
  }
}