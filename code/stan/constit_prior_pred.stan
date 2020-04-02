data {
  // Size of data set
  int<lower=1> N;
}


generated quantities {
  // Draw model params from priors: burst rate alpha & burst size b
  real alpha = lognormal_rng(-0.5, 2.0);
  real b = lognormal_rng(0.5, 1.0);
  // Stan parametrizes negbinom w/ beta, not b
  real beta = 1.0 / b;

  // Generated data
  real mRNA_counts[N];

  // Draw samples
  for (i in 1:N) {
    mRNA_counts[i] = neg_binomial_rng(alpha, beta);
  }
}