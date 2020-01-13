data {
  // Parameters for priors
  real log_alpha_loc;
  real log_alpha_scale;
  real log_b_loc;
  real log_b_scale;
  
  // Size of data set
  int<lower=1> N;
}


generated quantities {
  // Draw model params from priors: burst rate alpha & burst size b
  real alpha = lognormal_rng(log_alpha_loc, log_alpha_scale);
  real b = lognormal_rng(log_b_loc, log_b_scale);
  // Stan parametrizes negbinom w/ beta, not b
  real beta = 1.0 / b;

  // Generated data
  real mRNA_counts[N];

  // Draw samples
  for (i in 1:N) {
    mRNA_counts[i] = neg_binomial_rng(alpha, beta);
  }
}