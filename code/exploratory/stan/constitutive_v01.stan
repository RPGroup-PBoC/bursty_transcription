data {
  // Parameters for priors
  real log_alpha_loc;
  real log_alpha_scale;
  real log_b_loc;
  real log_b_scale;
  
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
  real beta_ = 1.0 / b;
}

model {
  // Priors
  alpha ~ lognormal(log_alpha_loc, log_alpha_scale);
  b ~ lognormal(log_b_loc, log_b_scale);

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