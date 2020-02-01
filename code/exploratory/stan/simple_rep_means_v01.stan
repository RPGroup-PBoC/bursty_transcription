functions {
  real fold_change(real bohr) {
    return (1 + exp(-bohr))^(-1);
  }
}

data {
  // Parameters for priors
  real log_alpha_loc;
  real log_alpha_scale;
  real log_b_loc;
  real log_b_scale;
  real bohr_loc;
  real bohr_scale;
  
  // data set
  int<lower=1> N_cells_uv5;
  int<lower=1> N_cells_rep;
  int mRNA_counts_uv5[N_cells_uv5];
  int mRNA_counts_rep[N_cells_rep];

  // Do posterior predictive checks?
  int ppc;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> b;
  real bohr;
}

transformed parameters{
  real beta_ = 1.0 / b;
}

model {
  // Priors
  alpha ~ lognormal(log_alpha_loc, log_alpha_scale);
  b ~ lognormal(log_b_loc, log_b_scale);
  bohr ~ normal(bohr_loc, bohr_scale);

  // Likelihood
  mRNA_counts_uv5 ~ neg_binomial(alpha, beta_);
  mRNA_counts_rep ~ neg_binomial(alpha*fold_change(bohr), beta_);
}

generated quantities {
  // Post pred check
  int mRNA_counts_uv5_ppc[N_cells_uv5*ppc];
  int mRNA_counts_rep_ppc[N_cells_rep*ppc];

  if(ppc) {
    for (i in 1:N_cells_uv5) {
      mRNA_counts_uv5_ppc[i] = neg_binomial_rng(alpha, beta_);
    }
    for (i in 1:N_cells_rep) {
      mRNA_counts_rep_ppc[i] = neg_binomial_rng(alpha*fold_change(bohr), beta_);
    }
  }
}