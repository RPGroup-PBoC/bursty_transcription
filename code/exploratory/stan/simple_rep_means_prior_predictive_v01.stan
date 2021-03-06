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
  
  // Size of data set
  int<lower=1> N_cells_uv5;
  int<lower=1> N_cells_rep;
}

generated quantities {
  // Draw model params from priors: burst rate alpha & burst size b
  real alpha = lognormal_rng(log_alpha_loc, log_alpha_scale);
  real b = lognormal_rng(log_b_loc, log_b_scale);
  real bohr = normal_rng(bohr_loc, bohr_scale);

  // Stan parametrizes negbinom w/ beta, not b
  real beta = 1.0 / b;
  real alpha_rep = alpha * fold_change(bohr);

  // Generated data
  int mRNA_counts_uv5[N_cells_uv5];
  int mRNA_counts_rep[N_cells_rep];

  // Draw samples
  for (i in 1:N_cells_uv5) {
    mRNA_counts_uv5[i] = neg_binomial_rng(alpha, beta);
  }
  for (i in 1:N_cells_rep) {
    mRNA_counts_rep[i] = neg_binomial_rng(alpha_rep, beta);
  }
}