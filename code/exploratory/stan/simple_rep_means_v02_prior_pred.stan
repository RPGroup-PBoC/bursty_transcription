/* Handling repressed data sets following GC & JB's pattern
for hierarchical models, e.g., 2018 bebi103 tutorial8b. */

functions {
  real fold_change(real log_R, real op_E) {
    // 15.34 is log(N_NS)
    return (1 + exp(log_R - 15.34 - op_E))^(-1);
  }
}

data {
  // data set
  int<lower=1> N; // num of repressed cells across all conditions
  int<lower=1> N_uv5; // num of UV5 cells 
  // int mRNA[N]; // # of counts of mRNA for each cell
  // int mRNA_uv5[N_uv5];

  // condition counters: for each single cell measurement, what was aTc & op?
  int<lower=1> aTc_idx[N];
  int<lower=1> op_idx[N];

  // Do posterior predictive checks?
  // int ppc;
}

transformed data {
  // num of conditions
  int<lower=1> n_aTc = 4;
  int<lower=1> n_op = 4;
  // Parameters for priors
  real log_alpha_loc = 1.75;
  real<lower=0> log_alpha_scale = 0.1;
  real log_b_loc = 1.25;
  real<lower=0> log_b_scale = 0.1;
  real op_E_loc[n_op] = {-15.3, -13.7, -9.6, -17.7};
  real<lower=0> op_E_scale = 0.1;
  real log_R_loc[n_aTc] = {-2.3, 0.7, 2.3, 3.9};
  real<lower=0> log_R_scale = 2.0;
}

generated quantities {
  real log_R[n_aTc];
  real op_E[n_op];
  real fc;
    // Generated data
  real mRNA[N];
  real mRNA_uv5[N_uv5];
  // Draw burst rate alpha & burst size b from prior
  real alpha = lognormal_rng(log_alpha_loc, log_alpha_scale);
  real b = lognormal_rng(log_b_loc, log_b_scale);

  // transform parameters: Stan parametrizes negbinom w/ beta, not b
  real beta_ = 1.0 / b;
  // Finish draws from prior
  for (i in 1:n_aTc) {
    log_R[i] = normal_rng(log_R_loc[i], log_R_scale);
  }
  for (i in 1:n_op) {
    op_E[i] = normal_rng(op_E_loc[i], op_E_scale);
  }

  // Draw samples
  for (i in 1:N_uv5) {
    mRNA_uv5[i] = neg_binomial_rng(alpha, beta_);
  }
  for (i in 1:N) {
    fc = fold_change(log_R[aTc_idx[i]], op_E[op_idx[i]]);
    mRNA[i] = neg_binomial_rng(alpha*fc, beta_);
  }
}