#include helper_functions.stan

data {
  real a;
  real b;
  real c;
  real z;
}

generated quantities {
  real output;
  output = gauss2F1(a, b, c, z);
}