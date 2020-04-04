functions {
  real _simple_gauss2F1(real a, real b, real c, real z) {
    /*
    Computes by simple brute forcing Taylor series
    until successive terms are below as spec'd tolerance.
    Thoroughly tested on Pearson's test cases and does at least as well
    vs their results (in thesis, not their paper).
    As expected seems to works well for 0 < z < 1/2, |a|,|b| < ~20 or 30,
    and |c| not too close to zero.
    */
    real tol = 1e-10;
    real running_total = 1.0;
    real term = 1.0;
    int n = 1;
    real local_a = a;
    real local_b = b;
    real local_c = c;

    while (abs(term) > tol) {
      term *= z * local_a * local_b / local_c / n;
      running_total += term;
      local_a += 1.0;
      local_b += 1.0;
      local_c += 1.0;
      n += 1;
    }
    return running_total;
  }

real gauss2F1(real a, real b, real c, real z) {
  /*
  First transform z to the interval [0,1/2],
  which will make the series computation much easier.
  */
  if (0 <= z && z <= 0.5) { //the easy base case
    return _simple_gauss2F1(a, b, c, z);
  }
  else if (z < -1) { // Pearson et al case 1
    real prefac1 = (1-z)^(-a) * tgamma(c) * tgamma(b-a)
                             / (tgamma(b) * tgamma(c-a));
    real prefac2 = (1-z)^(-b) * tgamma(c) * tgamma(a-b)
                             / (tgamma(a) * tgamma(c-b));
    return prefac1 * _simple_gauss2F1(a, c-b, a-b+1, 1/(1-z))
         + prefac2 * _simple_gauss2F1(b, c-a, b-a+1, 1/(1-z));
  }
  else if (z < 0) { //i.e. -1 < z < 0, Pearson et al case 2
    return (1-z)^(-a) * _simple_gauss2F1(a, c-b, c, z/(z-1));
  }
  else if (z <= 1) { //i.e. 1/2 < z <= 1, Pearson et al case 4
    real prefac1 = tgamma(c) * tgamma(c-a-b) / (tgamma(c-a) * tgamma(c-b));
    real prefac2 = (1-z)^(c-a-b) * tgamma(c) * tgamma(a+b-c)
                                / (tgamma(a) * tgamma(b));
    return prefac1 * _simple_gauss2F1(a, b, a+b-c+1, 1-z)
         + prefac2 * _simple_gauss2F1(c-a,c-b,c-a-b+1, 1-z);
  }
  /*
  else if (z <= 2) { // 1 < z <= 2, Pearson et al case 5
    
    // I think this case always blows up or goes complex?
    // - If c-a-b isn't an integer, then (1-z)^(c-a-b) is complex.
    // - If c-a-b is integer, either gamma(c-a-b) or gamma(a+b-c) hits a pole.
    // I'll leave it as a placeholder, but I don't think this is useful for me
    
    real prefac1 = z^(-a) * tgamma(c) * tgamma(c-a-b)
                         / (tgamma(c-a) * tgamma(c-b));
    real prefac2 = z^(a-c) * (1-z)^(c-a-b) * tgamma(c) * tgamma(a+b-c)
                                          / (tgamma(a) * tgamma(b));
    return prefac1 * _simple_gauss2F1(a, a-c+1, a+b-c+1, 1-1/z)
         + prefac2 * _simple_gauss2F1(c-a, 1-a, c-a-b+1, 1-1/z);
  }
  else if (z > 2) { Pearson et al case 6
    // If either a or b is non integer, (-z)^(-a) and/or (-z)^(-b)
    // is complex, sim to case 5.
  }
  */
  /*
  If we haven't caught z yet, then z>1. It appears that if z > 1, then 2F1
  is real only at isolated points. Not worth coding, b/c I think that means
  something went fundamentally wrong with my inference, corresponding to
  negative burst sizes?? So I want to throw an error, but cmdstanpy
  seems to silence it somewhere? or maybe it only shows up in logs?
  */
  reject("2F1 is complex almost everywhere for z > 1; found z=", z);
}
}