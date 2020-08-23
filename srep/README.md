## `srep` software module

The analysis perform in this manuscript required personalized `python`
functions. In order to keep all computational routines organized we generated a
small package with different sub-modules. This package needs to be locally
installed in order to reproduce the analysis. The different modules contained in
this package are:
- **`models.py`**: Statistical models used when defining the log probability
  used to sample the posterior distribution with the
  [`emcee`](https://emcee.readthedocs.io/en/stable/) *Affine Invariant MCMC
  samples*.
- **`utils.py`**: Series of convenient computational routines used for
  repetitive routines such as loading data or generating tidy dataframes out of
  MCMC samples.
- **`viz.py`**: Visualization functions. From the plotting style used in all
  plots in the manuscript, to functions to generate ECDF plots.