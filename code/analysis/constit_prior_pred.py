import re #regex
from git import Repo #for directory convenience

import numpy as np
import pandas as pd

import cmdstanpy
import arviz as az

from srep.data_loader import load_FISH_by_promoter

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

# first load data using module util
df_unreg, _ = load_FISH_by_promoter()
# pull out one specific promoter for convenience for prior pred check & SBC
df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]

# ############################################################################
# PRIOR PREDICTIVE CHECK
# ############################################################################

sm_prior_pred = cmdstanpy.CmdStanModel(
    stan_file=f"{repo_rootdir}/code/stan/constit_prior_pred.stan",
    compile=True,)

# stan needs to know how many data points to generate,
# so pick a representative promoter
data_prior_pred = dict(
    N=len(df_UV5)
    )

prior_pred_samples = sm_prior_pred.sample(
    data=data_prior_pred,
    fixed_param=True,
    iter_sampling=1000,
)

# Convert to ArviZ InferenceData object
prior_pred_samples = az.from_cmdstanpy(
    posterior=prior_pred_samples, # this line b/c of arviz bug, PR#979
    prior=prior_pred_samples,
    prior_predictive=['mRNA_counts']
)

prior_pred_samples.to_netcdf(f"{repo_rootdir}/data/stan_samples/constit_prior_pred.nc")