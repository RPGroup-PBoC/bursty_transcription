from git import Repo #for directory convenience

import numpy as np

import cmdstanpy
import arviz as az

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo                 
repo_rootdir = repo.working_tree_dir

sm_gaussF = cmdstanpy.CmdStanModel(
    stan_file=f"{repo_rootdir}/code/stan/test_gaussF.stan",
    compile=True,)

# stan needs to know how many data points to generate,
# so pick a representative promoter
stan_data = dict(
    a=22.35,
    b=-17.3,
    c=12.5,
    z=-1.59,
    )

stan_output = sm_gaussF.sample(
    data=stan_data,
    fixed_param=True,
    iter_sampling=1,
)

# Convert to ArviZ InferenceData object
stan_output = az.from_cmdstanpy(
    stan_output, 
    posterior_predictive=["output"]
)

# uncomment for running live in ipython,
stan_output.posterior_predictive.output 