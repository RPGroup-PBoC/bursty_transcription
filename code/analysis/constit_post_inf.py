import re #regex
import pickle
from git import Repo #for directory convenience


import numpy as np
import pandas as pd

import cmdstanpy
import arviz as az

from bebi103.stan import disable_logging as be_quiet_stan
from bebi103.stan import check_all_diagnostics
from srep.data_loader import load_FISH_by_promoter

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

# first load data using module util
df_unreg, _ = load_FISH_by_promoter()
# pull out one specific promoter for convenience for prior pred check & SBC
df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]

sm = cmdstanpy.CmdStanModel(
    stan_file=f"{repo_rootdir}/code/stan/constit_post_inf.stan",
    compile=True,)

all_samples = {}
for gene in df_unreg['experiment'].unique():
    temp_df = df_unreg[df_unreg['experiment'] == gene]
    stan_data = dict(
        N=len(temp_df),
        mRNA_counts=temp_df["mRNA_cell"].values.astype(int),
        ppc=0 # if you produce ppc samples, the InferenceData obj is HUGE
    )
    with be_quiet_stan():
        posterior_samples = sm.sample(data=stan_data, cores=4)
    all_samples[gene] = az.from_cmdstanpy(
        posterior_samples, posterior_predictive=["mRNA_counts_ppc"]
    )
    print(f"For promoter {gene}...")
    check_all_diagnostics(all_samples[gene])

# pickle the samples. ~20 separate netcdfs, only for use together? No thanks
outfile = open(f"{repo_rootdir}/data/stan_samples/constit_post_inf.pkl", 'wb')
pickle.dump(all_samples, outfile)
outfile.close()