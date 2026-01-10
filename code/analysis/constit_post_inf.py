import re  # regex
import pickle
from git import Repo  # for directory convenience

import numpy as np
import pandas as pd

import cmdstanpy
import arviz as az

from bebi103.stan import disable_logging as be_quiet_stan
from bebi103.stan import check_all_diagnostics
from srep.utils import load_FISH_by_promoter

if __name__ == "__main__":
    repo = Repo("./", search_parent_directories=True)
    # repo_rootdir holds the absolute path to the top-level of our repo
    repo_rootdir = repo.working_tree_dir

    # first load data using module util
    df_unreg, = load_FISH_by_promoter(("unreg",))
    # pull out one specific promoter for convenience for prior pred check & SBC
    df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]

    sm = cmdstanpy.CmdStanModel(
        stan_file=f"{repo_rootdir}/code/stan/constit_post_inf.stan",
        compile=True,)

    all_samples = {}
    for gene in df_unreg['experiment'].unique():
        temp_df = df_unreg[df_unreg['experiment'] == gene]
        # Set ppc=1 to generate posterior predictive samples (needed for
        # plotting) Set ppc=0 if you want to save space and skip PPC
        stan_data = dict(
            N=len(temp_df),
            mRNA_counts=temp_df["mRNA_cell"].values.astype(int),
            ppc=1  # Generate 1 posterior predictive sample per data point
        )
        with be_quiet_stan():
            posterior_samples = sm.sample(data=stan_data, chains=6)
        # Only include posterior_predictive if ppc > 0 (otherwise variable doesn't exist)
        if stan_data["ppc"] > 0:
            all_samples[gene] = az.from_cmdstanpy(
                posterior_samples, posterior_predictive=["mRNA_counts_ppc"]
            )
        else:
            all_samples[gene] = az.from_cmdstanpy(posterior_samples)
        print(f"For promoter {gene}...")
        check_all_diagnostics(all_samples[gene])

    # pickle the samples. ~20 separate netcdfs, only for use together? No thanks
    outfile = open(
        f"{repo_rootdir}/data/mcmc_samples/constit_post_inf.pkl", 'wb')
    pickle.dump(all_samples, outfile)
    outfile.close()
