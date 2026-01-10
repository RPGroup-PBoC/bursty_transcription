"""
Extract parameter summaries (mode and credible intervals) from MCMC inferences
used in fig03 and fig04.

This script loads the inference results and computes:
- Mode (using median as approximation)
- 68% credible interval (16th-84th percentiles)
- 95% credible interval (2.5th-97.5th percentiles)

Outputs a tidy pandas DataFrame with all parameter summaries.
"""

import pickle
import dill
import re
from git import Repo

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az


def extract_constitutive_params(repo_rootdir):
    """
    Extract parameters from constitutive promoter inference (fig03).

    Returns DataFrame with columns:
    - promoter: promoter name
    - parameter: 'alpha' or 'b'
    - mode: median (approximation of mode)
    - ci_68_lower: 16th percentile
    - ci_68_upper: 84th percentile
    - ci_95_lower: 2.5th percentile
    - ci_95_upper: 97.5th percentile
    """
    # Load constitutive inference
    pklfile = open(
        f"{repo_rootdir}/data/mcmc_samples/constit_post_inf.pkl", 'rb')
    all_samples = pickle.load(pklfile)
    pklfile.close()

    results = []

    for promoter, inf_data in all_samples.items():
        # Extract alpha samples
        alpha_samples = inf_data.posterior.alpha.values.flatten()

        # Compute statistics
        mode_alpha = np.median(alpha_samples)
        ci_68_alpha = np.percentile(alpha_samples, [16, 84])
        ci_95_alpha = np.percentile(alpha_samples, [2.5, 97.5])

        results.append({
            'promoter': promoter,
            'parameter': 'alpha',
            'mode': mode_alpha,
            'ci_68_lower': ci_68_alpha[0],
            'ci_68_upper': ci_68_alpha[1],
            'ci_95_lower': ci_95_alpha[0],
            'ci_95_upper': ci_95_alpha[1],
            'model_type': 'constitutive'
        })

        # Extract b samples
        b_samples = inf_data.posterior.b.values.flatten()

        # Compute statistics
        mode_b = np.median(b_samples)
        ci_68_b = np.percentile(b_samples, [16, 84])
        ci_95_b = np.percentile(b_samples, [2.5, 97.5])

        results.append({
            'promoter': promoter,
            'parameter': 'b',
            'mode': mode_b,
            'ci_68_lower': ci_68_b[0],
            'ci_68_upper': ci_68_b[1],
            'ci_95_lower': ci_95_b[0],
            'ci_95_upper': ci_95_b[1],
            'model_type': 'constitutive'
        })

    return pd.DataFrame(results)


def extract_repression_params(repo_rootdir):
    """
    Extract parameters from pooled repression inference (fig04).

    Returns DataFrame with columns:
    - promoter: experiment name (e.g., 'O2_1ngmL')
    - parameter: parameter name (e.g., 'kR_on', 'kR_off', 'k_burst', 'b')
    - mode: median (approximation of mode)
    - ci_68_lower: 16th percentile
    - ci_68_upper: 84th percentile
    - ci_95_lower: 2.5th percentile
    - ci_95_upper: 97.5th percentile
    - note: parameters are in log_10 space
    """
    # Load repression inference
    pklfile = open(
        f"{repo_rootdir}/data/mcmc_samples/repression_pooled_expts.pkl", 'rb'
    )
    model, sampler, _, _ = dill.load(pklfile)
    pklfile.close()

    # Manually extract chain to avoid ArviZ trying to convert inhomogeneous args
    chain = sampler.get_chain()  # shape: (n_steps, n_walkers, n_dim)
    n_steps, n_walkers, n_dim = chain.shape

    # Reshape to (chain, draw, var) format for ArviZ
    posterior_dict = {}
    for i, var_name in enumerate(model.var_labels):
        posterior_dict[var_name] = (["chain", "draw"], chain[:, :, i].T)

    # Create InferenceData manually to avoid args conversion issues
    inf_dat = az.InferenceData(
        posterior=xr.Dataset(
            posterior_dict,
            coords={"chain": range(n_walkers), "draw": range(n_steps)}
        )
    )

    results = []

    # Extract global parameters (k_burst, b)
    for param_name in ['k_burst', 'b']:
        if param_name in inf_dat.posterior:
            samples = inf_dat.posterior[param_name].values.flatten()

            mode = np.median(samples)
            ci_68 = np.percentile(samples, [16, 84])
            ci_95 = np.percentile(samples, [2.5, 97.5])

            results.append({
                'promoter': 'global',
                'parameter': param_name,
                'mode': mode,
                'ci_68_lower': ci_68[0],
                'ci_68_upper': ci_68[1],
                'ci_95_lower': ci_95[0],
                'ci_95_upper': ci_95[1],
                'model_type': 'repression',
                'note': 'log_10 space'
            })

    # Extract experiment-specific parameters
    for expt in model.expts:
        op, aTc = re.split('_', expt)
        kRon_label, kRoff_label = model.expt_rates[expt]

        # Extract kR_on samples
        kR_on_samples = inf_dat.posterior[kRon_label].values.flatten()
        mode_kRon = np.median(kR_on_samples)
        ci_68_kRon = np.percentile(kR_on_samples, [16, 84])
        ci_95_kRon = np.percentile(kR_on_samples, [2.5, 97.5])

        results.append({
            'promoter': expt,
            'parameter': 'kR_on',
            'mode': mode_kRon,
            'ci_68_lower': ci_68_kRon[0],
            'ci_68_upper': ci_68_kRon[1],
            'ci_95_lower': ci_95_kRon[0],
            'ci_95_upper': ci_95_kRon[1],
            'model_type': 'repression',
            'note': 'log_10 space'
        })

        # Extract kR_off samples
        kR_off_samples = inf_dat.posterior[kRoff_label].values.flatten()
        mode_kRoff = np.median(kR_off_samples)
        ci_68_kRoff = np.percentile(kR_off_samples, [16, 84])
        ci_95_kRoff = np.percentile(kR_off_samples, [2.5, 97.5])

        results.append({
            'promoter': expt,
            'parameter': 'kR_off',
            'mode': mode_kRoff,
            'ci_68_lower': ci_68_kRoff[0],
            'ci_68_upper': ci_68_kRoff[1],
            'ci_95_lower': ci_95_kRoff[0],
            'ci_95_upper': ci_95_kRoff[1],
            'model_type': 'repression',
            'note': 'log_10 space'
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    repo = Repo("./", search_parent_directories=True)
    repo_rootdir = repo.working_tree_dir

    # Extract parameters from both models
    print("Extracting constitutive promoter parameters...")
    df_constit = extract_constitutive_params(repo_rootdir)

    print("Extracting repression parameters...")
    df_repress = extract_repression_params(repo_rootdir)

    # Combine into single DataFrame
    df_all = pd.concat([df_constit, df_repress], ignore_index=True)

    # Reorder columns for readability
    column_order = [
        'model_type', 'promoter', 'parameter', 'mode',
        'ci_68_lower', 'ci_68_upper', 'ci_95_lower', 'ci_95_upper'
    ]
    if 'note' in df_all.columns:
        column_order.append('note')

    df_all = df_all[column_order]

    # Sort by model type, then promoter, then parameter
    df_all = df_all.sort_values(
        ['model_type', 'promoter', 'parameter']).reset_index(drop=True)

    # Save to CSV
    output_file = f"{repo_rootdir}/data/parameter_summaries.csv"
    df_all.to_csv(output_file, index=False)
    print(f"\nParameter summaries saved to: {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"  Constitutive promoters: {len(df_constit)} parameter estimates")
    print(f"  Repression experiments: {len(df_repress)} parameter estimates")
    print(f"  Total: {len(df_all)} parameter estimates")

    # Display first few rows
    print("\nFirst 10 rows:")
    print(df_all.head(10).to_string())

    print("\nNote: Mode is approximated using the median.")
    print("For repression model, parameters are in log_10 space.")
