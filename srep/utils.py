import re #regex
import copy #for deepcopy
from git import Repo #for directory convenience

import numpy as np
import pandas as pd
from numba import njit

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

def load_FISH_by_promoter(dfs_to_return):
    """
    Load Jones/Brewster 2014 FISH data and sort into 2 dataframes:
    one containing constitutive promoters, but only those for
    which we have energies from Brewster/Jones 2012, and another
    for all the repressed data, labeled by lacI operator and aTc dose.
    
    Input: tuple of strings indicating which data should be returned,
    allowed strings are "unreg", "reg", and "energies".
    Returns: list of requested dataframes, in order unreg, reg, and energies
    """

    # check requested output makes sense
    for item in dfs_to_return:
        if item not in ("unreg", "reg", "energies"):
            raise ValueError(
                'Unknown output requested, try "unreg", "reg", and/or "energies"')
    return_dfs = []

    # ignore regulated csv, this file has everything.
    # mRNA_cell is the data of interest, NOT spots_totals
    df_fish = pd.read_csv(f"{repo_rootdir}/data/jones_brewster_2014.csv")
    # remove spurious column that was index in csv
    del df_fish['Unnamed: 0']

    # binding energies for most of the constitutive promoters in the FISH dataset
    df_energies = pd.read_csv(f"{repo_rootdir}/data/brewster_jones_2012.csv")

    # parse all the different promoters/operators/conditions in the dataset
    raw_expt_labels = df_fish['experiment'].unique()
    raw_expt_labels.sort()

    if "unreg" in dfs_to_return:
        # grab labels for all constitutive promoters that we have energies for
        constitutive_labels = [label for label in raw_expt_labels if label in tuple(df_energies.Name)]
        # and then grab all FISH data corresponding to those promoters
        df_unreg = df_fish[df_fish['experiment'].isin(constitutive_labels)]
        return_dfs.append(df_unreg)

    if "reg" in dfs_to_return:
        # put all strings that start w/ 'O' in one list: these are regulated,
        # labeled by whichever lacI binding site was present (Oid, O1, etc)
        regulated_labels = [label for label in raw_expt_labels if re.match('^O', label)]
        df_reg = df_fish[df_fish['experiment'].isin(regulated_labels)]
        return_dfs.append(df_reg)

    if "energies" in dfs_to_return:
        return_dfs.append(df_energies)

    return return_dfs

def condense_data(expts):
    # first load data using module util
    df_unreg, df_reg = load_FISH_by_promoter(("unreg", "reg"))
    df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]
    data_uv5 = np.unique(df_UV5['mRNA_cell'], return_counts=True)
    
    rep_data = []
    for expt in expts:
        df = df_reg[df_reg["experiment"] == expt]
        rep_data.append(
            np.unique(df['mRNA_cell'], return_counts=True)
            )
    return data_uv5, rep_data

@njit
def uncondense_valuescounts(vc_tuple):
    # assume vc_tuple is (values, counts) which are arrays of
    # mRNA values and occurences in a dataset, respectively
    values, counts = vc_tuple
    list_of_lists = [values[i] * np.ones(counts[i]) for i,_ in enumerate(counts)]
    # return np.array(list(itertools.chain(*list_of_lists)))
    return np.array([cell for sublist in list_of_lists for cell in sublist])

    # return np.array([ [ x * y for x in range(n) ] for y in range(n) ])

@njit
def uncondense_ppc(ppc_samples):
    """
    This function exists to solve the following problem:
    My bursty_rep_rng generates samples, then returns
    np.unique(samples, return_counts=True).
    That's convenient mostly, and keeps .pkl size down, but
    srep.viz.predictive_ecdf, borrowed from JB, needs raw samples as input.
    This funtion regenerates the raw samples from the condensed form.
    """
    output_samples = np.empty((len(ppc_samples), ppc_samples[0][1].sum()))
    for i in range(len(ppc_samples)):
        output_samples[i,:] = uncondense_valuescounts(ppc_samples[i])
    return output_samples
