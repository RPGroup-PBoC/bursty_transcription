import re #regex
import copy #for deepcopy
from git import Repo #for directory convenience

import numpy as np
import pandas as pd

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

def load_FISH_by_promoter():
    """
    Load Jones/Brewster 2014 FISH data and sort into 2 dataframes:
    one containing constitutive promoters, but only those for
    which we have energies from Brewster/Jones 2012, and another
    for all the repressed data, labeled by lacI operator and aTc dose.
    Returns df_unreg, df_reg.
    """
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

    # grab labels for all constitutive promoters that we have energies for
    constitutive_labels = [label for label in raw_expt_labels if label in tuple(df_energies.Name)]
    # and then grab all FISH data corresponding to those promoters
    df_unreg = df_fish[df_fish['experiment'].isin(constitutive_labels)]

    # put all strings that start w/ 'O' in one list: these are regulated,
    # labeled by whichever lacI binding site was present (Oid, O1, etc)
    regulated_labels = [label for label in raw_expt_labels if re.match('^O', label)]
    df_reg = df_fish[df_fish['experiment'].isin(regulated_labels)]

    return df_unreg, df_reg