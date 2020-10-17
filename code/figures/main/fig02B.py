# %%
import pickle
from git import Repo #for directory convenience

import numpy as np
import scipy.stats as st
import pandas as pd
import arviz as az

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as path_effects
import seaborn as sns
import bebi103.viz

import srep.viz
from srep.utils import load_FISH_by_promoter

# Set PBoC plotting style
srep.viz.plotting_style()
# %%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

# Import sm-FISH data along with promoter information
df_unreg, df_energies, = load_FISH_by_promoter(("unreg", "energies"))
df_energies.sort_values('Energy (kT)', inplace=True)

# %%
# Group FISH data by promoters
df_group = df_unreg.groupby("experiment")

# Initialize dataframe to save mean expression and Fano factor
names = ["promoter", "mean", "var", "fano", "energy_kT"]
df_summary = pd.DataFrame([], columns=names)

# Loop through promoters computing mean expression and Fano
for prom, data in df_group:
    # Compute mean expression
    mean_m = data["mRNA_cell"].mean()
    # Compute variance
    var_m = data["mRNA_cell"].var()
    # Compute Fano factor
    fano_m = var_m / mean_m
    # Extract binding energy
    energy = df_energies[df_energies["Name"] == prom]["Energy (kT)"].values[0]
    # Append to dataframe
    df_summary = df_summary.append(
        pd.Series([prom, mean_m, var_m, fano_m, energy], index=names),
        ignore_index=True
    )
# Sort dataframe by energy
df_summary.sort_values("energy_kT", inplace=True)

# %%
# Set function to normalize colors to energy range
col_norm = matplotlib.colors.Normalize(
    vmin=df_energies["Energy (kT)"].min() - 2,
    vmax=df_energies["Energy (kT)"].max() + 2,
)

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(6, 2))

# Set sections of plot for noise
# sub-Poissonian
ax.axhspan(0, 1, facecolor='#A9BFE3', alpha=0.5, zorder=-100)
ax.text(
    5,
    0.5,
    "sub-Poissonian",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
)
# super-Poissonian
ax.axhspan(1, 10, facecolor='#E8B19D', alpha=0.5, zorder=-100)
ax.text(
    5,
    2.5,
    "super-Poissonian",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
)
# Poissonian
ax.axhline(1, color="black", linestyle=":", linewidth=3)

# Define colormap
cmap = "magma"
# Plot mean vs Fano
im = ax.scatter(
    df_summary["mean"].values,
    df_summary["fano"],
    c=df_summary["energy_kT"].values, 
    vmin=df_energies["Energy (kT)"].min() - 2,
    vmax=df_energies["Energy (kT)"].max() + 2,
    s=35,
    cmap=cmap,
    linewidth=0.2,
)
# Add colorbar
cbar = fig.colorbar(im, ax=ax, pad=0.01)
# Set colorbar legend
cbar.set_label(r"$\Delta\epsilon_P \; (k_BT)$")

# Annotate points
for i, (row, data) in enumerate(df_summary.iterrows()):
    text = ax.annotate(
        f"{i + 1}",
        (data["mean"], data["fano"],),
        fontsize=7,
        color="white",
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=1.5, foreground="black"),
         path_effects.Normal()]
    )

# Label axis
ax.set_xlabel(r"$\left\langle \right.$mRNA$\left.\right\rangle$/cell")
ax.set_ylabel(r"Fano factor $\nu$")
# Define plotting range
# ax.set_xlim([-1, 21])
ax.set_ylim([0, 5.5])

ax.set_xscale("log")

plt.savefig(
    f"{repo_rootdir}/figures/main/fig02B.pdf", bbox_inches='tight'
)
# %%
