# %%
import pickle
from git import Repo  # for directory convenience

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


def predictive_histogram(
    samples,
    data=None,
    percentiles=[95],
    color="blue",
    data_color="black",
    data_size=1,
    bins=None,
    density=True,
    ax=None,
    pred_label=None,
    data_label=None,
):
    """
    Plot a predictive histogram from samples with credible regions.

    Parameters
    ----------
    samples : Numpy array, shape (n_samples, n)
        Predictive samples. n_samples is the number of posterior samples,
        and n is the number of posterior predictive samples for each.
    data : Numpy array, shape (n,), optional
        Observed data to overlay as histogram outline.
    percentiles : list, default [95, 75, 50, 25]
        Percentiles for credible intervals.
    color : str, default 'blue'
        Color scheme for credible regions.
    data_color : str, default 'black'
        Color for data histogram outline.
    data_size : float, default 1
        Linewidth for data histogram outline.
    bins : int or array, optional
        Histogram bins. If None, auto-determined.
    density : bool, default True
        If True, normalize histogram to density.
    ax : matplotlib axes, optional
        Axes to plot on.
    pred_label : str, optional
        Label for predictive distribution.
    data_label : str, optional
        Label for data.

    Returns
    -------
    ax : matplotlib axes
        Axes with plots.
    """
    if ax is None:
        ax = plt.gca()

    # Get color scheme
    colors = srep.viz.bebi103_colors()
    if color not in colors:
        color = "blue"
    color_palette = colors[color]

    # Determine bins
    if bins is None:
        # Use natural integer bins of size 1 for discrete counts
        data_range = (int(samples.min()), int(samples.max()))
        if data is not None:
            data_range = (min(data_range[0], int(data.min())), max(
                data_range[1], int(data.max())))
        # Bins are at integer boundaries: [0, 1), [1, 2), [2, 3), ...
        bins = np.arange(data_range[0], data_range[1] + 2)

    # Compute histograms for each sample
    n_samples = len(samples)
    histograms = []
    for i in range(n_samples):
        counts, _ = np.histogram(samples[i], bins=bins, density=density)
        histograms.append(counts)

    histograms = np.array(histograms)  # shape: (n_samples, n_bins)
    bin_edges = bins

    # Compute percentiles across samples for each bin
    percentiles = np.sort(percentiles)[::-1]
    ptiles = [50 - pt / 2 for pt in percentiles] + \
        [50] + [50 + pt / 2 for pt in percentiles[::-1]]

    # Plot credible regions (from outer to inner)
    for i, ptile in enumerate(ptiles[:-1]):
        lower = np.percentile(histograms, ptile, axis=0)
        upper = np.percentile(histograms, ptiles[i+1], axis=0)
        # Use lighter colors for outer regions
        alpha = 0.3 / (len(ptiles) - 1)
        # Create step plot coordinates for fill
        x_fill = np.repeat(bin_edges, 2)[1:-1]
        lower_fill = np.repeat(lower, 2)
        upper_fill = np.repeat(upper, 2)
        ax.fill_between(
            x_fill, lower_fill, upper_fill,
            color=color_palette[0],
            alpha=0.75,  # alpha * (i + 1),
            label=pred_label if i == 0 else None
        )

    # Plot median
    median_hist = np.percentile(histograms, 50, axis=0)
    ax.step(
        bin_edges[:-1], median_hist,
        where='post',
        color=color_palette[-1],
        linewidth=1,
        label=pred_label if len(ptiles) == 1 else None
    )

    # Plot data histogram outline
    if data is not None:
        data_counts, _ = np.histogram(data, bins=bins, density=density)
        ax.step(
            bin_edges[:-1], data_counts,
            where='post',
            color=data_color,
            linewidth=data_size,
            label=data_label
        )

    return ax


# %%
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

colors = srep.viz.color_selector('constit')
srep.viz.plotting_style()
# %%

# Import necessary data

repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

colors = srep.viz.color_selector('constit')
srep.viz.plotting_style()

pklfile = open(f"{repo_rootdir}/data/mcmc_samples/constit_post_inf.pkl", 'rb')
all_samples = pickle.load(pklfile)
pklfile.close()

df_unreg, df_energies, = load_FISH_by_promoter(("unreg", "energies"))
df_energies.sort_values('Energy (kT)', inplace=True)


# %%

# Generate composition figure to put all plot sections in a
# single mpl file

# Initialize figure
fig = plt.figure(constrained_layout=False, figsize=(6, 6))

# Set gridspec for UV5 samples corner plot
gs_a = fig.add_gridspec(
    nrows=5,
    ncols=5,
    left=0.05,
    right=0.45,
    top=1,
    bottom=0.55,
    wspace=0,
    hspace=0
)

ax_a1 = fig.add_subplot(gs_a[0, :-1])
ax_a2 = fig.add_subplot(gs_a[1:, :-1])
ax_a3 = fig.add_subplot(gs_a[1:, -1])
# Set shared axes
ax_a2.sharex(ax_a1)
ax_a3.sharey(ax_a2)
# turn axis off for marginal plots
ax_a1.get_xaxis().set_visible(False)
ax_a1.get_yaxis().set_visible(False)
ax_a3.get_xaxis().set_visible(False)
ax_a3.get_yaxis().set_visible(False)
# Set facecolor white
ax_a1.set_facecolor("white")
ax_a3.set_facecolor("white")

# Set girdspec for UV5 posterior predictive checks
gs_b = fig.add_gridspec(
    nrows=1,
    ncols=1,
    left=0.55,
    right=0.98,
    top=1,
    bottom=0.55,
)
# Add axis
ax_b = fig.add_subplot(gs_b[0, 0])

# Set girdspec for unregulated promotres analysis
gs_cd = fig.add_gridspec(
    nrows=1,
    ncols=2,
    left=0.05,
    right=0.98,
    top=0.45,
    bottom=0.05,
    wspace=0.08
)
# Add axis
ax_c = fig.add_subplot(gs_cd[0, 0])
ax_d = fig.add_subplot(gs_cd[0, 1])
# Set shared axes
ax_d.sharey(ax_c)
# Turn off axis
ax_d.get_yaxis().set_visible(False)

# Label plots
plt.gcf().text(-0.1, 1, "(A)", fontsize=14)
plt.gcf().text(0.45, 1, "(B)", fontsize=14)
plt.gcf().text(-0.1, 0.48, "(C)", fontsize=14)
plt.gcf().text(0.48, 0.48, "(D)", fontsize=14)

# Plotting

# (A)
# Set UV5 parameters samples
alpha_samples = all_samples['UV5'].posterior.alpha.values.flatten()
b_samples = all_samples['UV5'].posterior.b.values.flatten()
log_prob = all_samples['UV5'].sample_stats.lp.values.flatten()

ax_a2.scatter(
    b_samples[::3],
    alpha_samples[::3],
    c=log_prob[::3],
    cmap="gray_r",
    marker="o",
    s=5,
    alpha=0.5,
    label='UV5 posterior samples')
ax_a2.set_ylabel(r'$k_i$ (bursts per mRNA lifetime)')
ax_a2.set_xlabel(r'$b$ (transcripts per burst)')

# Plot marginal distributions
ax_a1.hist(b_samples,
           50,
           density=1,
           histtype="step",
           color="black",
           )
# Add gridlines
[ax_a3.axhline(x, color="white", linewidth=0.5)
 for x in [4.75, 5, 5.25, 5.5, 5.75, 6]]

ax_a3.hist(alpha_samples,
           50,
           density=1,
           histtype="step",
           color="black",
           orientation="horizontal",
           )

# (B) - Histogram version
df_UV5 = df_unreg[df_unreg["experiment"] == "UV5"]
n_samples = (
    all_samples["UV5"].posterior_predictive.dims["chain"]
    * all_samples["UV5"].posterior_predictive.dims["draw"]
)
ptiles = (95,)  # , 75, 50, 25)

# first compute params in analytical gamma-Poisson posterior.
# ignore prior b/c data completely(!) overwhelms it.
alpha = df_UV5["mRNA_cell"].sum()
beta = len(df_UV5)
# approx gamma posterior as normal b/c alpha is so huge
poiss_post_draws = st.norm.rvs(alpha/beta, np.sqrt(alpha)/beta, size=n_samples)
poiss_ppc_draws = np.empty((len(poiss_post_draws), len(df_UV5)))
for i, draw in enumerate(poiss_post_draws):
    poiss_ppc_draws[i, :] = st.poisson.rvs(draw, size=len(df_UV5))

# Plot Poisson model histogram
predictive_histogram(
    poiss_ppc_draws,
    color='green',
    percentiles=ptiles,
    ax=ax_b,
    pred_label='Model 1 (Poisson)',
)

# Plot negative binomial model histogram
neg_binom_ppc = all_samples['UV5'].posterior_predictive["mRNA_counts_ppc"].values.reshape(
    (n_samples, len(df_UV5))
)
predictive_histogram(
    neg_binom_ppc,
    data=df_UV5["mRNA_cell"],
    color='blue',
    percentiles=ptiles,
    ax=ax_b,
    pred_label='Model 5 (N. Binom)',
    data_label='UV5 data,\n Jones et. al. 2014',
    data_color='black',
    data_size=1,
)

# Add plots for composite legend
# 1. Extract colors
blues = srep.viz.bebi103_colors()["blue"]
greens = srep.viz.bebi103_colors()["green"]
# 2. Declare plots
poisson_1 = ax_b.plot([], [], color=greens[-1])
poisson_2 = ax_b.fill([], [], color=greens[0])
neg_binom_1 = ax_b.plot([], [], color=blues[-1])
neg_binom_2 = ax_b.fill([], [], color=blues[0])
uv5 = ax_b.plot([], [], color="black")

ax_b.legend(
    [(poisson_2[0], poisson_1[0]), (neg_binom_2[0], neg_binom_1[0]), (uv5[0],),],
    [
        "Model 1 (Poisson)",
        "Model 5 (Neg. Binom)",
        r"$lacUV5$ data," + "\n Jones et. al. 2014"
    ],
    fontsize=8,
    loc="upper right",
)

ax_b.set_xlabel('mRNA counts per cell')
ax_b.set_ylabel('Probability')
ax_b.set_xlim(right=60)


# (C)
# Set function to normalize colors to energy range

col_norm = matplotlib.colors.Normalize(
    vmin=df_energies["Energy (kT)"].min() - 2,
    vmax=df_energies["Energy (kT)"].max() + 2,
)
# Chose colormap for energies
cmap = matplotlib.cm.get_cmap('magma')

# # loop thru df, not all_samples keys, so we get deterministic order!
for i, promoter in enumerate(df_energies.Name):
    alpha_samples = all_samples[promoter].posterior.alpha.values.flatten()
    b_samples = all_samples[promoter].posterior.b.values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        b_samples, alpha_samples, levels=0.95, smooth=0.025
    )
    ax_c.loglog(
        x_contour[0],
        y_contour[0],
        label=promoter,
        linewidth=1.0,
        c=cmap(
            col_norm(df_energies[df_energies.Name == promoter]["Energy (kT)"].values[0])),
    )
    text = ax_c.annotate(
        f"{i + 1}",
        (np.mean(x_contour[0]),
         np.mean(y_contour),),
        fontsize=7,
        color="white"
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=1.5, foreground="black"),
         path_effects.Normal()]
    )

ax_c.set_xlim(right=1.2e1)
ax_c.set_ylim(top=1e1)
ax_c.set_ylabel(r'$k_i$ (bursts per mRNA lifetime)')
ax_c.set_xlabel(r'$b$ (transcripts per burst)')

# (D)
# Add gridlines
[ax_d.axhline(x, color="white", linewidth=0.5)
 for x in [0.1, 1]]

# initialize lsit to save y position
y_pos = list()
for i, promoter in enumerate(df_energies.Name):
    samples = all_samples[promoter].posterior.alpha.values.flatten()
    ptile_low, ptile_med, ptile_upr = np.percentile(samples, (2.5, 50, 97.5))
    err_lower = ptile_med - ptile_low
    err_upper = ptile_upr - ptile_med
    ax_d.errorbar(
        df_energies[df_energies.Name == promoter]["Energy (kT)"],
        ptile_med,
        yerr=np.array((err_lower, err_upper)).reshape((2, 1)),
        fmt="o",
        markersize=4.0,
        # label=promoter,
        c=cmap(
            col_norm(df_energies[df_energies.Name == promoter]["Energy (kT)"].values[0])),
    )
    # Append y position
    y_pos.append(ptile_med)

    text = ax_d.annotate(
        f"{i + 1}",
        (df_energies[df_energies.Name == promoter]["Energy (kT)"],
         ptile_med,),
        fontsize=7,
        color="white"
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=1.5, foreground="black"),
         path_effects.Normal()]
    )

# Add colorbar
im = ax_d.scatter(
    df_energies["Energy (kT)"].values,
    y_pos,
    c=df_energies["Energy (kT)"].values,
    vmin=df_energies["Energy (kT)"].min() - 2,
    vmax=df_energies["Energy (kT)"].max() + 2,
    s=35,
    cmap=cmap,
    linewidth=0.2,
)

im.set_visible(False)
cbar = fig.colorbar(im, ax=ax_d, pad=0.01)
cbar.set_label(r"$\Delta\epsilon_P \; (k_BT)$")

# add a guideline for the eye for the predicted log(k_i) ~ - binding E
guide_x = np.linspace(-7, -3.5)
guide_y = np.exp(-guide_x)/9e2
ax_d.plot(guide_x, guide_y, 'k--', label='predicted \n scaling')

ax_d.text(
    0.29,
    0.25,
    r"$\log k_i \sim - \Delta\epsilon_P$",
    transform=ax_d.transAxes,
    rotation=-45,
)

ax_d.set_xlabel(r'binding energy $(k_BT)$')
ax_d.set_yscale("log")

plt.savefig(
    f"{repo_rootdir}/figures/main/fig03hist.pdf", bbox_inches='tight'
)

# %%
