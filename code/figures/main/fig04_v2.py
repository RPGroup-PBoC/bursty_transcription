# %%
import re
import dill
from git import Repo #for directory convenience

import numpy as np
import pandas as pd

import emcee
import arviz as az

import matplotlib.pyplot as plt
import seaborn as sns
import bebi103.viz

import srep

srep.viz.plotting_style()
pboc_colors = srep.viz.color_selector('pboc')

# %%
# Initialize figure
fig = plt.figure(constrained_layout=False, figsize=(6, 5))
# Set gridspec for top row
gs_top = fig.add_gridspec(
    nrows=7, 
    ncols=2, 
    left=0.05, 
    right=0.98,
    top=1, 
    bottom=0.55,
    wspace=.55,
    hspace=0.1,
)

ax_0 = fig.add_subplot(gs_top[:, 0])
ax_1 = fig.add_subplot(gs_top[:3, 1])
ax_2 = fig.add_subplot(gs_top[4:, 1])

# Set gridspec for bottom row
gs_bottom = fig.add_gridspec(
    nrows=1, 
    ncols=3, 
    left=0.03, 
    right=0.98,
    top=0.4, 
    bottom=0,
    wspace=0.05,
    hspace=0.2
)

ax_3 = fig.add_subplot(gs_bottom[0])
ax_4 = fig.add_subplot(gs_bottom[1])
ax_5 = fig.add_subplot(gs_bottom[2])

# Set shared axes
ax_4.get_shared_y_axes().join(ax_3, ax_4)
ax_5.get_shared_y_axes().join(ax_3, ax_5)

# Turn off axis
ax_4.yaxis.set_ticklabels([])
ax_5.yaxis.set_ticklabels([])

# Pack all ax together
ax = [ax_0, ax_1, ax_2, ax_3, ax_4, ax_5]
# Modify tick font size
for a in ax:
    a.tick_params(axis="both", which="major", labelsize=8)

# Set global colors for aTc concentrations
aTc_colors = ('blue', 'betancourt', 'green', 'orange')
aTc_col_dict = dict(zip(["0p5ngmL","1ngmL", "2ngmL", "10ngmL"], aTc_colors))


# TOP ROW
repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

colors = srep.viz.color_selector('pboc')
srep.viz.plotting_style()

pklfile = open(
    f"{repo_rootdir}/data/mcmc_samples/many_pooled_test.pkl", 'rb'
    )
model, sampler = dill.load(pklfile)
pklfile.close()
inf_dat = az.convert_to_inference_data(
    sampler, var_names=model.var_labels
    )

lstyle = {'O2':'-', 'O1':'-.', 'Oid':'--'}

# Initialize dataframe to save centers for all experiments
center = pd.DataFrame([], columns=["op", "aTc", "x", "y"])
# Initialize list to save central position
for i, expt in enumerate(model.expts):
    # parse op/aTc values
    op, aTc = re.split('_', expt)
    # look up rates
    kRon_label, kRoff_label = model.expt_rates[expt]
    kR_on_samples = inf_dat.posterior[kRon_label].values.flatten()
    kR_off_samples = inf_dat.posterior[kRoff_label].values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        kR_off_samples, kR_on_samples, levels=(0.50, 0.95), smooth=0.025
    )
    contour_opts = {
        'linewidth':0.6,
        'linestyle':lstyle[op], 
        'color':srep.viz.bebi103_colors()[aTc_col_dict[aTc]][-1]
        }
    ax[0].plot(x_contour[0], y_contour[0], **contour_opts)
    ax[0].plot(x_contour[1], y_contour[1], **contour_opts)
    # Find center
    y_center = np.mean(kR_on_samples)
    x_center = np.mean(kR_off_samples)
    # Save in dataframe
    center = center.append(
        pd.Series(
            [op, aTc, x_center, y_center],
            index=["op", "aTc", "x", "y"]
        ),
        ignore_index=True
    )

# Add operator text 
df_group = center.groupby("op")
for group, data in df_group:
    ax[0].text(
        data["x"].mean(),
        2,
        group,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=8,
        bbox=dict(facecolor="#FFEDC0", edgecolor=None, pad=.5)
    )
# Add concentration text 
df_group = center.groupby("aTc")
for group, data in df_group:
    # Clean label name
    label = group.replace("ngmL", " ng/mL")
    label = label.replace("0p5", "0.5")
    ax[0].text(
        0.77,
        data["y"].mean(),
        label,
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=8,
        bbox=dict(facecolor="#FFEDC0", edgecolor=None, pad=.5)
    )

ax[0].set_xlim(-0.7, 0.75)
ax[0].set_ylim(-0.6, 1.9)
ax[0].set_ylabel(r'$log_{10}(k_R^+/\gamma)$', fontsize=8, labelpad=0)
ax[0].set_xlabel(r'$log_{10}(k_R^-/\gamma)$', fontsize=8)

# ##########################################################################
# COMPARING k_off RATIOS TO BINDING E DIFFERENCES
# ##########################################################################
# Define colors for markesr
col_markers = sns.color_palette("colorblind", n_colors=4)

hg_rp_ep = (-17., -15.3, -13.9) # Oid, O1, O2
hg_rp_err = 0.2
mwc_soc_ep = (-17.7, -15.2, -13.6)
mwc_soc_err = 0.15

hg_diff_ep = np.diff(np.array(hg_rp_ep))
soc_diff_ep = np.diff(np.array(mwc_soc_ep))

oid_samples = inf_dat.posterior['kRoff_Oid'].values.flatten()
o1_samples = inf_dat.posterior['kRoff_O1'].values.flatten()
o2_samples = inf_dat.posterior['kRoff_O2'].values.flatten()

# remember the samples are of log_10 of rates
o1_oid = np.percentile(np.log(10) * (o1_samples - oid_samples), (5, 50, 95))
o2_o1  = np.percentile(np.log(10) * (o2_samples - o1_samples), (5, 50, 95))
# compute errbars. so very close to Gaussian, don't bother w/ asymm
o1_oid_err = np.mean(np.diff(o1_oid))
o2_o1_err  = np.mean(np.diff(o2_o1))

# Initialize label counter
counter = 1
ax[1].errorbar(o2_o1[1], hg_diff_ep[1], xerr=o2_o1_err, yerr=hg_rp_err,
                fmt='o', capsize=0, label='O2-O1, HG & RP 2011', 
                color=col_markers[0])
ax[1].annotate(
    f"{counter}",
    (o2_o1[1],
    hg_diff_ep[1]),
    fontsize=7,
    ha='center', 
    va='center',
    color="white"
)
counter += 1
ax[1].errorbar(o1_oid[1], hg_diff_ep[0], xerr=o1_oid_err, yerr=hg_rp_err,
                fmt='o', capsize=0, label='O1-Oid, HG & RP 2011',
                color=col_markers[1])
ax[1].annotate(
    f"{counter}",
    (o1_oid[1], hg_diff_ep[0]),
    fontsize=7,
    ha='center', 
    va='center',
    color="white"
)
counter += 1
ax[1].errorbar(o2_o1[1], soc_diff_ep[1], xerr=o2_o1_err, yerr=mwc_soc_err,
                fmt='o', capsize=0, label='O2-O1, MRM et al 2018',
                color=col_markers[0])
ax[1].annotate(
    f"{counter}",
    (o2_o1[1], soc_diff_ep[1]),
    fontsize=7,
    ha='center', 
    va='center',
    color="white"
)
counter += 1
ax[1].errorbar(o1_oid[1], soc_diff_ep[0], xerr=o1_oid_err, yerr=mwc_soc_err,
                fmt='o', capsize=0, label='O1-Oid, MRM et al 2018',
                color=col_markers[1])
ax[1].annotate(
    f"{counter}",
    (o1_oid[1], soc_diff_ep[0]),
    fontsize=7,
    ha='center', 
    va='center',
    color="white"
)
counter += 1
ax[1].plot((0, 2.5),(0, 2.5), 'k--')
ax[1].set_xlabel(r'$\ln(k_1^-/k_2^-)$ (this study)', fontsize=8, labelpad=0)
ax[1].set_ylabel(
    r'$\beta(\Delta\epsilon_1-\Delta\epsilon_2)$' + '\n(prev studies)',
    fontsize=8
)

# ##########################################################################
# COMPARING INFERRED k_off RATES TO HAMMAR ET AL 2014
# ##########################################################################
hammar_O1 = (5.3, 0.2) #minutes, mean +/- sem
hammar_Oid = (9.3, 0.4)
# sloppy transform to rates
ham_kO1  = (1/5.3, 7e-3) #min^(-1)
ham_kOid = (1/9.3, 4.6e-3)

mRNA_deg = 1/3 # min^(-1)
koff_O1 = np.percentile(10**o1_samples, (5,50,95)) # still dim-less
koff_Oid = np.percentile(10**oid_samples, (5,50,95))
# now convert to real units, min^(-1)
koff_O1 *= mRNA_deg
koff_Oid *= mRNA_deg
# diff to get errorbars
err_O1 = np.diff(koff_O1).reshape(2,1)
err_Oid = np.diff(koff_Oid).reshape(2,1)
ax[2].errorbar(koff_O1[1], ham_kO1[0], xerr=err_O1, yerr=ham_kO1[1],
                fmt='o', capsize=0, label=r"O1, $\gamma^{-1} = $3 min",
                color=col_markers[2])
ax[2].annotate(
    f"{counter}",
    (koff_O1[1], ham_kO1[0]),
    fontsize=7,
    ha='center', 
    va='center',
    color="white"
)
counter += 1
ax[2].errorbar(koff_Oid[1], ham_kOid[0], xerr=err_Oid, yerr=ham_kOid[1],
                fmt='o', capsize=0, label=r"Oid, $\gamma^{-1} = $3 min",
                color=col_markers[3])
ax[2].annotate(
    f"{counter}",
    (koff_Oid[1], ham_kOid[0]),
    fontsize=7,
    ha='center', 
    va='center',
    color="white"
)
counter += 1
# repeat our calculation with a different mRNA lifetime
mRNA_deg = 1/5 # min^(-1)
koff_O1 = np.percentile(10**o1_samples, (5,50,95)) # still dim-less
koff_Oid = np.percentile(10**oid_samples, (5,50,95))
# now convert to real units, min^(-1)
koff_O1 *= mRNA_deg
koff_Oid *= mRNA_deg
# diff to get errorbars
err_O1 = np.diff(koff_O1).reshape(2,1)
err_Oid = np.diff(koff_Oid).reshape(2,1)
ax[2].errorbar(koff_O1[1], ham_kO1[0], xerr=err_O1, yerr=ham_kO1[1],
                fmt='o', capsize=0, label=r"O1, $\gamma^{-1} = $5 min",
                color=col_markers[2])
ax[2].annotate(
    f"{counter}",
    (koff_O1[1], ham_kO1[0]),
    fontsize=7,
    ha='center', 
    va='center',
    color="white"
)
counter += 1
ax[2].errorbar(koff_Oid[1], ham_kOid[0], xerr=err_Oid, yerr=ham_kOid[1],
                fmt='o', capsize=0, label=r"Oid, $\gamma^{-1} = $5 min",
                color=col_markers[3])
ax[2].annotate(
    f"{counter}",
    (koff_Oid[1], ham_kOid[0]),
    fontsize=7,
    ha='center', 
    va='center',
    color="white"
)

ax[2].plot((0, 0.3),(0, 0.3), 'k--')
ax[2].set_xlabel(r'$k_R^-$ (min$^{-1})$ (this study)', fontsize=8)
ax[2].set_ylabel(
    r'$k_R^-$ (min$^{-1})$' + '\n(Hammar et al 2014)',
    fontsize=8,
    multialignment='center',
)


# BOTTOM

# load in the pickled samples
pklfile = open(
    f"{repo_rootdir}/data/mcmc_samples/repression_pooled_expts.pkl", 'rb'
    )
model, sampler, ppc_uv5, ppc_rep = dill.load(pklfile)
pklfile.close()
inf_dat = az.convert_to_inference_data(
    sampler, var_names=model.var_labels
    )

data_uv5, data_rep = srep.utils.condense_data(model.expts)

n_dim = np.shape(model.var_labels)

# organize all the options upfront
all_expts = (
    ("Oid_2ngmL", "Oid_1ngmL"),
    ("O1_1ngmL", "O1_2ngmL", "O1_10ngmL"),
    ("O2_0p5ngmL", "O2_1ngmL", "O2_2ngmL", "O2_10ngmL")
    )

data_colors = ('black', 'black', 'black', 'black')
uv5_colors = {'ppc':'green', 'data':'black'}
# convert labels to hex colors
# ppc_colors = [pboc_colors[label] for label in ppc_colors]
# data_colors = [pboc_colors[label] for label in data_colors]
ppc_alpha = 0.3
ppc_lw = 0.2
data_lw = 0.6
ptiles = (95,)
# then (nearly?) all the rest below will not need changing

for k, ax in enumerate([ax_3, ax_4, ax_5]):
    #     continue # upper left panel is handled separately
    ppc_labels = all_expts[k]

    # Extract title   
    title = f"operator {ppc_labels[0].split('_')[0]}"

    # now loop over repressed datasets & plot ppc + observed data
    for j, expt in enumerate(ppc_labels):
        expt_idx = model.expts.index(expt)
        # Extract legend
        label = expt.split("_")[1]
        label = label.replace("ngmL", " ng/mL")
        label = label.replace("0p5", "0.5")
        # Find color for concentration
        col = aTc_col_dict[expt.split("_")[1]]
        srep.viz.predictive_ecdf(
            srep.utils.uncondense_ppc(ppc_rep[expt_idx]),
            data=srep.utils.uncondense_valuescounts(data_rep[expt_idx]),
            color=col,
            data_color='black',
            data_size=1,
            percentiles=ptiles,
            discrete=True,
            ax=ax,
            pred_label=label,
            )

    # finally add UV5
    srep.viz.predictive_ecdf(
        srep.utils.uncondense_ppc(ppc_uv5),
        data=srep.utils.uncondense_valuescounts(data_uv5),
        color='purple',
        data_color='black',
        data_size=1,
        percentiles=ptiles,
        discrete=True,
        ax=ax,
        pred_label='UV5',
        data_label='data'
        )

    # finishing touches
    ax.set_xlabel('mRNA counts per cell')
    ax.set_xlim(-2.5, 55)
    srep.viz.titlebox(ax, title, bgcolor="#FFEDC0")
    # ax.legend(loc='lower right', fontsize='small')
# Set yaxis label for first plot
ax_3.set_ylabel('ECDF')
# Set legend for last plot
ax_5.legend(loc="lower right", bbox_to_anchor=(1, -0.4), ncol=6, fontsize=8)

# Add plot labels
plt.gcf().text(-0.03, 1, "(A)", fontsize=14)
plt.gcf().text(0.55, 1, "(B)", fontsize=14)
plt.gcf().text(-0.03, 0.40, "(C)", fontsize=14)

plt.savefig(
    f"{repo_rootdir}/figures/main/fig04_v2.pdf", bbox_inches='tight'
)
# %%

