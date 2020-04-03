import pickle
from git import Repo #for directory convenience

import numpy as np
import pandas as pd
import arviz as az

import matplotlib.pyplot as plt
import bebi103.viz

import srep.viz
from srep.data_loader import load_FISH_by_promoter


repo = Repo("./", search_parent_directories=True)
# repo_rootdir holds the absolute path to the top-level of our repo
repo_rootdir = repo.working_tree_dir

colors = srep.viz.color_selector('pboc')
srep.viz.plotting_style()

pklfile = open(f"{repo_rootdir}/data/stan_samples/constit_post_inf.pkl", 'rb')
all_samples = pickle.load(pklfile)
pklfile.close()

df_energies, = load_FISH_by_promoter(("energies",))

fig, ax = plt.subplots(2, 2, figsize=(9,9))
color_pal = srep.viz.color_selector('constit')

# 95% HPD for all 18 constitutive promoters
# loop thru df, not all_samples keys, so we get deterministic order!
for promoter in df_energies.Name:
    alpha_samples = all_samples[promoter].posterior.alpha.values.flatten()
    b_samples = all_samples[promoter].posterior.b.values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        b_samples, alpha_samples, levels=0.95, smooth=0.025
    )
    ax[1,0].loglog(x_contour[0],
        y_contour[0],
        label=promoter,
        linewidth=0.6,
        color=color_pal[promoter])
ax[1,0].set_xlim(right=1.2e1)
ax[1,0].set_ylim(top=1e1)
ax[1,0].set_ylabel(r'$\alpha$ (bursts per mRNA lifetime)')
ax[1,0].set_xlabel(r'$b$ (transcripts per burst)')

# log(burst rate) vs energies from energy matrix
for promoter in df_energies.Name:
    samples = all_samples[promoter].posterior.alpha.values.flatten()
    ptile_low, ptile_med, ptile_upr = np.percentile(samples, (2.5, 50, 97.5))
    err_lower = ptile_med - ptile_low
    err_upper = ptile_upr - ptile_med
    ax[1,1].errorbar(
        df_energies[df_energies.Name == promoter]["Energy (kT)"],
        ptile_med,
        yerr=np.array((err_lower, err_upper)).reshape((2,1)),
        fmt='.',
        label=promoter,
        color=color_pal[promoter])
ax[1,1].set_ylabel(r'$\alpha$ (bursts per mRNA lifetime)')
ax[1,1].set_xlabel(r'Binding energy $(k_BT)$')
ax[1,1].set_yscale("log")
ax[1,1].legend(loc='upper right', ncol=2, fontsize='small')

plt.savefig(f"{repo_rootdir}/figures/fig2/all_constit_post.pdf")