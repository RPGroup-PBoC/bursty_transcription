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

fig, (ax, _) = plt.subplots(1, 2, figsize=(9,5))
for gene in all_samples:
    alpha_samples = all_samples[gene].posterior.alpha.values.flatten()
    b_samples = all_samples[gene].posterior.b.values.flatten()
    x_contour, y_contour = bebi103.viz.contour_lines_from_samples(
        alpha_samples, b_samples, levels=0.95, smooth=0.025
    )
    xy_path = list(zip(x_contour[0], y_contour[0]))
    ax.loglog(x_contour[0], y_contour[0], label=gene, linewidth=0.6)
ax.legend(loc='lower center', ncol=3, fontsize='small')
ax.set_ylim(top=1.2e1)
ax.set_xlim(right=1e1)
ax.set_xlabel(r'$\alpha$ (bursts per mRNA lifetime)')
ax.set_ylabel(r'$b$ (transcripts per burst)')

plt.savefig(f"{repo_rootdir}/figures/fig2/all_constit_post.pdf")