#%%
import re
import dill
from git import Repo #for directory convenience

import numpy as np
import pandas as pd
import arviz as az

import matplotlib.pyplot as plt
import bebi103.viz

import srep

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

#%%
fig, ax = plt.subplots(2, 2, figsize=(9,9))

# ##########################################################################
# PLOTTING ALL PAIRWISE POSTERIORS FROM POOLED DATA MODEL
# ##########################################################################
lstyle = {'O2':'-', 'O1':'-.', 'Oid':'--'}
lcolor = {'0p5ngmL':'green', '1ngmL':'purple', '2ngmL':'blue', '10ngmL':'red'}
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
        'linewidth':0.6, 'linestyle':lstyle[op], 'color':lcolor[aTc]
        }
    ax[0,1].plot(x_contour[0], y_contour[0], **contour_opts)
    ax[0,1].plot(x_contour[1], y_contour[1], **contour_opts)
        
# off-screen markers to trick legend
for conc in lcolor.keys():
    ax[0,1].plot(
        (100,100),(100,100), '-',
        label=conc,
        color=lcolor[conc],
        # lw=data_lw
    )
for op in lstyle.keys():
    ax[0,1].plot(
        (-1,-1),(100,100), '-',
        label=op,
        color='k',
        linestyle=lstyle[op],
        # lw=data_lw
    )
ax[0,1].set_xlim(-0.7, 0.75)
ax[0,1].set_ylim(-0.6, 1.9)
ax[0,1].set_ylabel(r'$log_{10}(k_R^+/\gamma)$')
ax[0,1].set_xlabel(r'$log_{10}(k_R^-/\gamma)$')
ax[0,1].legend(loc='lower left', fontsize='small', ncol=2)

# ##########################################################################
# COMPARING k_off RATIOS TO BINDING E DIFFERENCES
# ##########################################################################
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

ax[1,0].errorbar(o2_o1[1], hg_diff_ep[1], xerr=o2_o1_err, yerr=hg_rp_err,
                fmt='^b', capsize=2, label='O2-O1, HG & RP 2011')
ax[1,0].errorbar(o1_oid[1], hg_diff_ep[0], xerr=o1_oid_err, yerr=hg_rp_err,
                fmt='^g', capsize=2, label='O1-Oid, HG & RP 2011')
ax[1,0].errorbar(o2_o1[1], soc_diff_ep[1], xerr=o2_o1_err, yerr=mwc_soc_err,
                fmt='ob', capsize=2, label='O2-O1, MRM et al 2018')
ax[1,0].errorbar(o1_oid[1], soc_diff_ep[0], xerr=o1_oid_err, yerr=mwc_soc_err,
                fmt='og', capsize=2, label='O1-Oid, MRM et al 2018')
ax[1,0].plot((0, 2.5),(0, 2.5), 'k--')
ax[1,0].set_xlabel(r'$\ln(k_1^-/k_2^-)$ (this study)')
ax[1,0].set_ylabel(
    r'$\beta(\Delta\epsilon_1-\Delta\epsilon_2)$ (prev studies)'
    )
ax[1,0].legend(fontsize='small')

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
ax[1,1].errorbar(koff_O1[1], ham_kO1[0], xerr=err_O1, yerr=ham_kO1[1],
                fmt='^b', capsize=2, label=r"O1, $\gamma^{-1} = $3 min")
ax[1,1].errorbar(koff_Oid[1], ham_kOid[0], xerr=err_Oid, yerr=ham_kOid[1],
                fmt='^g', capsize=2, label=r"Oid, $\gamma^{-1} = $3 min")

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
ax[1,1].errorbar(koff_O1[1], ham_kO1[0], xerr=err_O1, yerr=ham_kO1[1],
                fmt='ob', capsize=2, label=r"O1, $\gamma^{-1} = $5 min")
ax[1,1].errorbar(koff_Oid[1], ham_kOid[0], xerr=err_Oid, yerr=ham_kOid[1],
                fmt='og', capsize=2, label=r"Oid, $\gamma^{-1} = $5 min")

ax[1,1].plot((0, 0.3),(0, 0.3), 'k--')
ax[1,1].set_xlabel(r'$k_R^-$ min$^{-1}$ (this study)')
ax[1,1].set_ylabel(r'$k_R^-$ min$^{-1}$ (Hammar et al 2014)')
ax[1,1].legend(fontsize='small')

plt.savefig(f"{repo_rootdir}/figures/fig3/fig3_v3.pdf")

# %%
