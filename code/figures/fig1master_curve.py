# -*- coding: utf-8 -*-
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import srep.viz
colors = srep.viz.color_selector('pboc')
srep.viz.plotting_style()

fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=150)
x = np.linspace(-6,6)
ax.plot(x, 1/(1+np.exp(-x)))
ax.set_ylabel('fold-change')
ax.set_xlabel(r"$\Delta F_R - \log\rho~(k_BT)$")
plt.savefig('../../figures/fig1/fig1_master_curve.pdf', bbox_inches='tight')
