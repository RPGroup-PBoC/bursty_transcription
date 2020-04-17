# This script is to test my apparent error in derivation of PGF for
# bursty transcription + repression. Conclusion: no, my PGF is fine,
# I just screwed up somewhere in deriving moments from the PGF
#%%
import numpy as np
import numba

import matplotlib.pyplot as plt
import bokeh
import bokeh.io
bokeh.io.output_notebook()

import biocircuits
import srep

kRoff = 1.
kRon  = 1.
k_i   = 1.
k_u   = 100.
r     = 100.
gamma = 1.

@numba.njit
def simple_propensity(propensities, population, t):
    """Updates an array of propensities given a set of parameters
    and an array of populations.
    """
    # Unpack population
    rep, emp, act, m = population
    
    # Update propensities
    propensities[0] = kRoff * rep # unbind repressor
    propensities[1] = kRon * emp  # bind repressor
    propensities[2] = k_i * emp   # initiate burst
    propensities[3] = k_u * act   # terminate burst
    propensities[4] = r * act     # make transcript
    propensities[5] = gamma * m   # degrade transcript

# Columns are rep bound, empty, active, and m count
simple_update = np.array([[-1,1, 0, 0],  # unbind repressor
                          [1,-1, 0, 0],  # bind repressor
                          [0,-1, 1, 0],  # initiate burst
                          [0, 1,-1, 0],  # terminate burst
                          [0, 0, 0, 1],  # make transcript
                          [0, 0, 0,-1]], # degrade transcript
                         dtype=np.int)

time_points = np.linspace(0,1000,100)
samples = biocircuits.gillespie_ssa(
    simple_propensity,
    simple_update,
    np.array([1,0,0,0]), # init pop
    time_points,
    # return_time_points=True,
    size=500,
    n_threads=2,
    progress_bar=True,
)

# %%
plots = bokeh.plotting.figure(plot_width=400,
                              plot_height=400,
                              x_axis_label='dimensionless time',
                              y_axis_label='number of mRNAs')

# Plot trajectories and mean
for x in samples[::100,:,-1]:
    plots.line(time_points, x, line_width=0.3, 
                alpha=0.2, line_join='bevel')
    plots.line(time_points, samples[:,:,-1].mean(axis=0),
                line_width=6, color='orange', line_join='bevel')


bokeh.io.show(plots)

# %%
# pack params for my rng fcn
params = np.array([k_i, r/k_u, kRon, kRoff])
# run rng to generate samples from my PGF solution to compare w/ Gillespie
rng_samples = srep.models.bursty_rep_rng(params, 100000)

p_m = biocircuits.viz.ecdf(samples[:,-50:,-1].flatten(),
                           plot_width=400,
                           plot_height=400,
                           formal=True,
                           x_axis_label='mRNA copy number')
p_m.line(*srep.viz.ecdf(rng_samples), color='orange')
bokeh.io.show(p_m)
# %%
print('mRNA mean copy number =', samples[:,-50:,-1].mean())
print('\nmRNA variance =', samples[:,-50:,-1].std()**2)
print('\nmRNA noise =', samples[:,-50:,-1].std() / samples[:,-50:,-1].mean())

# %%
# So my PGF solution of burstiness + repression is ok, just my derivation
# of <m^2> from the PGF was flawed. Phew!