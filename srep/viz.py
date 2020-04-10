import bokeh.io
import bokeh.plotting
import bokeh.layouts
import bokeh.palettes
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

from bebi103.viz import predictive_ecdf

def plotting_style(grid=True):
    """
    Sets the style to the publication style
    """
    rc = {'axes.facecolor': '#E3DCD0',
          'font.family': 'Lucida Sans Unicode',
          'grid.linestyle': '-',
          'grid.linewidth': 0.5,
          'grid.alpha': 0.75,
          'grid.color': '#ffffff',
          'axes.grid': grid,
          'ytick.direction': 'in',
          'xtick.direction': 'in',
          'xtick.gridOn': True,
          'ytick.gridOn': True,
          'ytick.major.width':5,
          'xtick.major.width':5,
          'ytick.major.size': 5,
          'xtick.major.size': 5,
          'mathtext.fontset': 'stixsans',
          'mathtext.sf': 'sans',
          'legend.frameon': True,
          'legend.facecolor': '#FFEDCE',
          'figure.dpi': 150,
           'xtick.color': 'k',
           'ytick.color': 'k'}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('darkgrid', rc=rc)

def color_selector(style):
    """
    Select the color palette of your choice.

    Parameters
    ----------
    style: str "srep" or "pboc"
        A string identifier for the style. "srep" gives colors for operators.
        "pboc" returns the PBoC2e color palette.

    Returns
    -------
    colors: dict
        Dictionary of colors. For pboc, the keys will be
        the typical color descriptors. 

    """
    # Ensure the provided style name makes sense.
    if  style.lower() not in ['srep', 'pboc', 'constit']:
        raise ValueError(
            "Style {} provided. I only know \
            'srep', 'pboc', and 'constit'.".format(style))

    # Set the color styles and return.
    if  style.lower() == 'srep':
        colors = {'O3': '#7AA974', 'O2': '#2d98da',
                'Oid': '#EE5A24', 'O1': '#1289A7'} 

    elif style.lower() == 'pboc':
        colors = {
            'green': '#7AA974',
            'light_green': '#BFD598',
            'pale_green': '#DCECCB',
            'yellow': '#EAC264',
            'light_yellow': '#F3DAA9',
            'pale_yellow': '#FFEDCE',
            'blue': '#738FC1',
            'light_blue': '#A9BFE3',
            'pale_blue': '#C9D7EE',
            'red': '#D56C55',
            'light_red': '#E8B19D',
            'pale_red': '#F1D4C9',
            'purple': '#AB85AC',
            'light_purple': '#D4C2D9',
            'dark_green':'#7E9D90',
            'dark_brown':'#905426'}

    elif style.lower() == 'constit':
        colors = {
            'UV5': '#7AA974',
            'WTDL20v2': '#BFD598',
            'WT': '#DCECCB',
            'WTDL10': '#EAC264',
            'WTDL20': '#F3DAA9',
            'WTDL30': '#FFEDCE',
            'WTDR30': '#738FC1',
            '5DL1': '#A9BFE3',
            '5DL5': '#C9D7EE',
            '5DL10': '#D56C55',
            '5DL20': '#E8B19D',
            '5DL30': '#F1D4C9',
            '5DR1': '#AB85AC',
            '5DR5': '#D4C2D9',
            '5DR1v2':'#7E9D90',
            '5DR10':'#905426',
            '5DR20':'#a8a8a8',
            '5DR30':'#0f0f0f'}
    return colors

def ppc_ecdfs(posterior_samples, df):
    """Plot posterior predictive ECDFs.
    Credit to JB for at least part of this function,
    double check how much I wrote & which tutorial/year
    I borrowed from, maybe the finch beak example??

    predictive_ecdf only works for Bokeh. Will it be easier to hack
    this for matplotlib, or just take the logic guts and write my own??
    
    Input: posterior samples is an arviz InferenceData object
    w/ posterior and posterior predictive samples."""
    n_samples = (
        posterior_samples.posterior_predictive.dims["chain"]
        * posterior_samples.posterior_predictive.dims["draw"]
    )

    p1 = predictive_ecdf(
        posterior_samples.posterior_predictive["mRNA_counts_ppc"].values.reshape(
            (n_samples, len(df))
        ),
        data=df["mRNA_cell"],
        discrete=True,
        x_axis_label="mRNA counts per cells",
        frame_width=200,
        frame_height=200
    )

    p2 = predictive_ecdf(
        posterior_samples.posterior_predictive["mRNA_counts_ppc"].values.reshape(
            (n_samples, len(df))
        ),
        data=df["mRNA_cell"],
        percentiles=[95, 90, 80, 50],
        diff=True,
        discrete=True,
        x_axis_label="mRNA counts per cells",
        frame_width=200,
        frame_height=200
    )
    p1.x_range = p2.x_range
    
    return [p1, p2]

def ecdf(data):
    """
    data[0] should be a 1D array of observed values, and
    data[1] should be a 1D array of the # of observations of each value.
    e.g., the output of np.unique([0,1,0,3,3,2], return_counts=True).
    Logic is credit to Justin Bois in the bebi103 module,
    bebi103.viz.cdf_to_staircase.
    """
    yvals = np.asfarray(np.cumsum(data[1]))
    yvals *= 1.0 / yvals[-1]

    x_staircase = np.empty(2 * len(yvals))
    y_staircase = np.empty(2 * len(yvals))

    y_staircase[0] = 0
    y_staircase[1::2] = yvals
    y_staircase[2::2] = yvals[:-1]

    x_staircase[::2] = data[0]
    x_staircase[1::2] = data[0]

    return x_staircase, y_staircase

def traceplot(sampler, labels=None):
    samples = sampler.get_chain()
    n_dim = np.shape(sampler.get_chain())[-1]
    fig, axes = plt.subplots(n_dim, figsize=(10, 2*n_dim), sharex=True)
    if labels == None:
        labels = [f"foo#{i}" for i in range(n_dim)]
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    return fig, axes