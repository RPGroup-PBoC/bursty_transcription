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
    if  style.lower() not in ['srep', 'pboc']:
        raise ValueError("Provided style must be 'pboc' or 'srep'. {} provided.".format(style))

    # Set the color styles and return.
    if  style.lower() == 'srep':
        colors = {'O3': '#7AA974', 'O2': '#2d98da',
                'Oid': '#EE5A24', 'O1': '#1289A7'} 

    elif style.lower() == 'pboc':
        colors = {'green': '#7AA974', 'light_green': '#BFD598',
              'pale_green': '#DCECCB', 'yellow': '#EAC264',
              'light_yellow': '#F3DAA9', 'pale_yellow': '#FFEDCE',
              'blue': '#738FC1', 'light_blue': '#A9BFE3',
              'pale_blue': '#C9D7EE', 'red': '#D56C55', 'light_red': '#E8B19D',
              'pale_red': '#F1D4C9', 'purple': '#AB85AC',
              'light_purple': '#D4C2D9', 'dark_green':'#7E9D90', 'dark_brown':'#905426'}
    return colors

def ppc_ecdfs(posterior_samples, df):
    """Plot posterior predictive ECDFs."""
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