import bokeh.io
import bokeh.plotting
import bokeh.layouts
import bokeh.palettes
import seaborn as sns
import numpy as np
import pandas as pd
import xarray
import os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
from matplotlib.patches import BoxStyle
from matplotlib.offsetbox import AnchoredText

import bebi103

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
    plt.rc('mathtext', fontset='stixsans', sf='sansserif')
    plt.rc('legend', title_fontsize='8', frameon=True,
            facecolor='#E3DCD0', framealpha=1)
    sns.set_style('darkgrid', rc=rc)
    sns.set_palette("colorblind", color_codes=True)

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
            'dark_brown':'#905426',
            'black':'#060606'}

    elif style.lower() == 'constit':
        colors = [
            '#7AA974',
            '#ffef14',
            '#A9BFE3',
            '#D56C55',
            '#65c9b2',
            '#AB85AC',
            '#905426',
            '#a8a8a8',
            '#0f0f0f']
    return colors


def predictive_ecdf(
    samples,
    data=None,
    diff=False,
    percentiles=[95, 75, 50, 25],
    color="blue",
    data_color="orange",
    data_staircase=True,
    data_size=2,
    median_lw=1,
    x=None,
    discrete=False,
    ax=None,
    pred_label=None,
    data_label=None,
    **kwargs,
):
    """Plot a predictive ECDF from samples.
    Code drawn directly from Justin Bois' bebi103 module, only mod is
    changing plotting backend from bokeh -> mpl.

    TODO:: UPDATE DOCS, currently outdated!!
    Parameters
    ----------
    samples : Numpy array or xarray, shape (n_samples, n) or xarray DataArray
        A Numpy array containing predictive samples.
        n_samples is the number of posterior samples, and n is the number
        of observed data points, i.e., n is the number of posterior
        predictive samples for each of the posterior samples
    data : Numpy array, shape (n,) or xarray DataArray
        If not None, ECDF of measured data is overlaid with predictive
        ECDF.
    diff : bool, default True
        If True, the ECDFs minus median of the predictive ECDF are
        plotted.
    percentiles : list, default [80, 60, 40, 20]
        Percentiles for making colored envelopes for confidence
        intervals for the predictive ECDFs. Maximally four can be
        specified.
    color : str, default 'blue'
        One of ['green', 'blue', 'red', 'gray', 'purple', 'orange'].
        There are used to make the color scheme of shading of
        percentiles.
    data_color : str, default 'orange'
        String representing the color of the data to be plotted over the
        confidence interval envelopes.
    data_staircase : bool, default True
        If True, plot the ECDF of the data as a staircase.
        Otherwise plot it as dots.
    data_size : int, default 2
        Size of marker (if `data_line` if False) or thickness of line
        (if `data_staircase` is True) of plot of data.
    x : Numpy array, default None
        Points at which to evaluate the ECDF. If None, points are
        automatically generated based on the data range.
    discrete : bool, default False
        If True, the samples take on discrete values.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    kwargs
        All other kwargs are passed to bokeh.plotting.figure().

    Returns
    -------
    output : Bokeh figure
        Figure populated with glyphs describing range of values for the
        ECDF of the samples. The shading goes according to percentiles
        of samples of the ECDF, with the median ECDF plotted as line in
        the middle.
    """
    if type(samples) != np.ndarray:
        if type(samples) == xarray.core.dataarray.DataArray:
            samples = samples.squeeze().values
        else:
            raise RuntimeError("Samples can only be Numpy arrays and xarrays.")

    if len(percentiles) > 4:
        raise RuntimeError("Can specify maximally four percentiles.")

    # Build ptiles
    percentiles = np.sort(percentiles)[::-1]
    ptiles = [pt for pt in percentiles if pt > 0]
    ptiles = (
        [50 - pt / 2 for pt in percentiles]
        + [50]
        + [50 + pt / 2 for pt in percentiles[::-1]]
    )
    ptiles_str = [str(pt) for pt in ptiles]

    if color not in ["green", "blue", "red", "gray", "purple", "orange", "betancourt"]:
        raise RuntimeError(
            "Only allowed colors are 'green', 'blue', 'red', 'gray', 'purple', 'orange'"
        )

    colors = {
        "blue": ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
        "green": ["#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"],
        "red": ["#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"],
        "orange": ["#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#8c2d04"],
        "purple": ["#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"],
        "gray": ["#bdbdbd", "#969696", "#737373", "#525252", "#252525"],
        "betancourt": [
            "#DCBCBC",
            "#C79999",
            "#B97C7C",
            "#A25050",
            "#8F2727",
            "#7C0000",
        ],
    }

    data_range = samples.max() - samples.min()
    if discrete and x is None:
        x = np.arange(samples.min(), samples.max() + 1)
    elif x is None:
        x = np.linspace(
            samples.min() - 0.05 * data_range, samples.max() + 0.05 * data_range, 400
        )

    ecdfs = np.array([bebi103.viz._ecdf_arbitrary_points(sample, x) for sample in samples])

    df_ecdf = pd.DataFrame()
    for ptile in ptiles:
        df_ecdf[str(ptile)] = np.percentile(
            ecdfs, ptile, axis=0, interpolation="higher"
        )

    df_ecdf["x"] = x

    if data is not None and diff:
        ecdfs = np.array(
            [bebi103.viz._ecdf_arbitrary_points(sample, np.sort(data)) for sample in samples]
        )
        ecdf_data_median = np.percentile(ecdfs, 50, axis=0, interpolation="higher")

    if diff:
        for ptile in filter(lambda item: item != "50", ptiles_str):
            df_ecdf[ptile] -= df_ecdf["50"]
        df_ecdf["50"] = 0.0

    # for now, force user to creat fig, ax ahead of time
    # if p is None:
    #     x_axis_label = kwargs.pop("x_axis_label", "x")
    #     y_axis_label = kwargs.pop("y_axis_label", "ECDF difference" if diff else "ECDF")

    #     if "plot_height" not in kwargs and "frame_height" not in kwargs:
    #         kwargs["frame_height"] = 325
    #     if "plot_width" not in kwargs and "frame_width" not in kwargs:
    #         kwargs["frame_width"] = 400
    #     p = bokeh.plotting.figure(
    #         x_axis_label=x_axis_label, y_axis_label=y_axis_label, **kwargs
    #     )

    for i, ptile in enumerate(ptiles_str[: len(ptiles_str) // 2]):
        if discrete:
            x, y1 = bebi103.viz.cdf_to_staircase(df_ecdf["x"].values, df_ecdf[ptile].values)
            _, y2 = bebi103.viz.cdf_to_staircase(
                df_ecdf["x"].values, df_ecdf[ptiles_str[-i - 1]].values
            )
        else:
            x = df_ecdf["x"]
            y1 = df_ecdf[ptile]
            y2 = df_ecdf[ptiles_str[-i - 1]]
        ax.fill_between(x, y1, y2, color=colors[color][i],)

    # The median as a solid line
    if discrete:
        x, y = bebi103.viz.cdf_to_staircase(df_ecdf["x"], df_ecdf["50"])
    else:
        x, y = df_ecdf["x"], df_ecdf["50"]
    ax.plot(x, y, linewidth=median_lw, color=colors[color][-1], label=pred_label)

    # Overlay data set
    if data is not None:
        x_data, y_data = bebi103.viz._ecdf_vals(data, staircase=False)
        if diff:
            # subtracting off median wrecks y-coords for duplicated x-values...
            y_data -= ecdf_data_median
            # ...so take only unique values,...
            unique_x = np.unique(x_data)
            # ...find the (correct) max y-value for each...
            unique_inds = np.searchsorted(x_data, unique_x, side="right") - 1
            # ...and use only that going forward
            y_data = y_data[unique_inds]
            x_data = unique_x
        if data_staircase:
            x_data, y_data = bebi103.viz.cdf_to_staircase(x_data, y_data)
            ax.plot(x_data, y_data, color=data_color, linewidth=data_size, label=data_label)
        else:
            ax.scatter(x_data, y_data, color=data_color, s=data_size, label=data_label)

    return ax


def ppc_ecdf_pair(posterior_samples, ppc_var, df, percentiles=(95, 75, 50, 25), ax=None, **kwargs):
    """Plot posterior predictive ECDFs.
    Credit to JB for part of this function,
    double check how much I wrote & which tutorial/year
    I borrowed from, maybe the finch beak example??
    """
    n_samples = (
        posterior_samples.posterior_predictive.dims["chain"]
        * posterior_samples.posterior_predictive.dims["draw"]
    )

    ax[0] = predictive_ecdf(
        posterior_samples.posterior_predictive[ppc_var].values.reshape(
            (n_samples, len(df))
        ),
        data=df["mRNA_cell"],
        percentiles=percentiles,
        discrete=True,
        ax=ax[0],
        **kwargs
    )

    ax[1] = predictive_ecdf(
        posterior_samples.posterior_predictive[ppc_var].values.reshape(
            (n_samples, len(df))
        ),
        data=df["mRNA_cell"],
        percentiles=percentiles,
        discrete=True,
        diff=True,
        ax=ax[1],
        **kwargs
    )
    # p1.x_range = p2.x_range
    
    return ax

# %%
fig, ax = plt.subplots(1, 2, figsize= (6, 2.5))
srep.viz.ppc_ecdf_pair(
    all_samples['5DL10'],
    'mRNA_counts_ppc',
    df_unreg[df_unreg['experiment'] == '5DL10'],
    ax=ax,
    data_color='black',
    color='green',
    data_label='5DL10'
    )
plt.legend()
# %%

def ecdf(data):
    """
    data[0] should be a 1D array of observed values, and
    data[1] should be a 1D array of the # of observations of each value.
    e.g., the output of np.unique([0,1,0,3,3,2], return_counts=True).
    Logic is credit to Justin Bois in the bebi103 module,
    bebi103.viz.bebi103.viz.cdf_to_staircase.
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

def titlebox(
    ax, text, color='black', bgcolor=None, size=8, boxsize=0.1, pad=0.05, loc=10, **kwargs
):
    """Sets a colored box about the title with the width of the plot.
   To use, simply call `titlebox(ax, 'plot title', bgcolor='blue')
   """
    boxsize=str(boxsize * 100)  + '%'
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size=boxsize, pad=pad)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["bottom"].set_visible(False)
    cax.spines["left"].set_visible(False)
    plt.setp(cax.spines.values(), color=color)
    if bgcolor != None:
        cax.set_facecolor(bgcolor)
    else:
        cax.set_facecolor("white")
    at = AnchoredText(text, loc=loc, frameon=False, prop=dict(size=size, color=color))
    cax.add_artist(at)

def bebi103_colors():
    """
    Function to return the list of colors used for the quantile PPC plots 
    """
    return {
        "blue": ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
        "green": ["#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"],
        "red": ["#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"],
        "orange": ["#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#8c2d04"],
        "purple": ["#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"],
        "gray": ["#bdbdbd", "#969696", "#737373", "#525252", "#252525"],
        "betancourt": [
            "#DCBCBC",
            "#C79999",
            "#B97C7C",
            "#A25050",
            "#8F2727",
            "#7C0000",
        ],
    }
