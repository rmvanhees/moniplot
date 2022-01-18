"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Defines functions fig_draw_lplot and close_draw_lplot
used by MONplot::draw_lplot()

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from ..tol_colors import tol_cset


def fig_draw_lplot(axx, xdata, ydata, icol: str, use_steps=False, **kwargs):
    """
    add line plot to figure
    """
    # define colors
    cset = tol_cset('bright')
    color = cset[icol % len(cset)]

    if use_steps:
        edges = np.append(xdata, xdata[-1])
        values = np.append(ydata, ydata[-1])
        axx.stairs(values, edges, color=color, **kwargs)
    else:
        axx.plot(xdata, ydata, color=color, **kwargs)


def close_draw_lplot(axx, time_axis: bool, title: str, **kwargs):
    """
    close the figure created with MONplot::draw_lplot()
    """
    # add title to image panel
    if title is not None:
        axx.set_title(title, fontsize='large')
    # add grid lines (default settings)
    axx.grid(True)
    # add X & Y label
    if 'xlabel' in kwargs:
        axx.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        axx.set_ylabel(kwargs['ylabel'])
    # set the limits of the X-axis & Y-axis
    if 'xlim' in kwargs:
        axx.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        axx.set_ylim(kwargs['ylim'])
    # set the scale of the X & Y axis {"linear", "log", "symlog", ...}
    if 'xscale' in kwargs:
        axx.set_xscale(kwargs['xscale'])
    if 'yscale' in kwargs:
        axx.set_ylabel(kwargs['yscale'])
    # format the X-axis when it is a time-axis
    if time_axis:
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

    # draw legenda in figure
    if axx.get_legend_handles_labels()[1]:
        axx.legend(fontsize='small', loc='best')
