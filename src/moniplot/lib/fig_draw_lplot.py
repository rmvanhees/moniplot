#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2023 SRON - Netherlands Institute for Space Research
#
# License:  GPLv3
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This module holds `fig_draw_lplot` and `close_draw_lplot`
which are used by `draw_lplot`.
"""
from __future__ import annotations

__all__ = ['fig_draw_lplot', 'close_draw_lplot']

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter

if TYPE_CHECKING:
    from matplotlib import Axes


def fig_draw_lplot(axx: Axes,
                   xdata: np.ndarray,
                   ydata: np.ndarray, use_steps:
                   bool = False,
                   **kwargs: int) -> None:
    """Add line plot to figure.

    Parameters
    ----------
    axx : matplotlib.Axes
       Matplotlib Axes object of plot window.
    xdata : array_like
       X data.
    ydata : array_like
       Y data.
    use_steps : bool, default=False
       use plt.stairs() instead of plt.plot().
    **kwargs : keyword arguments
       Keywords passed to matplotlib.pyplot.plot

    See Also
    --------
    matplotlib.pyplot.plot, matplotlib.pyplot.stairs
    """
    if use_steps:
        edges = np.append(xdata, xdata[-1])
        values = np.append(ydata, ydata[-1])
        axx.stairs(values, edges, **kwargs)
    else:
        axx.plot(xdata, ydata, **kwargs)


def close_draw_lplot(axx: Axes,
                     time_axis: bool,
                     title: str | None,
                     kwlegend: dict | None,
                     **kwargs: int) -> None:
    """Close the figure created with MONplot::draw_lplot().

    Parameters
    ----------
    axx : matplotlib.Axes
       Matplotlib Axes object of plot window
    time_axis : bool
       The xlabels represent date/time
    title : str | None
       Title of the figure
    kwlegend : dict | None
       Provide keywords for the function `Axes.legend`.
       Default: {'fontsize': 'small', 'loc': 'best'}
    **kwargs : keyword arguments
       Recognized are 'xlabel', 'ylabel', 'xlim', 'ylim', 'xscale', 'yscale'
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
        if kwlegend is None:
            axx.legend(fontsize='small', loc='best')
        else:
            axx.legend(**kwlegend)
