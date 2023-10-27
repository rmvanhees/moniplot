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
"""This module holds `fig_draw_qhist` which are used by `draw_qhist`."""

from __future__ import annotations

__all__ = ["fig_draw_qhist"]

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.ticker import AutoMinorLocator

from .lib.fig_legend import blank_legend_handle

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# - main functions ---------------------------------
def fig_draw_qhist(axx: Axes, qdata: np.ndarray, label: str, density: bool) -> None:
    """Add a subplot showing pixel-quality data as a histogram.

    Parameters
    ----------
    axx :  matplotlib.Axes
       Matplotlib Axes object of plot window
    qdata :  numpy.ndarray
       Object holding pixel-quality data and attributes
    label :  str
       Name describing qdata, displayed in the upper left corner
    density :  bool
       See method MONplot::draw_qhist for a description
    """
    qdata[np.isnan(qdata)] = 0.0
    # draw histogram
    axx.hist(
        qdata,
        bins=10,
        range=(0, 1),
        density=density,
        histtype="bar",
        align="mid",
        log=True,
        fill=True,
        edgecolor="#4477AA",
        facecolor="#77AADD",
        linewidth=1.5,
    )
    # add label
    legend = axx.legend([blank_legend_handle()], [label], loc="upper left")
    legend.draw_frame(False)
    # add decoration
    axx.grid(which="major", axis="y", color="#AAAAAA", ls="--")
    axx.set_xlim([0, 1])
    axx.xaxis.set_minor_locator(AutoMinorLocator(2))
    axx.set_ylabel("density" if density else "count")
    axx.set_ylim([1e-4, 10])
    axx.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
