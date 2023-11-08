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
"""This module contains the class `DrawHist`."""

from __future__ import annotations

__all__ = ["DrawQhist"]

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# - global variables -------------------------------
DEFAULT_CSET = "bright"


# - class definition -------------------------------
class DrawQhist:
    r"""Display pixel-quality data as histograms.

    Notes
    -----
    When the input data is a xarray.DataArray then the attribute 'long_name'
    are used in the plot decoration.

    Examples
    --------
    Generate a figure with pixel-quality histogram plots.

    >>> report = MONplot("test_monplot.pdf", "This is an example figure")
    >>> report.set_institute("SRON")
    >>> plot = DrawQhist()
    >>> fig, axarr = plot.subplot(len(xds.data_vars))
    >>> plot.draw(axarr, xds)
    >>> report.add_copyright(axx[-1])
    >>> report.close_this_page(fig, fig_info)
    >>> report.close()
    """

    @property
    def blank_legend_handle(self: DrawQhist) -> Rectangle:
        """Show only label in a legend entry, no handle.

        See Also
        --------
        matplotlib.pyplot.legend : Place a legend on the Axes.
        """
        return Rectangle((0, 0), 0, 0, fill=False, edgecolor="none", visible=False)

    def subplots(self: DrawQhist, npanels: int) -> tuple[Figure, list[Axes, ...]]:
        """Create a figure and a set of subplots for qhist-plots."""
        figsize = (10.0, 1 + (npanels + 1) * 1.65)
        fig, axarr = plt.subplots(npanels, sharex="all", figsize=figsize)
        if npanels == 1:
            axarr = [axarr]
        margin = min(1.1 / (1.8 * (npanels + 1)), 0.25)
        fig.subplots_adjust(bottom=margin, top=1 - margin, hspace=0.02)
        return fig, axarr

    def draw(
            self: DrawQhist,
            axarr: list[Axes],
            qxds: xr.Dataset,
            *,
            density: bool = True,
            title: str | None = None
    ) -> None:
        """Add a subplot showing pixel-quality data as a histogram.

        Parameters
        ----------
        axarr :  list[matplotlib.Axes]
           Matplotlib Axes object of plot window
        qxds :  xr.Dataset
           Object holding pixel-quality data and attributes
        density :  bool, default=True
           See method MONplot::draw_qhist for a description
        title :  str, optional
           Title of this figure using `Axis.set_title`
        """
        # add title to image panel
        if title is None:
            title = "Histograms of pixel-quality"
        axarr[0].set_title(title)

        for axx, key in zip(axarr, qxds.data_vars, strict=True):
            qdata = np.flatten(qxds[key].values)
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
            legend = axx.legend(
                [self.blank_legend_handle],
                [qxds[key].attrs["long_name"]],
                loc="upper left",
            )
            legend.draw_frame(False)
            # add decoration
            axx.grid(which="major", axis="y", color="#AAAAAA", ls="--")
            axx.set_xlim([0, 1])
            axx.xaxis.set_minor_locator(AutoMinorLocator(2))
            axx.set_ylabel("density" if density else "count")
            axx.set_ylim([1e-4, 10])
            axx.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])

        # finally add a label for the X-coordinate
        axarr[-1].set_xlabel("pixel quality")
