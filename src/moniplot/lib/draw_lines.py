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
"""This module contains the class `DrawLines`."""

from __future__ import annotations

__all__ = ["DrawLines"]

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter

from moniplot.tol_colors import tol_rgba

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# - global variables -------------------------------
DEFAULT_CSET = "bright"


# - class definition -------------------------------
class DrawLines:
    """Create a line-plot (thin layer around plt.plot/plt.stairs).

    Examples
    --------
    Generate a figure with four line-plots.

    >>> report = MONplot("test_monplot.pdf", "This is an example figure")
    >>> report.set_institute("SRON")plot = DrawLines()
    >>> fig, axx = plt.subplots(2, 2, sharex="all", figsize=(10, 10))
    >>> plot.add_line(axx[0, 0], ydata1, xdata=xdata, marker=".", ls="-", label="1")
    >>> plot.add_line(axx[0, 0], ydata2, xdata=xdata, marker=".", ls="-", label="2")
    >>> plot.draw(axx[0, 0], title="fig 1", xlabel="X", ylabel="Y")
    >>> plot.add_line(axx[0, 1], ydata1, xdata=xdata, marker=".", ls="", label="a")
    >>> plot.add_line(axx[0, 1], ydata2, xdata=xdata, marker="x", ls="", label="b")
    >>> plot.draw(axx[0, 1], title="fig 2", xlabel="X", ylabel="Y")
    >>> plot.add_line(axx[1, 0], ydata1, xdata=xdata, marker="o", ls="-", label="I")
    >>> plot.add_line(axx[1, 0], ydata2, xdata=xdata, marker=".", ls="-", label="II")
    >>> plot.draw(axx[1, 0], title="fig 3", xlabel="X", ylabel="Y")
    >>> plot.add_line(axx[1, 1], ydata1, xdata=xdata, marker=".", ls="", label="one")
    >>> plot.add_line(axx[1, 1], ydata2, xdata=xdata, marker="+", ls="", label="two")
    >>> plot.draw(axx[1, 1], title="fig 4", xlabel="X", ylabel="Y")
    >>> report.add_copyright(axx[1, 1])
    >>> report.close_this_page(fig, None)
    >>> report.close()
    """

    def __init__(
        self: DrawLines,
        *,
        square: bool = False,
    ) -> None:
        """Create DrawLines object."""
        self._cset = tol_rgba(DEFAULT_CSET)
        self.square = square
        self.time_axis = False

    def set_cset(self: DrawLines, cname: str, cnum: int | None = None) -> None:
        """Use alternative color-set through which `draw_lplot` will cycle.

        Parameters
        ----------
        cname :  str
           Name of color set. Use None to get the default matplotlib value.
        cnum : int, optional
           Number of discrete colors in colormap (*not colorset*).
        """
        if not isinstance(cname, str):
            raise ValueError("The name of a color-set should be a string.")
        self._cset = tol_rgba(cname, cnum)

    def unset_cset(self: DrawLines) -> None:
        """Set color set to its default."""
        self._cset = tol_rgba(DEFAULT_CSET)

    def figsize(
        self: DrawLines, ydata: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Create a figure and a set of subplots for line-plots."""
        if self.square:
            figsize = (10, 10)
        elif ydata is None:
            figsize = (10, 7)
        else:
            figsize = {0: (10, 7), 1: (10, 7), 2: (12, 7)}.get(
                len(ydata) // 256, (14, 8)
            )
        return figsize

    def add_line(
        self: DrawLines,
        axx: Axes,
        ydata: np.ndarray,
        *,
        xdata: np.ndarray | None = None,
        use_steps: bool = False,
        **kwargs: int,
    ) -> None:
        """Add a line or scatter-points to a subplot.

        Parameters
        ----------
        axx :  matplotlib.Axes
            Matplotlib Axes object of plot window.
        xdata :  array_like
            X data.
        ydata :  array_like
            Y data.
        use_steps :  bool, default=False
            Use `matplotlib.pyplot.stairs` instead of matplotlib.pyplot..plot.
        **kwargs : keyword arguments
            Keywords passed to matplotlib.pyplot.plot

        See Also
        --------
        matplotlib.pyplot.plot, matplotlib.pyplot.stairs
        """
        if not (axx.dataLim.mutatedx() or axx.dataLim.mutatedy()):
            axx.set_prop_cycle(color=self._cset)

        # draw line in subplot
        if use_steps:
            edges = np.append(xdata, xdata[-1])
            values = np.append(ydata, ydata[-1])
            axx.stairs(values, edges, **kwargs)
        else:
            axx.plot(xdata, ydata, **kwargs)

    def draw(
        self: DrawLines,
        axx: Axes,
        title: str | None = None,
        kwlegend: dict | None = None,
        **kwargs: int,
    ) -> None:
        """Add annotations to a subplot, before closing it.

        Parameters
        ----------
        axx :  matplotlib.Axes
            Matplotlib Axes object of plot window
        title :  str | None
            Title of the figure
        kwlegend :  dict | None
            Provide keywords for the function `Axes.legend`.
            Default: {'fontsize': 'small', 'loc': 'best'}
        **kwargs :  keyword arguments
            Recognized are 'xlabel', 'ylabel', 'xlim', 'ylim', 'xscale', 'yscale'
        """
        # add title to image panel
        if title is not None:
            axx.set_title(title, fontsize="large")
        # add X & Y label
        if "xlabel" in kwargs:
            axx.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            axx.set_ylabel(kwargs["ylabel"])
        # set the limits of the X-axis & Y-axis
        if "xlim" in kwargs:
            axx.set_xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            axx.set_ylim(kwargs["ylim"])
        # set the scale of the X & Y axis {"linear", "log", "symlog", ...}
        if "xscale" in kwargs:
            axx.set_xscale(kwargs["xscale"])
        if "yscale" in kwargs:
            axx.set_ylabel(kwargs["yscale"])

        # format the X-axis when it is a time-axis
        if self.time_axis:
            plt.gcf().autofmt_xdate()
            plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))

        # draw legenda in figure
        if axx.get_legend_handles_labels()[1]:
            if kwlegend is None:
                axx.legend(fontsize="small", loc="best")
            else:
                axx.legend(**kwlegend)

        # add grid lines (default settings)
        axx.grid(True)
