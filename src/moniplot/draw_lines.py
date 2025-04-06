#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2025 SRON
#    All rights reserved.
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
#
"""Definition of the monitplot class `DrawLines`."""

from __future__ import annotations

__all__ = ["DrawLines"]

import datetime as dt
from typing import TYPE_CHECKING, NotRequired, TypedDict, Unpack

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import (
    AutoDateLocator,
    ConciseDateFormatter,
    DateFormatter,
)

from .tol_colors import tol_rgba

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# - global variables -------------------------------
DEFAULT_CSET = "bright"


class DrawKeys(TypedDict):
    """Define keyword arguments of DrawLines.draw()."""

    title: NotRequired[str]
    xlabel: NotRequired[str]
    ylabel: NotRequired[str]
    xlim: NotRequired[list[float, float]]
    ylim: NotRequired[list[float, float]]
    xscale: NotRequired[str]
    yscale: NotRequired[str]
    kwlegend: NotRequired[dict]


# - class definition -------------------------------
class DrawLines:
    """Create a line-plot (thin layer around plt.plot/plt.stairs).

    Examples
    --------
    Generate a figure with four line-plots.

    >>> report = MONplot("test_monplot.pdf", "This is an example figure")
    >>> report.set_institute("SRON")
    >>> plot = DrawLines()
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
        self.xdata = None
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
        self: DrawLines,
        ydata: np.ndarray | tuple | list | None = None,
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
        ydata: np.ndarray | tuple | list,
        *,
        xdata: np.ndarray | tuple | list | None = None,
        use_steps: bool = False,
        **kwargs: int,
    ) -> None:
        """Add a line or scatter-points to a subplot.

        Parameters
        ----------
        axx :  matplotlib.Axes
            Matplotlib Axes object of plot window.
        ydata :  np.ndarray | tuple | list
            Y data.
        xdata :  np.ndarray | tuple | list, optional
            X data.
        use_steps :  bool, default=False
            Use `matplotlib.pyplot.stairs` instead of matplotlib.pyplot.plot.
        **kwargs : keyword arguments
            Keywords passed to matplotlib.pyplot.plot

        See Also
        --------
        matplotlib.pyplot.plot, matplotlib.pyplot.stairs

        """
        if xdata is None:
            xdata = np.arange(len(ydata)) if self.xdata is None else self.xdata
        else:
            if isinstance(xdata[0], dt.date | dt.time | dt.datetime):
                xdata = np.asarray(xdata, dtype="datetime64")

        if np.issubdtype(xdata.dtype, np.datetime64):
            self.time_axis = True

        if self.xdata is None:
            self.xdata = xdata

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
        /,
        axx: Axes,
        **kwargs: Unpack[DrawKeys],
    ) -> None:
        """Add annotations to a subplot, before closing it.

        Parameters
        ----------
        axx :  matplotlib.Axes
            Matplotlib Axes object of plot window
        **kwargs :  Unpack[DrawKeys]
            keyword arguments, recognized are
            'kwlegend', 'title', 'xlabel', 'ylabel', 'xlim', 'ylim', 'xscale', 'yscale'.
            where dictionary kwlegend is passed to `Axes.legend`
            Default: {'fontsize': 'small', 'loc': 'best'}

        """
        # add title to image panel
        if "title" in kwargs:
            axx.set_title(kwargs["title"], fontsize="large")

        # add X & Y label
        if "xlabel" in kwargs:
            axx.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            axx.set_ylabel(kwargs["ylabel"])

        # set the limits of the X-axis & Y-axis
        if "xlim" in kwargs:
            axx.set_xlim(*kwargs["xlim"])
        if "ylim" in kwargs:
            axx.set_ylim(*kwargs["ylim"])

        # set the scale of the X & Y axis {"linear", "log", "symlog", ...}
        if "xscale" in kwargs:
            axx.set_xscale(kwargs["xscale"])
        if "yscale" in kwargs:
            axx.set_yscale(kwargs["yscale"])

        # define parameters for `Axes.legend`
        kwlegend = kwargs.get("kwlegend", {"fontsize": "small", "loc": "best"})

        # format the X-axis when it is a time-axis
        if self.time_axis:
            if abs(self.xdata[-1] - self.xdata[0]) <= np.timedelta64(1, "D"):
                plt.gcf().autofmt_xdate()
                axx.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            else:
                locator = AutoDateLocator()
                axx.xaxis.set_major_locator(locator)
                axx.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        else:
            if "xlim" in kwargs and (kwargs["xlim"][1] % 10) == 0:
                axx.set_xticks(np.linspace(0, kwargs["xlim"][1], 6, dtype=int))
            elif "xlim" in kwargs and (kwargs["xlim"][1] % 8) == 0:
                axx.set_xticks(np.linspace(0, kwargs["xlim"][1], 5, dtype=int))

        # draw legenda in figure
        if axx.get_legend_handles_labels()[1]:
            axx.legend(**kwlegend)

        # add grid lines (default settings)
        axx.grid(True)
