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
"""Definition of the monitplot class `DrawMulti`."""

from __future__ import annotations

__all__ = ["DrawMulti"]

import datetime as dt
from typing import TYPE_CHECKING, NotRequired, Self, TypedDict, Unpack

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import (
    AutoDateLocator,
    ConciseDateFormatter,
    DateFormatter,
)
from xarray import DataArray

from moniplot.tol_colors import tol_rgba

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from moniplot.lib.fig_info import FIGinfo

# - global variables -------------------------------
DEFAULT_CSET = "bright"


class HistKeys(TypedDict):
    """Define keyword arguments of method add_hist()."""

    bins: NotRequired[int]
    range: NotRequired[tuple[float, float]]
    weights: NotRequired[ArrayLike[float]]
    density: NotRequired[bool]


class DrawKeys(TypedDict):
    """Define keyword arguments of DrawMulti.draw()."""

    title: NotRequired[str]
    xlabel: NotRequired[str]
    ylabel: NotRequired[str]
    xlim: NotRequired[list[float, float]]
    ylim: NotRequired[list[float, float]]
    xscale: NotRequired[str]
    yscale: NotRequired[str]
    kwlegend: NotRequired[dict]


# - class definition -------------------------------
class DrawMulti:
    """Draw a multi-panel figure."""

    def __init__(
        self: DrawMulti,
        n_panel: int,
        *,
        one_column: bool = False,
        sharex: bool = False,
        sharey: bool = False,
    ) -> None:
        """Create DrawMuli object."""
        self._cset = tol_rgba(DEFAULT_CSET)
        self._decoration = {
            "fig_info": None,
            "kw_adjust": {},
            "sharex": sharex,
            "sharey": sharey,
            "title": [],
            "xlim": [],
            "xlabel": [],
            "ylim": [],
            "ylabel": [],
        }
        self.fig = None
        self.axxs = None
        self.show_xlabel = ()
        self.show_ylabel = ()
        self.time_axis = False

        if not 1 <= n_panel <= 9:
            raise ValueError("value out of range: 1 <= n_panel <= 9")

        self._subplots_(n_panel, one_column, sharex, sharey)

    def __enter__(self: DrawMulti) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: DrawMulti, *args: object) -> bool:
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self: DrawMulti) -> None:
        """Decorate the panels of the figure."""
        for ii, axx in enumerate(self.axxs):
            if self._decoration["xlim"]:
                axx.set_xlim(*self._decoration["xlim"])

            if self._decoration["ylim"]:
                axx.set_ylim(*self._decoration["ylim"])

            if len(self._decoration["title"]) >= ii:
                if self._decoration["sharex"]:
                    axx.set_title(
                        self._decoration["title"][ii],
                        loc="right",
                        y=1.0,
                        pad=-12,
                        fontsize="medium",
                    )
                else:
                    axx.set_title(self._decoration["title"][ii], fontsize="large")
            if self.show_xlabel[ii]:
                axx.set_xlabel(self._decoration["xlabel"][ii])
            if self.show_ylabel[ii]:
                axx.set_ylabel(self._decoration["ylabel"][ii])

        if self._decoration["fig_info"] is not None:
            self._decoration["fig_info"].draw(self.fig)

        if self._decoration["kw_adjust"]:
            if self._decoration["fig_info"] is not None and (
                n_lines := len(self._decoration["fig_info"]) > 5
            ):
                self._decoration["kw_adjust"]["top"] -= (
                    (n_lines - 5) * 0.1 / self.fig.get_figheight()
                )
            plt.subplots_adjust(**self._decoration["kw_adjust"])

    def _subplots_(
        self: DrawMulti,
        n_panel: int,
        one_column: bool,
        sharex: bool,
        sharey: bool,
    ) -> None:
        """Create a figure and a set of panels.

        Notes
        -----
        Distributes the panels as follows:
         1: X

         2: X X

         3: X X X

         4: X X
            X X

         5: X X X
            X X

         6: X X X
            X X X

         7: X X X
            X X X
            X

         8: X X X
            X X X
            X X

         9: X X X
            X X X
            X X X

        """
        if one_column:
            n_row = n_panel
            n_col = 1
            fig_size = ((10, 6), (10, 7), (10, 9), (10, 10), (10, 12))[n_panel - 1]
        else:
            n_row, n_col = (
                ((1, 1), (1, 2), (1, 3), (2, 2)) + 2 * ((2, 3),) + 3 * ((3, 3),)
            )[n_panel - 1]
            fig_size = (
                ((8, 8), (12, 6), (14, 6), (9, 9)) + 2 * ((12, 9),) + 3 * ((12, 12),)
            )[n_panel - 1]

        layout = "constrained" if not (sharex or sharey) else None
        self.fig = plt.figure(figsize=fig_size, layout=layout)
        if layout is None:
            margin = min(1.0 / (1.65 * (n_row + 1)), 0.3)
            self._decoration["kw_adjust"]["bottom"] = margin
            self._decoration["kw_adjust"]["top"] = 1 - 1.1 / self.fig.get_figheight()
            if sharex:
                self._decoration["kw_adjust"]["hspace"] = 0.05
            if sharey:
                self._decoration["kw_adjust"]["wspace"] = 0.1

        axx_arr = ()
        for ii in range(n_panel):
            axx = self.fig.add_subplot(n_row, n_col, ii + 1)
            axx_arr += (axx,)

            # decoration X-axis
            show_label = True
            if (
                sharex
                and ii < (n_row - 1) * n_col
                and not (
                    (n_panel == 5 and ii == 2)
                    or (n_panel == 7 and ii >= 4)
                    or (n_panel == 8 and ii == 5)
                )
            ):
                axx.set_xticklabels("")
                show_label = False

            self.show_xlabel += (show_label,)

            # decoration Y-axis
            show_label = True
            if sharey and ii % n_col:
                axx.set_yticklabels("")
                show_label = False

            self.show_ylabel += (show_label,)

        self.axxs = np.array(axx_arr)

    # - Public Methods ---------------------------------
    def add_caption(self: DrawMulti, text: str) -> None:
        """Add figure caption."""
        self.fig.suptitle(
            text,
            fontsize="x-large",
            linespacing=2,
            position=(0.5, 1 - 0.3 / self.fig.get_figheight()),
        )

    def add_copyright(self: DrawMulti, ipanel: int, institute: str = "SRON") -> None:
        """Display copyright statement in the lower right corner.

        Parameters
        ----------
        ipanel: int
           index of the multi-plot panel, use -1 for the last panel
        institute: str, default="SRON"
           name of the copyright owner

        """
        self.axxs[ipanel].text(
            1,
            0,
            rf" $\copyright$ {institute}",
            horizontalalignment="right",
            verticalalignment="bottom",
            rotation="vertical",
            fontsize="xx-small",
            transform=self.axxs[ipanel].transAxes,
        )

    def set_fig_info(self: DrawMulti, fig_info: FIGinfo) -> None:
        """Add fig_info box to the figure."""
        self._decoration["fig_info"] = fig_info

    def set_xlim(
        self: DrawMulti,
        left: float | None = None,
        right: float | None = None,
    ) -> None:
        """Set the X-axis view limits of all panels."""
        self._decoration["xlim"] = [left, right]

    def update_xlim(self: DrawMulti, xdata: ArrayLike) -> None:
        """Update the X-axis view limits if sharex is True."""
        xlim = [np.min(xdata), np.max(xdata)]
        if self._decoration["xlim"]:
            xlim = [
                min(self._decoration["xlim"][0], xlim[0]),
                max(self._decoration["xlim"][1], xlim[1]),
            ]
        self._decoration["xlim"] = xlim

    def set_ylim(
        self: DrawMulti,
        left: float | None = None,
        right: float | None = None,
    ) -> None:
        """Set the Y-axis view limits of all panels."""
        self._decoration["ylim"] = [left, right]

    def update_ylim(self: DrawMulti, ydata: ArrayLike) -> None:
        """Update the Y-axis view limits if sharey is True."""
        ylim = [np.min(ydata), np.max(ydata)]
        if self._decoration["ylim"]:
            ylim = [
                min(self._decoration["ylim"][0], ylim[0]),
                max(self._decoration["ylim"][1], ylim[1]),
            ]
        self._decoration["ylim"] = ylim

    def set_cset(self: DrawMulti, cname: str, cnum: int | None = None) -> None:
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

    def unset_cset(self: DrawMulti) -> None:
        """Set color set to its default."""
        self._cset = tol_rgba(DEFAULT_CSET)

    # pylint: disable=too-many-arguments
    def add_hist(
        self: DrawMulti,
        ipanel: int,
        arr: DataArray | NDArray,
        *,
        clip: tuple[float | None, float | None] | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        **kwargs: Unpack[HistKeys],
    ) -> None:
        """Draw minplot histogram.

        Parameters
        ----------
        ipanel: int
            index of the multi-plot panel
        arr: DataArray | NDArray
            data-array
        clip: tuple[float | None, float | None], optional
            clip values of data-array using numpy.clip
        title: str, optional
            text displayed above the panel
        xlabel: str, optional
            text displayed below the X-axis
        **kwargs :  Unpack[HistKeys]
            keyword arguments, see numpy.histogram

        See Also
        --------
        numpy.histogram

        """
        axx = self.axxs[ipanel]
        self._decoration["title"].append(title)
        self._decoration["xlabel"].append("values" if xlabel is None else xlabel)
        self._decoration["ylabel"].append(
            "density" if kwargs.get("density") else "number"
        )
        if isinstance(arr, DataArray):
            data = np.ravel(arr.values)
            if "long_name" in arr.attrs:
                self._decoration["xlabel"][ipanel] = arr.attrs["long_name"]
            if "units" in arr.attrs:
                self._decoration["xlabel"][ipanel] += f" [{arr.attrs['units']}]"
        else:
            data = np.ravel(arr)

        if clip is not None:
            data = np.clip(data, *clip)
            vrange = (
                data.min() if clip[0] is None else clip[0],
                data.max() if clip[1] is None else clip[1],
            )
        else:
            vrange = (data.min(), data.max())

        hist, edges = np.histogram(data, range=vrange, **kwargs)
        if self._decoration["sharex"]:
            self.update_xlim(edges)

        if self._decoration["sharey"]:
            self.update_ylim(hist)

        # draw histogram
        if len(hist) > 24:
            axx.stairs(
                hist,
                edges,
                edgecolor="#4477AA",
                facecolor="#77AADD",
                fill=True,
                linewidth=1.5,
            )
            axx.grid(which="major", color="#AAAAAA", linestyle="--")
        else:
            axx.bar(
                edges[:-1],
                hist,
                width=np.diff(edges),
                align="edge",
                edgecolor="#4477AA",
                facecolor="#77AADD",
                linewidth=1.5,
            )
            axx.grid(which="major", axis="y", color="#AAAAAA", linestyle="--")

    def add_line(
        self: DrawMulti,
        ipanel: int,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        use_steps: bool = False,
        **kwargs: int,
    ) -> None:
        """Add a line or scatter-points to a subplot.

        Parameters
        ----------
        ipanel: int
            index of the multi-plot panel
        xdata :  ArrayLike
            X data.
        ydata :  ArrayLike
            Y data.
        use_steps :  bool, default=False
            Use `matplotlib.pyplot.stairs` instead of matplotlib.pyplot.plot.
        **kwargs : keyword arguments
            Keywords passed to matplotlib.pyplot.plot

        See Also
        --------
        matplotlib.pyplot.plot, matplotlib.pyplot.stairs

        """
        axx = self.axxs[ipanel]
        if isinstance(xdata[0], dt.date | dt.time | dt.datetime):
            xdata = np.asarray(xdata, dtype="datetime64")
            self.time_axis = True

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
        self: DrawMulti,
        ipanel: int,
        **kwargs: Unpack[DrawKeys],
    ) -> None:
        """Add annotations to a subplot, before closing it.

        Parameters
        ----------
        ipanel: int
            index of the multi-plot panel
        **kwargs :  Unpack[DrawKeys]
            keyword arguments, recognized are
            'kwlegend', 'title', 'xlabel', 'ylabel', 'xscale', 'yscale'.
            where dictionary kwlegend is passed to `Axes.legend`
            Default: {'fontsize': 'small', 'loc': 'best'}

        """
        axx = self.axxs[ipanel]
        self._decoration["title"].append(kwargs.get("title"))

        # add X & Y label
        self._decoration["xlabel"].append(kwargs.get("xlabel"))
        self._decoration["ylabel"].append(kwargs.get("ylabel"))

        # set the scale of the X & Y axis {"linear", "log", "symlog", ...}
        if "xscale" in kwargs:
            axx.set_xscale(kwargs["xscale"])
        if "yscale" in kwargs:
            axx.set_yscale(kwargs["yscale"])

        # format the X-axis when it is a time-axis
        if self.time_axis:
            if abs(self.xdata[-1] - self.xdata[0]) <= np.timedelta64(1, "D"):
                plt.gcf().autofmt_xdate()
                axx.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            else:
                locator = AutoDateLocator()
                axx.xaxis.set_major_locator(locator)
                axx.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        elif self._decoration["xlim"]:
            if (self._decoration["xlim"][1] % 10) == 0:
                axx.set_xticks(np.linspace(0, kwargs["xlim"][1], 6, dtype=int))
            elif (self._decoration["xlim"][1] % 8) == 0:
                axx.set_xticks(np.linspace(0, kwargs["xlim"][1], 5, dtype=int))

        # draw legenda in figure
        if axx.get_legend_handles_labels()[1]:
            kwlegend = kwargs.get("kwlegend", {"fontsize": "small", "loc": "best"})
            axx.legend(**kwlegend)

        # add grid lines (default settings)
        axx.grid(which="major", color="#AAAAAA", linestyle="--")
