#
# This file is part of Python package: `moniplot`
#
#     https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2025 SRON
#    All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from .tol_colors import tol_cset, tol_rgba

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
            "mode": [],
            "fig_info": None,
            "kw_adjust": {},
            "sharex": sharex,
            "sharey": sharey,
            "title": n_panel * [None],
            "xlabel": n_panel * [None],
            "ylabel": n_panel * [None],
            "xlim": [],
            "ylim": [],
        }
        self.fig = None
        self.axxs = None
        self.show_xlabel = ()
        self.show_ylabel = ()
        self.time_axis = False

        if not 1 <= n_panel <= 9:
            raise ValueError("value out of range: 1 <= n_panel <= 9")

        self.__subplots__(n_panel, one_column, sharex, sharey)

    def __enter__(self: DrawMulti) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: DrawMulti, *args: object) -> bool:
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def __decorate__(self: DrawMulti) -> None:
        """Decorate the panels of the figure."""
        for ii, axx in enumerate(self.axxs):
            if self._decoration["sharex"] and self._decoration["xlim"]:
                axx.set_xlim(*self._decoration["xlim"])

            if self._decoration["sharey"] and self._decoration["ylim"]:
                axx.set_ylim(*self._decoration["ylim"])

            # add an title to each panel
            if self._decoration["title"][ii] is not None:
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

            # add labels along the axis
            if self.show_xlabel[ii] and self._decoration["xlabel"][ii] is not None:
                axx.set_xlabel(self._decoration["xlabel"][ii])
            if self.show_ylabel[ii] and self._decoration["ylabel"][ii] is not None:
                axx.set_ylabel(self._decoration["ylabel"][ii])

        # add fig_info box
        if self._decoration["fig_info"] is not None:
            self._decoration["fig_info"].draw(self.fig)

        # adjust the white space
        if self._decoration["kw_adjust"]:
            if self._decoration["fig_info"] is not None and (
                (n_lines := len(self._decoration["fig_info"])) > 5
            ):
                self._decoration["kw_adjust"]["top"] -= (
                    (n_lines - 5) * 0.1 / self.fig.get_figheight()
                )
            plt.subplots_adjust(**self._decoration["kw_adjust"])

    def _xlim_update_(self: DrawMulti, xdata: ArrayLike) -> None:
        """Update the X-axis view limits if sharex is True."""
        if not self._decoration["sharex"]:
            return

        xlim = [np.min(xdata), np.max(xdata)]
        if self._decoration["xlim"]:
            xlim = [
                min(self._decoration["xlim"][0], xlim[0]),
                max(self._decoration["xlim"][1], xlim[1]),
            ]
        self._decoration["xlim"] = xlim

    def _ylim_update_(self: DrawMulti, ydata: ArrayLike) -> None:
        """Update the Y-axis view limits if sharey is True."""
        if not self._decoration["sharey"]:
            return

        ylim = [np.min(ydata), np.max(ydata)]
        if self._decoration["ylim"]:
            ylim = [
                min(self._decoration["ylim"][0], ylim[0]),
                max(self._decoration["ylim"][1], ylim[1]),
            ]
        self._decoration["ylim"] = ylim

    def __subplots__(
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

        self.fig = plt.figure(figsize=fig_size)
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
    def close(self: DrawMulti) -> None:
        """Finalize all panels."""
        self.__decorate__()

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

    def add_fig_info(self: DrawMulti, fig_info: FIGinfo) -> None:
        """Add fig_info box to the figure."""
        self._decoration["fig_info"] = fig_info

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

    def set_caption(self: DrawMulti, text: str) -> None:
        """Add figure caption."""
        self.fig.suptitle(
            text,
            fontsize="x-large" if self.fig.get_figheight() < 10 else "xx-large",
            linespacing=2,
            position=(0.5, 1 - 0.3 / self.fig.get_figheight()),
        )

    def set_title(self: DrawMulti, title: str, ipanel: int | None = None) -> None:
        """Set title of the current panel."""
        n_panel = len(self.axxs)
        if ipanel is None:
            self._decoration["title"] = n_panel * [title]
        else:
            if ipanel >= n_panel:
                raise ValueError("Invalid ipanel larger than number of panels")
            self._decoration["title"][ipanel] = title

    def set_xlabel(self: DrawMulti, xlabel: str, ipanel: int | None = None) -> None:
        """Set xlabel of the current panel."""
        n_panel = len(self.axxs)
        if ipanel is None:
            self._decoration["xlabel"] = n_panel * [xlabel]
        else:
            if ipanel >= n_panel:
                raise ValueError("Invalid ipanel larger than number of panels")
            self._decoration["xlabel"][ipanel] = xlabel

    def set_ylabel(self: DrawMulti, ylabel: str, ipanel: int | None = None) -> None:
        """Set ylabel of the current panel."""
        n_panel = len(self.axxs)
        if ipanel is None:
            self._decoration["ylabel"] = n_panel * [ylabel]
        else:
            if ipanel >= n_panel:
                raise ValueError("Invalid ipanel larger than number of panels")
            self._decoration["ylabel"][ipanel] = ylabel

    def set_xlim(
        self: DrawMulti,
        left: float | None = None,
        right: float | None = None,
        *,
        ipanel: int | None = None,
    ) -> None:
        """Set the X-axis view limits of all panels."""
        if ipanel is None:
            self._decoration["xlim"] = [left, right]
        else:
            self.axxs[ipanel].set_ylim(left, right)

    def set_ylim(
        self: DrawMulti,
        left: float | None = None,
        right: float | None = None,
        *,
        ipanel: int | None = None,
    ) -> None:
        """Set the Y-axis view limits of all panels when ipanel is None."""
        if ipanel is None:
            self._decoration["ylim"] = [left, right]
        else:
            self.axxs[ipanel].set_ylim(left, right)

    def hist(
        self: DrawMulti,
        ipanel: int,
        arr: DataArray | NDArray,
        *,
        clip: tuple[float | None, float | None] | None = None,
        **kwargs: Unpack[HistKeys],
    ) -> None:
        """Draw histogram.

        Parameters
        ----------
        ipanel: int
            index of the multi-plot panel
        arr: DataArray | NDArray
            data-array
        clip: tuple[float | None, float | None], optional
            clip values of data-array using numpy.clip
        **kwargs :  Unpack[HistKeys]
            keyword arguments, see numpy.histogram

        See Also
        --------
        numpy.histogram

        """
        self._decoration["mode"].append("HIST")
        self.set_xlabel("values")
        self.set_ylabel("density" if kwargs.get("density") else "number")
        if isinstance(arr, DataArray):
            data = arr.values.ravel()
            if "long_name" in arr.attrs:
                xlabel = arr.attrs["long_name"]
                if "units" in arr.attrs:
                    xlabel = f"{xlabel} [{arr.attrs['units']}]"
                self.set_xlabel(xlabel, ipanel=ipanel)
        else:
            data = arr.ravel()

        if clip is not None:
            data = np.clip(data, *clip)
            vrange = (
                data.min() if clip[0] is None else clip[0],
                data.max() if clip[1] is None else clip[1],
            )
        else:
            vrange = (data.min(), data.max())

        hist, edges = np.histogram(data, range=vrange, **kwargs)
        self._xlim_update_(edges)
        self._ylim_update_(hist)

        # draw histogram
        axx = self.axxs[ipanel]
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
        """Add X/Y data as line and/or markers to a subplot.

        Parameters
        ----------
        ipanel: int
            index of the multi-plot panel
        xdata :  ArrayLike
            X data.
        ydata :  ArrayLike
            Y data.
        use_steps :  bool, default=False
            Use `matplotlib.pyplot.stairs` instead of matplotlib.pyplot.plot
        **kwargs : keyword arguments
            Keywords passed to matplotlib.pyplot.plot or matplotlib.pyplot.stairs

        See Also
        --------
        matplotlib.pyplot.plot, matplotlib.pyplot.stairs

        """
        if len(self._decoration["mode"]) == ipanel:
            self._decoration["mode"].append("LPLOT")
        if isinstance(xdata[0], dt.date | dt.time | dt.datetime):
            xdata = np.asarray(xdata, dtype="datetime64")
            self.time_axis = True

        self._xlim_update_(xdata)
        self._ylim_update_(ydata)

        axx = self.axxs[ipanel]
        if not (axx.dataLim.mutatedx() or axx.dataLim.mutatedy()):
            axx.set_prop_cycle(color=self._cset)

        # draw line in subplot
        if use_steps:
            edges = np.append(xdata, xdata[-1])
            values = np.append(ydata, ydata[-1])
            axx.stairs(values, edges, **kwargs)
        else:
            axx.plot(xdata, ydata, **kwargs)

    def end_line(
        self: DrawMulti,
        ipanel: int,
        **kwargs: Unpack[DrawKeys],
    ) -> None:
        """Finalize line-plot.

        Parameters
        ----------
        ipanel: int
            index of the multi-plot panel
        **kwargs :  Unpack[DrawKeys]
            keyword arguments, recognized are
            'kwlegend', 'xscale', 'yscale'.
            where dictionary kwlegend is passed to `Axes.legend`
            Default: {'fontsize': 'small', 'loc': 'best'}

        """
        if self._decoration["mode"][ipanel] != "LPLOT":
            return

        # set the scale of the X & Y axis {"linear", "log", "symlog", ...}
        axx = self.axxs[ipanel]
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

        # draw legenda in figure
        if axx.get_legend_handles_labels()[1]:
            kwlegend = kwargs.get("kwlegend", {"fontsize": "small", "loc": "best"})
            axx.legend(**kwlegend)

        # add grid lines (default settings)
        axx.grid(which="major", color="#AAAAAA", linestyle="--")

    def yperc(
        self: DrawMulti,
        ipanel: int,
        xdata: ArrayLike,
        ydata: ArrayLike,
        bins: ArrayLike,
        *,
        min_in_bins: int = 25,
        percentiles: tuple[int, ...] | None = None,
    ) -> None:
        """Add distribution of points per bin to a subplot.

        Parameters
        ----------
        ipanel: int
           index of the multi-plot panel
        xdata :  ArrayLike
           X-coordinates of the sample points, dtype=float | 'datetime64[s]'
        ydata :  ArrayLike
           Y-coordinates of the sample points, NaN values are discarded
        bins :  ArrayLike
           Array of bins.  It has to be 1-dimensional and monotonic, see np.digitize.
        min_in_bins :  int, default=25
           minimum number of data-samples per date-bin
        percentiles :  tuple[int, ...], default=[50]
           you may define 1, 3 or 5 percentiles

        See Also
        --------
        numpy.digitize
        matplotlib.pyplot.steps

        """
        self._decoration["mode"].append("YPERC")
        if isinstance(xdata[0], dt.date | dt.time | dt.datetime):
            xdata = np.asarray(xdata, dtype="datetime64")
            self.time_axis = True

        # check percentiles
        if percentiles is None:
            val_perc = [50]
        elif len(percentiles) not in (1, 3, 5):
            raise KeyError("number of percentiles should be odd")
        else:
            val_perc = percentiles

        mask = np.isfinite(ydata)
        xarr = xdata[mask].copy()
        yarr = ydata[mask].copy()
        xbinned = np.asarray(bins)
        ybinned = np.full((len(val_perc), xbinned.size), np.nan)

        # collect data per bin
        indx = np.searchsorted(xbinned, xarr, side="right")
        for ii in range(xbinned.size):
            mask = indx == ii + 1
            if mask.sum() < min_in_bins:
                continue
            ybinned[:, ii] = np.nanpercentile(yarr[mask], val_perc)
        ybinned[:, -1] = ybinned[:, -2]

        # close data-gaps (date-bins without data)
        i_gaps = np.isnan(ybinned).all(axis=0).nonzero()[0]
        if i_gaps.size > 0:
            offs = 0
            i_diff = np.append([0], np.diff(i_gaps))
            for i_d, i_g in zip(i_diff, i_gaps, strict=True):
                if i_d == 1:
                    continue
                i_g += offs
                xbinned = np.insert(xbinned, i_g, xbinned[i_g])
                ybinned = np.insert(ybinned, i_g, ybinned[:, i_g - 1], axis=1)
                offs += 1

        # create plot
        blue = tol_cset("bright").blue
        plain_blue = tol_cset("light").light_blue
        grey = tol_cset("plain").grey

        # draw pannel
        axx = self.axxs[ipanel]
        if len(val_perc) in (3, 5):
            axx.fill_between(
                xbinned,
                ybinned[0, :],
                ybinned[-1, :],
                step="post",
                color=grey if len(val_perc) == 5 else plain_blue,
                label=None if ipanel else f"[{val_perc[0]}%, {val_perc[-1]}%]",
            )
            if len(val_perc) != 3:
                axx.fill_between(
                    xbinned,
                    ybinned[1, :],
                    ybinned[-2, :],
                    step="post",
                    color=plain_blue,
                    label=None if ipanel else f"[{val_perc[1]}%, {val_perc[-2]}%]",
                )
        axx.step(
            xbinned,
            ybinned[len(val_perc) // 2, :],
            where="post",
            linewidth=2,
            color=blue,
            label=None if ipanel else "median",
        )
        axx.grid(which="major", color="#AAAAAA", linestyle="--")
