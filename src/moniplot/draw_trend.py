#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2024 SRON - Netherlands Institute for Space Research
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
"""Definition of the monitplot class `DrawTrend`."""

from __future__ import annotations

__all__ = ["DrawTrend"]

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from .tol_colors import tol_cset

if TYPE_CHECKING:
    from collections.abc import Iterable

    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# - global variables -------------------------------


# - class definition -------------------------------
class DrawTrend:
    """Display trends of measurement data and/or housekeeping data.

    Notes
    -----
    The input data should be a xarray.DataArray with atleast attributes providing
    'long_name' and 'units'.

    Examples
    --------
    Generate a figure with four trend plots.

    >>> report = MONplot("test_monplot.pdf", "This is an example figure")
    >>> report.set_institute("SRON")
    >>> plot = DrawTrend()
    >>> fig, axx = plot.subplots(4)
    >>> dist2.attrs["long_name"] = "Detector temperature"
    >>> dist2.attrs["units"] = "K"
    >>> plot.draw_hk(axx[0], dist2)
    >>> dist2.attrs["long_name"] = "ICU 1.2V bus voltage"
    >>> dist2.attrs["units"] = "Volt"
    >>> plot.draw_hk(axx[1], dist2)
    >>> dist2.attrs["long_name"] = "ICU 4V bus current"
    >>> dist2.attrs["units"] = "\u03bcA"
    >>> plot.draw_hk(axx[2], dist2)
    >>> dist2.attrs["long_name"] = "Heater duty cycle"
    >>> dist2.attrs["units"] = "%"
    >>> plot.draw_hk(axx[3], dist2)
    >>> report.add_copyright(axx[3])
    >>> axx[-1].set_xlabel('orbit')
    >>> report.close_this_page(fig, None)
    >>> report.close()

    """

    @staticmethod
    def subplots(npanels: int) -> tuple[Figure, list[Axes, ...]]:
        """Create a figure and a set of subplots for trend-plots."""
        figsize = (10.0, 1 + (npanels + 1) * 1.5)
        fig, axarr = plt.subplots(npanels, figsize=figsize, sharex=True)
        if npanels == 1:
            axarr = [axarr]
        margin = min(1.0 / (1.65 * (npanels + 1)), 0.25)
        fig.subplots_adjust(bottom=margin, top=1 - margin, hspace=0.05)
        for ii in range(npanels - 1):
            for xtl in axarr[ii].get_xticklabels():
                xtl.set_visible(False)

        return fig, axarr

    @property
    def blank_legend_handle(self: DrawTrend) -> Rectangle:
        """Show only label in a legend entry, no handle.

        See Also
        --------
        matplotlib.pyplot.legend : Place a legend on the Axes.

        """
        return Rectangle((0, 0), 0, 0, fill=False, edgecolor="none", visible=False)

    @staticmethod
    def adjust_ylim(
        data: np.ndarray | Iterable,
        err1: np.ndarray | Iterable | None,
        err2: np.ndarray | Iterable | None,
        vperc: list[int, int] | None,
        vrange_last_orbits: int = 0,
    ) -> tuple[float, float]:
        """Set minimum and maximum values of ylim.

        Parameters
        ----------
        data :  array_like
            Values of the data to be plotted
        err1 :  array_like
            Values of the data minus its uncertainty, None if without uncertainty
        err2 :  array_like
            Values of the data plus its uncertainty, None if without uncertainty
        vperc : list[int, int] | None
            Limit the data range to the given percentiles
        vrange_last_orbits: int, default=0
            Use only data of the last N orbits, ignored when vrange_last_orbits <= 0

        Returns
        -------
        tuple of floats
            Return the limits of the Y-coordinate

        """
        if err1 is not None and err2 is not None:
            indx = np.isfinite(err1) & np.isfinite(err2)
            if np.all(~indx):
                ylim = [0.0, 0.0]
            elif np.sum(indx) > vrange_last_orbits > 0:
                ni = vrange_last_orbits
                ylim = [
                    min(err1[indx][0:ni].min(), err1[indx][-ni:].min()),
                    max(err2[indx][0:ni].max(), err2[indx][-ni:].max()),
                ]
            elif isinstance(vperc, list) and len(vperc) == 2:
                ylim = [
                    np.percentile(err1[indx], vperc[0]),
                    np.percentile(err2[indx], vperc[1]),
                ]
            else:
                ylim = [err1[indx].min(), err2[indx].max()]
            factor = 10
        else:
            indx = np.isfinite(data)
            if np.all(~indx):
                ylim = [0.0, 0.0]
            elif np.sum(indx) > vrange_last_orbits > 0:
                ni = vrange_last_orbits
                ylim = [
                    min(data[indx][0:ni].min(), data[indx][-ni:].min()),
                    max(data[indx][0:ni].max(), data[indx][-ni:].max()),
                ]
            elif isinstance(vperc, list) and len(vperc) == 2:
                ylim = np.percentile(data[indx], vperc)
            else:
                ylim = [data[indx].min(), data[indx].max()]
            factor = 5

        if ylim[0] == ylim[1]:
            delta = 0.01 if ylim[0] == 0 else ylim[0] / 20
        else:
            delta = (ylim[1] - ylim[0]) / factor

        return float(ylim[0] - delta), float(ylim[1] + delta)

    @staticmethod
    def adjust_units(zunit: str) -> str:
        """Adjust units: electron to 'e' and Volt to 'V'.

        Parameters
        ----------
        zunit :  str
            Units of the image data

        Returns
        -------
        str
            Units with consistent abbreviation of electron(s) and Volt

        """
        if zunit is None or zunit == "1":
            return "1"

        if zunit.find("electron") >= 0:
            zunit = zunit.replace("electron", "e")
        if zunit.find("Volt") >= 0:
            zunit = zunit.replace("Volt", "V")
        if zunit.find(".s-1") >= 0:
            zunit = zunit.replace(".s-1", " s$^{-1}$")
        if zunit.find("degC") >= 0:
            zunit = zunit.replace("degC", "\u00b0C")

        return zunit

    @staticmethod
    def get_gap_list(xdata: np.ndarray) -> tuple:
        """Identify data gaps for data where xdata = offs + N * xstep.

        Parameters
        ----------
        xdata :  numpy.ndarray
            Independent variable where the data is measured

        Returns
        -------
        list
            Indices to xdata where np.diff(xdata) greater than xstep

        """
        if issubclass(xdata.dtype.type, np.datetime64):
            diff_data = np.rint(np.diff(xdata).astype(float) / 1e9).astype(int)
        elif issubclass(xdata.dtype.type, Integral):
            diff_data = np.diff(xdata)
        else:
            return ()

        uvals, counts = np.unique(diff_data, return_counts=True)
        if counts.size > 1 and counts.max() / xdata.size > 0.5:
            xstep = uvals[counts.argmax()]
            return tuple(i for i in (diff_data > xstep).nonzero()[0])

        return ()

    def decoration(self: DrawTrend, xarr: xr.DataArray) -> dataclass:
        """Return decoration parameters for trend-plots.

        Notes
        -----
        Please make sure that your DataArray contains attributes: 'long_name'
        and 'units'. Then if 'long_name' contains temperature, voltage, current and
        duty-cycle ylabel & legend will be defined.


        Returns
        -------
        dataclass
         - fcolor: fill-color used in axx.fill_between()
         - lcolor: line-color used in axx.plot() or axx.step(), overwitten by '_color'
         - ylabel: text along the y-axis, overwritten by '_ylabel'
         - legend: text displayed with axx.legend(), overwritten by 'legend'

        """
        mytitle = xarr.attrs.get("long_name", "")
        if isinstance(mytitle, bytes):
            mytitle = mytitle.decode()

        units = xarr.attrs.get("units", "1")
        if isinstance(units, bytes):
            units = units.decode()
        units = self.adjust_units(units)

        line_cset = tol_cset("bright")
        l_color = line_cset.cyan
        match units:
            case "K" | "mK" | "degC" | "\u00b0C":
                if (ii := mytitle.find(" temperature")) > 0:
                    mytitle = mytitle[:ii]
                mylabel = f"temperature [{units}]"
                l_color = line_cset.purple
            case "V" | "pV" | "nV" | "\u03bcV" | "mV" | "Volt":
                if (ii := mytitle.find(" voltage")) > 0:
                    mytitle = mytitle[:ii]
                mylabel = f"voltage [{units}]"
                l_color = line_cset.yellow
            case "A" | "pA" | "nA" | "\u03bcA" | "mA":
                if (ii := mytitle.find(" current")) > 0:
                    mytitle = mytitle[:ii]
                mylabel = f"current [{units}]"
                l_color = line_cset.green
            case "e" | "1" if mytitle.find("noise") > 0:
                mytitle = ""
                mylabel = "value" if units == "1" else f"value [{units}]"
            case "W" if mytitle.find(" power") > 0:
                mytitle = mytitle[: mytitle.find(" power")]
                mylabel = f"power [{units}]"
                l_color = line_cset.blue
            case "%" if mytitle.find(" duty") > 0:
                mytitle = mytitle[: mytitle.find(" duty")]
                mylabel = f"duty cycle [{units}]"
                l_color = line_cset.red
            case _:
                if "legend" in xarr.attrs:
                    mylabel = "count" if units == "1" else f"value [{units}]"
                else:
                    mylabel = mytitle if units == "1" else f"{mytitle} [{units}]"
                mytitle = ""

        # redefine mylabel or mytitle when attributes of xarr are set
        mylabel = xarr.attrs.get("_ylabel", mylabel)
        mytitle = xarr.attrs.get("legend", mytitle)

        # set the fill-color according to the line-color
        fill_cset = tol_cset("plain")
        l_color = xarr.attrs.get("_color", l_color)
        match l_color:
            case "none":
                l_color = None
                f_color = None
            case line_cset.blue:
                f_color = fill_cset.blue
            case line_cset.cyan:
                f_color = fill_cset.cyan
            case line_cset.green:
                f_color = fill_cset.green
            case line_cset.yellow:
                f_color = fill_cset.yellow
            case line_cset.red:
                f_color = fill_cset.red
            case line_cset.purple:
                f_color = fill_cset.purple
            case _:
                f_color = fill_cset.grey

        @dataclass(frozen=True)
        class DecoFig:
            """Define figure decorations."""

            # fill-color
            fcolor: str = f_color
            # line-color
            lcolor: str = l_color
            # suggestion for the ylabel
            ylabel: str = mylabel
            # suggestion for the figure legend entry
            legend: str = mytitle

        return DecoFig()

    def draw(
        self: DrawTrend,
        axx: Axes,
        xarr: xr.DataArray,
        scatter: bool = False,
    ) -> None:
        """Add a subplot for measurement data.

        Parameters
        ----------
        axx :  matplotlib.Axes
            Matplotlib Axes object of the current panel
        xarr :  xarray.DataArray
            Object holding measurement data and attributes
            Dimension must be 'orbit', 'hours' or 'time'.
        scatter: bool, default=False
            Make a scatter plot

        """
        # derive line-decoration from attributes of the DataArray
        deco_fig = self.decoration(xarr)

        # define xdata and determine gap_list (always at least one element!)
        isel = np.s_[:]
        if "orbit" in xarr.coords:
            xdata = xarr.coords["orbit"].values
            gap_list = self.get_gap_list(xdata)
        elif "hours" in xarr.coords:
            xdata = xarr.coords["hours"].values
            isel = np.s_[0, :]
            gap_list = self.get_gap_list(np.round(3600 * xdata).astype(int))
        else:
            xdata = xarr.coords["time"].values
            gap_list = self.get_gap_list(xdata)
        gap_list += (xdata.size - 1,)

        # define avg, err1, err2
        # check if xarr.values is a structured array:
        #    xarr.values.dtype.names is None
        # check if xarr contains quality data
        # check if err1 and err2 are present
        if xarr.values.dtype.names is None:
            avg = xarr.values[isel]
            err1 = err2 = None
        else:
            avg = xarr.values["mean"][isel]
            err1 = xarr.values["err1"][isel]
            err2 = xarr.values["err2"][isel]

        ii = 0
        for jj in gap_list:
            isel = np.s_[ii : jj + 1]
            if err1 is not None:
                axx.fill_between(
                    xdata[isel],
                    err1[isel],
                    err2[isel],
                    step="post",
                    linewidth=0,
                    facecolor=deco_fig.fcolor,
                )
                axx.step(
                    np.append(xdata[isel], xdata[jj]),
                    np.append(avg[isel], avg[jj]),
                    where="post",
                    linewidth=1.5,
                    color=deco_fig.lcolor,
                )
            elif scatter:
                axx.plot(
                    xdata[isel],
                    avg[isel],
                    marker=".",
                    linestyle="",
                    color=deco_fig.lcolor,
                )
            else:
                axx.plot(xdata[isel], avg[isel], linewidth=1.5, color=deco_fig.lcolor)
            ii = jj + 1

        # adjust data X-coordinate
        if "hours" in xarr.coords:
            axx.xaxis.set_major_locator(MultipleLocator(3))
            axx.xaxis.set_minor_locator(MultipleLocator(1))
        elif "time" in xarr.coords:
            locator = AutoDateLocator()
            axx.xaxis.set_major_locator(locator)
            axx.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        else:
            axx.xaxis.set_minor_locator(AutoMinorLocator())
        axx.set_xlim(xdata[0], xdata[-1])

        # adjust data X-coordinate
        axx.locator_params(axis="y", nbins=5)
        if "orbit" in xarr.coords:
            axx.set_ylim(*self.adjust_ylim(avg, err1, err2, None, -1))

        axx.set_ylabel(deco_fig.ylabel)
        axx.grid(True)

        # add legend with name of dataset inside current subplots
        if deco_fig.legend:
            legend = axx.legend(
                [self.blank_legend_handle], [deco_fig.legend], loc="upper left"
            )
            legend.draw_frame(False)

    def draw_hk(
        self: DrawTrend,
        axx: Axes,
        xarr: xr.DataArray,
        vperc: list[int, int] | None = None,
        vrange_last_orbits: int = -1,
    ) -> None:
        """Add a subplot for housekeeping data.

        Parameters
        ----------
        axx :  matplotlib.Axes
            Matplotlib Axes object of the current panel
        xarr :  xarray.DataArray
            Object holding housekeeping data and attributes.
            Dimension must be 'orbit', 'hours' or 'time'.
        vperc :  list | None, optional
            Reject outliers before determining vrange
            (neglected when vrange_last_orbits is used)
        vrange_last_orbits :  int, default=-1
            Use the last N orbits to determine vrange (orbit coordinate only)

        """
        # derive line-decoration from attributes of the DataArray
        deco_fig = self.decoration(xarr)

        # define xdata and determine gap_list (always one element!)
        isel = np.s_[:]
        if "time" in xarr.coords:
            xdata = xarr.coords["time"].values
            if "orbit" in xarr.coords:
                gap_list = self.get_gap_list(xarr.coords["orbit"].values)
            else:
                gap_list = self.get_gap_list(xdata)
        elif "hours" in xarr.coords:
            xdata = xarr.coords["hours"].values
            isel = np.s_[0, :]
            gap_list = self.get_gap_list(np.round(3600 * xdata).astype(int))
        else:
            xdata = xarr.coords["orbit"].values
            gap_list = self.get_gap_list(xdata)

        # check if we have any data to work with
        if xdata.size == 0:
            print("WARNING: can not draw a figure without data")
            return
        gap_list += (xdata.size - 1,)

        # define avg, err1, err2
        if xarr.values.dtype.names is None:
            avg = xarr.values[isel]
            err1 = err2 = None
        else:
            avg = xarr.values["mean"][isel]
            err1 = xarr.values["err1"][isel]
            err2 = xarr.values["err2"][isel]

        # plot data
        ii = 0
        # print(f"gap_list: {gap_list}")
        for jj in gap_list:
            isel = np.s_[ii : jj + 1]
            if err1 is not None:
                axx.fill_between(
                    xdata[isel],
                    err1[isel],
                    err2[isel],
                    step="post",
                    linewidth=0,
                    facecolor=deco_fig.fcolor,
                )
                axx.step(
                    np.append(xdata[isel], xdata[jj]),
                    np.append(avg[isel], avg[jj]),
                    where="post",
                    linewidth=1.5,
                    color=deco_fig.lcolor,
                )
            else:
                axx.plot(xdata[isel], avg[isel], linewidth=1.5, color=deco_fig.lcolor)
            ii = jj + 1

        # adjust data X-coordinate
        if "hours" in xarr.coords:
            axx.xaxis.set_major_locator(MultipleLocator(3))
            axx.xaxis.set_minor_locator(MultipleLocator(1))
        elif "time" in xarr.coords:
            locator = AutoDateLocator()
            axx.xaxis.set_major_locator(locator)
            axx.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        else:
            axx.xaxis.set_minor_locator(AutoMinorLocator())
        axx.set_xlim(xdata[0], xdata[-1])

        # adjust data Y-coordinate
        axx.locator_params(axis="y", nbins=4)
        if "orbit" in xarr.coords:
            axx.set_ylim(*self.adjust_ylim(avg, err1, err2, vperc, vrange_last_orbits))
        else:
            axx.set_ylim(*self.adjust_ylim(avg, err1, err2, vperc))
        axx.set_ylabel(deco_fig.ylabel)
        axx.grid(True)

        # add legend with name of dataset inside current subplots
        legend = axx.legend(
            [self.blank_legend_handle],
            [deco_fig.legend],
            loc="upper left",
            fontsize="large",
        )
        legend.draw_frame(False)

    def masked_hk(
        self: DrawTrend, axx: Axes, xarr: xr.DataArray, mask: np.ndarray
    ) -> None:
        """Show where housekeeping data is available, but not shown.

        Parameters
        ----------
        axx :  matplotlib.Axes
            Matplotlib Axes object of the current panel
        xarr :  xarray.DataArray
            Object holding housekeeping data and attributes.
            Dimension must be 'orbit', 'hours' or 'time'.
        mask :  np.ndarray
            mask array of data which is not shown in the figure

        """
        if mask is None or np.sum(mask) == 0:
            return

        # derive line-decoration from attributes of the DataArray
        deco_fig = self.decoration(xarr)

        poly_bgn = (np.diff(mask.astype(int)) == 1).nonzero()[0] + 1
        poly_end = (np.diff(mask.astype(int)) == -1).nonzero()[0]
        if poly_end.size > poly_bgn.size:
            poly_bgn = np.concatenate(([0], poly_bgn))
        if poly_bgn.size > poly_end.size:
            poly_end = np.concatenate((poly_end, [-1]))

        ylim = axx.get_ylim()
        for ni, nj in zip(poly_bgn, poly_end, strict=True):
            tt0 = xarr.coords["time"].values[ni]
            tt0 = (tt0 - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
            tt1 = xarr.coords["time"].values[nj]
            tt1 = (tt1 - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
            verts = [(tt0, ylim[0]), (tt0, ylim[1]), (tt1, ylim[1]), (tt1, ylim[0])]
            poly = Polygon(verts, facecolor=deco_fig.lcolor, edgecolor=None)
            axx.add_patch(poly)
