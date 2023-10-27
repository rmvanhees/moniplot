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
This module holds `fig_data_to_xarr` and `fig_qdata_to_xarr`.

These functions are used by resp. `draw_signal` and `draw_quality`.
"""
from __future__ import annotations

__all__ = [
    "adjust_img_ticks",
    "fig_data_to_xarr",
    "fig_draw_panels",
    "fig_qdata_to_xarr",
]

import warnings
from math import log10
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from matplotlib.ticker import AutoMinorLocator

from .image_to_xarray import data_to_xr
from .tol_colors import tol_cmap, tol_cset

if TYPE_CHECKING:
    from matplotlib.axis import Axes

# - global parameters ------------------------------
CSET = tol_cset("bright")


# - local functions --------------------------------
def adjust_zunit(zunits: str, vmin: float, vmax: float) -> tuple[int, str | None]:
    """Adjust data range and its units.

    Units which are renamed: electron to `e` and Volt to `V`
    Scale data-range between -1000 and 1000.

    Parameters
    ----------
    zunits :  str
       Units of the image data
    vmin, vmax : float
        image-data range

    Returns
    -------
    tuple
        dscale, zunits
    """
    if zunits is None or zunits == "1":
        return 1, zunits

    zunits = zunits.replace("electron", "e")
    zunits = zunits.replace("Volt", "V")
    zunits = zunits.replace("-1", "$^{-1}$")
    zunits = zunits.replace("-2", "$^{-2}$")
    zunits = zunits.replace("-3", "$^{-3}$")
    zunits = zunits.replace("um", "\u03bcm")
    # 'thin space', alternative is 'mid space': '\u2005'
    zunits = zunits.replace(".", "\u2009")
    zunits = zunits.replace(" ", "\u2009")

    if zunits[0] in ("e", "V", "A"):
        key_to_zunit = {
            -4: "p",
            -3: "n",
            -2: "\u03bc",
            -1: "m",
            0: "",
            1: "k",
            2: "M",
            3: "G",
            4: "T",
        }
        max_value = max(abs(vmin), abs(vmax))
        key = min(4, max(-4, int(log10(max_value)) // 3))

        return 1000**key, key_to_zunit[key] + zunits

    return 1, zunits


def set_norm(zscale: str, vmin: float, vmax: float) -> mcolors:
    """Set data-range normalization.

    Parameters
    ----------
    zscale : str
        Scaling of the data values. Recognized values are: 'linear', 'log',
        'diff' or 'ratio'.
    vmin : float
        Minimum of the data range
    vmax : float
        Minimum of the data range

    Returns
    -------
    matplotlib.colors.mcolors
    """
    if zscale == "log":
        return mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)

    if vmin == vmax:
        scale = max(1, abs(round((vmin + vmax) / 2)))
        vmin -= 1e-3 * scale
        vmax += 1e-3 * scale

    if zscale == "diff":
        if vmin < 0 < vmax:
            vcntr = 0.0
            tmp1, tmp2 = (vmin, vmax)
            vmin = -max(-tmp1, tmp2)
            vmax = max(-tmp1, tmp2)
        else:
            vcntr = (vmin + vmax) / 2
        return mcolors.TwoSlopeNorm(vcntr, vmin=vmin, vmax=vmax)

    if zscale == "ratio":
        if vmin < 1 < vmax:
            vcntr = 1.0
            tmp1, tmp2 = (vmin, vmax)
            vmin = min(tmp1, 1 / tmp2)
            vmax = max(1 / tmp1, tmp2)
        else:
            vcntr = (vmin + vmax) / 2
        return mcolors.TwoSlopeNorm(vcntr, vmin=vmin, vmax=vmax)

    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def adjust_img_ticks(axx: Axes, xarr: xr.DataArray, dims: str | None = None) -> None:
    """Adjust ticks of the image axis.

    Parameters
    ----------
    axx : matplotlib.Axes
       Matplotlib Axes object for the central image panel
    xarr : xarray.DataArray
       Object holding measurement data and attributes.
    dims : str, optional
       Name of the plot dimension (`X` or `Y`).
    """
    if dims is None or dims == "X":
        if (xarr.shape[1] % 10) == 0:
            axx.set_xticks(np.linspace(0, xarr.shape[1], 6, dtype=int))
        elif (xarr.shape[1] % 8) == 0:
            axx.set_xticks(np.linspace(0, xarr.shape[1], 5, dtype=int))
        axx.xaxis.set_minor_locator(AutoMinorLocator())

    if dims is None or dims == "Y":
        if (xarr.shape[0] % 10) == 0:
            axx.set_yticks(np.linspace(0, xarr.shape[0], 6, dtype=int))
        elif (xarr.shape[0] % 8) == 0:
            axx.set_yticks(np.linspace(0, xarr.shape[0], 5, dtype=int))
        axx.yaxis.set_minor_locator(AutoMinorLocator())


def fig_draw_panels(axx_p: dict, xarr: xr.DataArray, side_panels: str) -> None:
    """Draw two side-panels, one left and one under the main image panel.

    Parameters
    ----------
    axx_p :  dict
       dictionary holding matplotlib Axes for panel `X` and `Y`
    xarr :  xarray.DataArray
       Object holding measurement data and attributes
    side_panels :  str
       Show row and column statistics in side plots.
    """
    # get numpy function to apply on image rows and columns for side panels
    func_panels = {
        "median": np.median,
        "nanmedian": np.nanmedian,
        "mean": np.mean,
        "nanmean": np.nanmean,
        "quality": "quality",
        "std": np.std,
        "nanstd": np.nanstd,
    }.get(side_panels)
    if func_panels is None:
        raise KeyError(f"unknown function for side_panels: {side_panels}")

    # draw panel below the image panel
    xdata = np.arange(xarr.shape[1])
    if side_panels == "quality":
        ydata = np.sum(((xarr.values == 1) | (xarr.values == 2)), axis=0)
        axx_p["X"].step(xdata, ydata, linewidth=0.75, color=CSET.yellow)
        ydata = np.sum((xarr.values == 1), axis=0)  # worst
        axx_p["X"].step(xdata, ydata, linewidth=0.75, color=CSET.red)
        if len(xarr.attrs["flag_values"]) == 6:
            ydata = np.sum((xarr.values == 4), axis=0)  # to_good
            axx_p["X"].step(xdata, ydata, linewidth=0.75, color=CSET.green)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            axx_p["X"].plot(
                xdata, func_panels(xarr.values, axis=0), linewidth=0.75, color=CSET.blue
            )
    adjust_img_ticks(axx_p["X"], xarr, dims="X")
    axx_p["X"].grid()

    # draw panel left of the image panel
    ydata = np.arange(xarr.shape[0])
    if side_panels == "quality":
        xdata = np.sum(((xarr.values == 1) | (xarr.values == 2)), axis=1)
        axx_p["Y"].step(xdata, ydata, linewidth=0.75, color=CSET.yellow)
        xdata = np.sum(xarr.values == 1, axis=1)  # worst
        axx_p["Y"].step(xdata, ydata, linewidth=0.75, color=CSET.red)
        if len(xarr.attrs["flag_values"]) == 6:
            xdata = np.sum(xarr.values == 4, axis=1)  # to_good
            axx_p["Y"].step(xdata, ydata, linewidth=0.75, color=CSET.green)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            axx_p["Y"].plot(
                func_panels(xarr.values, axis=1), ydata, linewidth=0.75, color=CSET.blue
            )
    # axx_p['Y'].xaxis.tick_top()
    axx_p["X"].tick_params(axis="y", labelrotation=45, labelsize="small")
    axx_p["Y"].tick_params(
        axis="x",
        bottom=False,
        top=True,
        labelbottom=False,
        labeltop=True,
        labelrotation=-45,
        labelsize="small",
    )
    adjust_img_ticks(axx_p["Y"], xarr, dims="Y")
    axx_p["Y"].grid()


# - main functions ---------------------------------
def fig_data_to_xarr(
    data: np.ndarray,
    zscale: str = None,
    vperc: list[int, int] | None = None,
    vrange: list[float, float] | None = None,
) -> xr.DataArray:
    """Prepare image data for plotting.

    Parameters
    ----------
    data :  array-like or xarray.DataArray
    zscale :  str, default='linear'
        Scaling of the data values. Recognized values are: 'linear', 'log',
        'diff' or 'ratio'.
    vperc :  list, default=[1, 99]
        Range to normalize luminance data between percentiles min and max of
        array data.
    vrange :  list, default=None
        Range to normalize luminance data between vmin and vmax.

    Returns
    -------
    xarray.DataArray
        image data and attributes ready for plotting

    Notes
    -----
    The input data should have two dimensions. If the input data is array-like,
    it will be converted into a xarray DataArray using the function data_to_xr.

    The returned xarray DataArray has addition attributes to facilitate the
    plotting methods of MONplot with names '_cmap' or start with '_z'.

    The default values of vperc are used when both vrange and vperc are None.
    When vrange and vperc are provided, then vrange is used.
    """
    # make sure that we are working with a xarray DataArray
    xarr = data.copy() if isinstance(data, xr.DataArray) else data_to_xr(data)

    if zscale is None:
        zscale = "linear"
    if zscale not in ("diff", "linear", "log", "ratio"):
        raise RuntimeError(f"unknown zscale: {zscale}")
    xarr.attrs["_zscale"] = zscale

    # obtain image-data range
    if vrange is None and vperc is None:
        vmin, vmax = np.nanpercentile(xarr.values, (1.0, 99.0))
    elif vrange is None:
        if len(vperc) != 2:
            raise TypeError("keyword vperc requires two values")
        vmin, vmax = np.nanpercentile(xarr.values, vperc)
    else:
        if len(vrange) != 2:
            raise TypeError("keyword vrange requires two values")
        vmin, vmax = vrange

    # set data units and scaling
    if "units" in xarr.attrs:
        dscale, zunits = adjust_zunit(xarr.attrs["units"], vmin, vmax)
        xarr.values[np.isfinite(xarr.values)] /= dscale
        xarr.attrs["_zunits"] = zunits
    else:
        dscale = 1.0
        xarr.attrs["_zunits"] = "1"

    # set data label
    if zscale == "ratio" or xarr.attrs["_zunits"] == "1":
        xarr.attrs["_zlabel"] = {"ratio": "ratio", "diff": "difference"}.get(
            zscale, "value"
        )
    elif zscale == "diff":
        xarr.attrs["_zlabel"] = f'difference [{xarr.attrs["_zunits"]}]'
    else:  # zscale in ('linear', 'log')
        xarr.attrs["_zlabel"] = f'value [{xarr.attrs["_zunits"]}]'

    # set matplotlib colormap
    xarr.attrs["_cmap"] = {
        "linear": tol_cmap("rainbow_PuRd"),
        "log": tol_cmap("rainbow_WhBr"),
        "diff": tol_cmap("sunset"),
        "ratio": tol_cmap("sunset"),
    }.get(zscale)

    # set matplotlib data normalization
    xarr.attrs["_znorm"] = set_norm(zscale, vmin / dscale, vmax / dscale)

    return xarr


# pylint: disable=too-many-arguments
def fig_qdata_to_xarr(
    data: np.ndarray,
    ref_data: np.ndarray | None = None,
    data_sel: slice | None = None,
    thres_worst: float = 0.1,
    thres_bad: float = 0.8,
    qlabels: tuple[str] = None,
) -> xr.DataArray:
    r"""Prepare pixel-quality data for plotting.

    Parameters
    ----------
    data :  array-like or xarray.DataArray
        Object holding detector pixel-quality data and attributes.
    ref_data :  numpy.ndarray, default=None
        Numpy array holding reference data, for example pixel quality
        reference map taken from the CKD. Shown are the changes with
        respect to the reference data.
    data_sel :  slice, optional
        Select a region on the detector by fancy indexing (using a
        boolean/integer arrays), or using index tuples for arrays
        (generated with `numpy.s\_`).
        Outside this region the pixels will be labeled: 'unusable'.
    thres_worst :  float, default=0.1
        Threshold to reject only the worst of the bad pixels, intended
        for CKD derivation.
    thres_bad :  float, default=0.8
        Threshold for bad pixels.
    qlabels : tuple of strings, optional
        Labels for the pixel-quality classes, see below

    Returns
    -------
    xarray.DataArray
        image data and attributes ready for plotting

    Notes
    -----
    Without a reference dataset, the default quality ranking labels are::

    'unusable' :  pixels outside the illuminated region
    'worst'    :  0 <= value < thres_worst
    'bad'      :  0 <= value < thres_bad
    'good'     :  thres_bad <= value <= 1

    Otherwise, the default quality ranking labels are::

    'unusable'    :  pixels outside the illuminated region
    'to worst'    :  from good or bad to worst
    'good to bad' :  from good to bad
    'to good'     :  from any rank to good
    'unchanged'   :  no change in rank
    """
    if data_sel is None:
        exclude_region = None
    else:
        if isinstance(data_sel, np.ndarray):
            # pylint: disable=invalid-unary-operand-type
            exclude_region = ~data_sel
        else:
            exclude_region = np.full(data.shape, True)
            exclude_region[data_sel] = False

    def float_to_quality(arr: np.ndarray) -> np.ndarray:
        """Convert float value [0, 1] to quality classes."""
        res = np.empty(arr.shape, dtype="i1")
        buff = arr.values if isinstance(arr, xr.DataArray) else arr
        res[buff >= thres_bad] = 4
        res[(buff > thres_worst) & (buff < thres_bad)] = 2
        res[buff <= thres_worst] = 1
        if exclude_region is not None:
            res[exclude_region] = 0
        return res

    qval = float_to_quality(data)
    if ref_data is not None:
        qdiff = float_to_quality(ref_data) - qval
        qval = np.full_like(qdiff, 8)
        qval[(qdiff == -2) | (qdiff == -3)] = 4
        qval[qdiff == 2] = 2
        qval[(qdiff == 1) | (qdiff == 3)] = 1
        if exclude_region is not None:
            qval[exclude_region] = 0

    # make sure that we are working with a xarray DataArray
    xarr = data_to_xr(qval)
    xarr.attrs["long_name"] = "Pixel Quality"
    xarr.attrs["thres_bad"] = thres_bad
    xarr.attrs["thres_worst"] = thres_worst
    xarr.attrs["_zscale"] = "quality"

    # define colors, data-range
    if ref_data is None:
        if qlabels is None:
            xarr.attrs["flag_meanings"] = ("unusable", "worst", "bad", "good")
        elif len(qlabels) != 4:
            raise TypeError("keyword qlabels requires four labels")
        else:
            xarr.attrs["flag_meanings"] = qlabels
        # define colors for resp. unusable, worst, bad and good
        ctuple = (CSET.grey, CSET.red, CSET.yellow, "#FFFFFF")
        xarr.attrs["valid_range"] = np.array([0, 8], dtype="i1")
        xarr.attrs["flag_values"] = np.array([0, 1, 2, 4, 8], dtype="i1")
    else:
        if qlabels is None:
            xarr.attrs["flag_meanings"] = (
                "unusable",
                "to worst",
                "good to bad ",
                "to good",
                "unchanged",
            )
        elif len(qlabels) != 5:
            raise TypeError("keyword qlabels requires five labels")
        else:
            xarr.attrs["flag_meanings"] = qlabels
        # define colors for resp. unusable, worst, bad, good and unchanged
        ctuple = (CSET.grey, CSET.red, CSET.yellow, CSET.green, "#FFFFFF")
        xarr.attrs["valid_range"] = np.array([0, 16], dtype="i1")
        xarr.attrs["flag_values"] = np.array([0, 1, 2, 4, 8, 16], dtype="i1")

    xarr.attrs["_cmap"] = mcolors.ListedColormap(ctuple)
    xarr.attrs["_znorm"] = mcolors.BoundaryNorm(
        xarr.attrs["flag_values"], xarr.attrs["_cmap"].N
    )
    return xarr
