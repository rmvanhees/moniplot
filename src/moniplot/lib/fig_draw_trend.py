#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022 SRON - Netherlands Institute for Space Research
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
This module contains `add_subplot` and `add_hk_subplot`
which are used by `draw_trend`.
"""
from __future__ import annotations

__all__ = ['add_subplot', 'add_hk_subplot']

from numbers import Integral
from typing import Iterable

import numpy as np
import xarray as xr
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from ..tol_colors import tol_cset
from .fig_legend import blank_legend_handle

# - global parameters ------------------------------
CSET = tol_cset('bright')


# - local functions --------------------------------
def set_labels_colors(xarr: xr.DataArray) -> tuple[str, str, str, str]:
    """Determine name and units of housekeeping data and line and fill color.

    Parameters
    ----------
    xarr :  xarray.DataArray

    Returns
    -------
    tuple
       plot parameters: hk_title, hk_label, lcolor, fcolor
    """
    hk_unit = xarr.attrs['units']
    if isinstance(hk_unit, bytes):
        hk_unit = hk_unit.decode()

    hk_title = xarr.attrs['long_name']
    if isinstance(hk_title, bytes):
        hk_title = hk_title.decode()

    if hk_unit == 'K':
        if (ii := hk_title.find(' temperature')) > 0:
            hk_title = hk_title[:ii]
        hk_label = f'temperature [{hk_unit}]'
        lcolor = CSET.blue
        fcolor = '#BBCCEE'
    elif hk_unit in ('A', 'mA'):
        if (ii := hk_title.find(' current')) > 0:
            hk_title = hk_title[:ii]
        hk_label = f'current [{hk_unit}]'
        lcolor = CSET.green
        fcolor = '#CCDDAA'
    elif hk_unit == '%':
        if (ii := hk_title.find(' duty')) > 0:
            hk_title = hk_title[:ii]
        hk_label = f'duty cycle [{hk_unit}]'
        lcolor = CSET.red
        fcolor = '#FFCCCC'
    else:
        hk_label = f'value [{hk_unit}]'
        lcolor = CSET.purple
        fcolor = '#EEBBDD'

    # overwrite ylabel
    if '_ylabel' in xarr.attrs:
        hk_label = xarr.attrs['_ylabel']

    return hk_title, hk_label, lcolor, fcolor


def adjust_ylim(data: np.ndarray | Iterable, err1: np.ndarray | Iterable | None,
                err2: np.ndarray | Iterable | None, vperc: list[int, int],
                vrange_last_orbits: int) -> tuple[float, float]:
    """Set minimum and maximum values of ylim.

    Parameters
    ----------
    data :  array_like
       Values of the data to be plotted
    err1 :  array_like
       Values of the data minus its uncertainty, None if without uncertainty
    err2 :  array_like
       Values of the data plus its uncertainty, None if without uncertainty
    vperc : list
       Limit the data range to the given percentiles
    vrange_last_orbits: int
       Use only data of the last N orbits

    Returns
    -------
    tuple of floats
       Return the limits of the Y-coordinate
    """
    if err1 is not None and err2 is not None:
        indx = np.isfinite(err1) & np.isfinite(err2)
        if np.all(~indx):
            ylim = [0., 0.]
        elif np.sum(indx) > vrange_last_orbits > 0:
            ni = vrange_last_orbits
            ylim = [min(err1[indx][0:ni].min(), err1[indx][-ni:].min()),
                    max(err2[indx][0:ni].max(), err2[indx][-ni:].max())]
        elif isinstance(vperc, list) and len(vperc) == 2:
            ylim = [np.percentile(err1[indx], vperc[0]),
                    np.percentile(err2[indx], vperc[1])]
        else:
            ylim = [err1[indx].min(), err2[indx].max()]
        factor = 10
    else:
        indx = np.isfinite(data)
        if np.all(~indx):
            ylim = [0., 0.]
        elif np.sum(indx) > vrange_last_orbits > 0:
            ni = vrange_last_orbits
            ylim = [min(data[indx][0:ni].min(), data[indx][-ni:].min()),
                    max(data[indx][0:ni].max(), data[indx][-ni:].max())]
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
    if zunit is None or zunit == '1':
        return '1'

    if zunit.find('electron') >= 0:
        zunit = zunit.replace('electron', 'e')
    if zunit.find('Volt') >= 0:
        zunit = zunit.replace('Volt', 'V')
    if zunit.find('.s-1') >= 0:
        zunit = zunit.replace('.s-1', ' s$^{-1}$')

    return zunit


def get_gap_list(xdata: np.ndarray) -> tuple:
    """Identify data gaps for data where xdata = offs + N * xstep.

    Parameters
    ----------
    xdata: numpy.ndarray
       Independent variable where the data is measured

    Returns
    -------
    list
       Indices to xdata where np.diff(xdata) greater than xstep
    """
    if not issubclass(xdata.dtype.type, Integral):
        return ()

    uvals, counts = np.unique(np.diff(xdata), return_counts=True)
    if counts.size > 1 and counts.max() / xdata.size > 0.5:
        xstep = uvals[counts.argmax()]
        return tuple(i for i in (np.diff(xdata) > xstep).nonzero()[0])

    return ()


# - main functions ---------------------------------
def add_subplot(axx, xarr: xr.DataArray) -> None:
    """Add a subplot for measurement data.

    Parameters
    ----------
    axx :  matplotlib.Axes
       Matplotlib Axes object of the current panel
    xarr :  xarray.DataArray
       Object holding measurement data and attributes
    """
    ylabel = xarr.attrs['long_name']
    if 'units' in xarr.attrs and xarr.attrs['units'] != '1':
        ylabel += f' [{adjust_units(xarr.attrs["units"])}]'
    lcolor = xarr.attrs['_color'] if '_color' in xarr.attrs else CSET.blue
    fcolor = '#BBCCEE'

    # define xdata and determine gap_list (always at least one element!)
    if 'orbit' in xarr.coords:
        xdata = xarr.coords['orbit'].values
        isel = np.s_[:]
    else:
        xdata = xarr.coords['time'].values
        isel = np.s_[0, :]
    gap_list = get_gap_list(xdata)
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
        avg = xarr.values['mean'][isel]
        err1 = xarr.values['err1'][isel]
        err2 = xarr.values['err2'][isel]

    ii = 0
    for jj in gap_list:
        isel = np.s_[ii:jj+1]
        if err1 is not None:
            axx.fill_between(xdata[isel], err1[isel], err2[isel],
                             step='post', linewidth=0, facecolor=fcolor)
            axx.step(np.append(xdata[isel], xdata[jj]),
                     np.append(avg[isel], avg[jj]), where='post',
                     linewidth=1.5, color=lcolor)
        else:
            axx.plot(xdata[isel], avg[isel], linewidth=1.5, color=lcolor)
        if 'legend' in xarr.attrs:
            legenda = axx.legend([blank_legend_handle()],
                                 [xarr.attrs['legend']], loc='upper left')
            legenda.draw_frame(False)
        ii = jj + 1

    # adjust data X-coordinate
    if 'time' in xarr.coords:
        axx.xaxis.set_major_locator(MultipleLocator(3))
        axx.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        axx.xaxis.set_minor_locator(AutoMinorLocator())
    axx.set_xlim([xdata[0], xdata[-1]])

    # adjust data X-coordinate
    axx.locator_params(axis='y', nbins=5)
    if 'orbit' in xarr.coords:
        axx.set_ylim(adjust_ylim(avg, err1, err2, [], -1))

    axx.set_ylabel(ylabel)
    axx.grid(True)


def add_hk_subplot(axx, xarr: xr.DataArray, vperc: list | None = None,
                   vrange_last_orbits: int = -1) -> None:
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
    vrange_last_orbits :  int
        Use the last N orbits to determine vrange (orbit coordinate only)
    """
    hk_title, hk_label, lcolor, fcolor = set_labels_colors(xarr)

    # define xdata and determine gap_list (always one element!)
    if 'orbit' in xarr.coords:
        xdata = xarr.coords['orbit'].values
        isel = np.s_[:]
        gap_list = get_gap_list(xdata)
    elif 'hours' in xarr.coords:
        xdata = xarr.coords['hours'].values
        isel = np.s_[0, :]
        gap_list = get_gap_list(np.round(3600 * xdata).astype(int))
    else:
        xdata = xarr.coords['time'].values
        isel = np.s_[0, :]
        gap_list = get_gap_list(xdata)
    gap_list += (xdata.size - 1,)

    # define avg, err1, err2
    avg = xarr.values['mean'][isel]
    err1 = xarr.values['err1'][isel]
    err2 = xarr.values['err2'][isel]

    # plot data
    ii = 0
    for jj in gap_list:
        isel = np.s_[ii:jj+1]
        axx.fill_between(xdata[isel], err1[isel], err2[isel],
                         step='post', linewidth=0, facecolor=fcolor)
        axx.plot(xdata[isel], avg[isel], linewidth=1.5, color=lcolor)
        ii = jj + 1

    # adjust data X-coordinate
    if 'hours' in xarr.coords:
        axx.xaxis.set_major_locator(MultipleLocator(3))
        axx.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        axx.xaxis.set_minor_locator(AutoMinorLocator())
    axx.set_xlim([xdata[0], xdata[-1]])

    # adjust data Y-coordinate
    axx.locator_params(axis='y', nbins=4)
    if 'orbit' in xarr.coords:
        axx.set_ylim(adjust_ylim(avg, err1, err2, vperc, vrange_last_orbits))
    axx.set_ylabel(hk_label)
    axx.grid(True)

    # add hk_title inside current subplots
    legenda = axx.legend([blank_legend_handle()],
                         [hk_title], loc='upper left')
    legenda.draw_frame(False)
