"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

This module

Functions
---------
add_subplot(axx, xarr)
   Add a subplot for measurement data.
add_hk_subplot(axx, xarr, vperc=None, vrange_last_orbits=-1)
   Add a subplot for housekeeping data.

Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
import numpy as np

from matplotlib.ticker import MultipleLocator

from ..tol_colors import tol_cset

from .fig_legend import blank_legend_key


# - local functions --------------------------------
def set_labels_colors(xarr) -> tuple:
    """
    Determine name and units of housekeeping data and line and fill color

    Returns
    -------
    tuple
       sub_title, ylabel, line color and fill color
    """
    cset = tol_cset('bright')

    hk_unit = xarr.attrs['units'].decode()
    hk_title = xarr.attrs['long_name'].decode()
    if hk_unit == 'K':
        hk_title = hk_title.rsplit(' ', 1)[0]
        hk_label = f'temperature [{hk_unit}]'
        lcolor = cset.blue
        fcolor = '#BBCCEE'
    elif hk_unit in ('A', 'mA'):
        hk_title = hk_title.rsplit(' ', 1)[0]
        hk_label = f'current [{hk_unit}]'
        lcolor = cset.green
        fcolor = '#CCDDAA'
    elif hk_unit == '%':
        hk_title = hk_title.rsplit(' ', 2)[0]
        hk_label = f'duty cycle [{hk_unit}]'
        lcolor = cset.red
        fcolor = '#FFCCCC'
    else:
        hk_label = f'value [{hk_unit}]'
        lcolor = cset.purple
        fcolor = '#EEBBDD'

    return hk_title, hk_label, lcolor, fcolor


def adjust_ylim(avg, err1, err2, vperc: list, vrange_last_orbits: int):
    """
    Return the limits of the Y-coordinate
    """
    if err1 is not None:
        if vrange_last_orbits > 0:
            ni = vrange_last_orbits
            ylim = [min(err1[0:ni].min(), err1[-ni:].min()),
                    max(err2[0:ni].max(), err2[-ni:].max())]
        elif isinstance(vperc, list) and len(vperc) == 2:
            ylim = [np.percentile(err1, vperc[0]),
                    np.percentile(err2, vperc[1])]
        else:
            ylim = [err1.min(), err2.max()]
        factor = 10
    else:
        if vrange_last_orbits > 0:
            ni = vrange_last_orbits
            ylim = [min(avg[0:ni].min(), avg[-ni:].min()),
                    max(avg[0:ni].max(), avg[-ni:].max())]
        elif isinstance(vperc, list) and len(vperc) == 2:
            ylim = np.percentile(err1, vperc)
        else:
            ylim = [avg.min(), avg.max()]
        factor = 5

    if ylim[0] == ylim[1]:
        delta = 0.01 if ylim[0] == 0 else ylim[0] / 20
    else:
        delta = (ylim[1] - ylim[0]) / factor

    return (ylim[0] - delta, ylim[1] + delta)


def adjust_units(zunit: str):
    """
    Adjust units: electron to 'e' and Volt to 'V'

    Parameters
    ----------
    zunit :  str
       Units of the image data

    Returns
    -------
    str
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


# - main functions ---------------------------------
def add_subplot(axx, xarr) -> None:
    """
    Add a subplot for measurement data

    Parameters
    ----------
    axx :  matplotlib.Axes
    xarr :  xarray.DataArray
       Object holding measurement data and attributes
    """
    cset = tol_cset('bright')
    ylabel = xarr.attrs['long_name']
    if 'units' in xarr.attrs and xarr.attrs['units'] != '1':
        ylabel += f' [{adjust_units(xarr.attrs["units"])}]'
    lcolor = cset.blue
    fcolor = '#BBCCEE'

    # define xdata and determine gap_list (always one element!)
    if 'orbit' in xarr.coords:
        xdata = xarr.coords['orbit'].values
        isel = np.s_[:]
    else:
        xdata = xarr.coords['time'].values
        isel = np.s_[0, :]
    xstep = np.gcd.reduce(np.diff(xdata))
    gap_list = (np.diff(xdata) > xstep).nonzero()[0].tolist()
    gap_list.append(len(xdata)-1)
    # use_steps = xdata.size <= 256

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

    # ToDo: implement use_steps
    ii = 0
    for jj in gap_list:
        isel = np.s_[ii:jj+1]
        if err1 is not None:
            axx.fill_between(xdata[isel], err1[isel], err2[isel],
                             step='post', linewidth=0, facecolor=fcolor)
        axx.plot(xdata[isel], avg[isel], linewidth=1.5, color=lcolor)
        if 'legend' in xarr.attrs:
            legenda = axx.legend([blank_legend_key()],
                                 [xarr.attrs['legend']], loc='upper left')
            legenda.draw_frame(False)
        ii = jj + 1

    # adjust data X-coordinate
    if 'time' in xarr.coords:
        minor_locator = MultipleLocator(1)
        major_locator = MultipleLocator(3)
        axx.xaxis.set_major_locator(major_locator)
        axx.xaxis.set_minor_locator(minor_locator)
    axx.set_xlim([xdata[0], xdata[-1]])

    # adjust data X-coordinate
    axx.locator_params(axis='y', nbins=5)
    if 'orbit' in xarr.coords:
        axx.set_ylim(adjust_ylim(avg, err1, err2, None, -1))

    axx.set_ylabel(ylabel)
    axx.grid(True)


def add_hk_subplot(axx, xarr, vperc=None, vrange_last_orbits=-1) -> None:
    """
    Add a subplot for housekeeping data

    Parameters
    ----------
    axx :  matplotlib.Axes
    xarr :  xarray.DataArray
       Object holding housekeeping data data and attributes
    vperc :  list, optional
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
    else:
        xdata = xarr.coords['time'].values
        isel = np.s_[0, :]
    xstep = np.gcd.reduce(np.diff(xdata))
    gap_list = (np.diff(xdata) > xstep).nonzero()[0].tolist()
    gap_list.append(len(xdata)-1)
    if 'time' in xarr.coords:
        xdata = xdata.astype(float) / 3600

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
    if 'time' in xarr.coords:
        minor_locator = MultipleLocator(1)
        major_locator = MultipleLocator(3)
        axx.xaxis.set_major_locator(major_locator)
        axx.xaxis.set_minor_locator(minor_locator)
    axx.set_xlim([xdata[0], xdata[-1]])

    # adjust data Y-coordinate
    axx.locator_params(axis='y', nbins=4)
    if 'orbit' in xarr.coords:
        axx.set_ylim(adjust_ylim(avg, err1, err2, vperc, vrange_last_orbits))
    axx.set_ylabel(hk_label)
    axx.grid(True)

    # add hk_title inside current subplots
    legenda = axx.legend([blank_legend_key()],
                         [hk_title], loc='upper left')
    legenda.draw_frame(False)
