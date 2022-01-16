"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

This module

Functions
---------
fig_data_to_xarr(data, zscale=None, vperc=None, vrange=None, cmap=None):
   Prepare image data for plotting.

fig_qdata_to_xarr(data, ref_data=None, thres_worst=0.1, thres_bad=0.8,
                  qlabels=None)
   Prepare pixel-quality data for plotting.

Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
from math import log10

import numpy as np
import xarray as xr

import matplotlib.colors as mcolors

from pys5p import swir_region

from ..image_to_xarray import data_to_xr
from ..tol_colors import tol_cmap, tol_cset


# - local functions --------------------------------
def adjust_zunit(zunit: str, vmin: float, vmax: float):
    """
    Adjust units: electron to 'e' and Volt to 'V'
    and scale data range to <-1000, 1000>.

    Parameters
    ----------
    zunit :  str
       Units of the image data
    vmin, vmax : float
        image-data range

    Returns
    -------
    tuple:
        dscale, zunit
    """
    if zunit is None or zunit == '1':
        return 1, zunit

    if zunit.find('electron') >= 0:
        zunit = zunit.replace('electron', 'e')
    if zunit.find('Volt') >= 0:
        zunit = zunit.replace('Volt', 'V')
    if zunit.find('.s-1') >= 0:
        zunit = zunit.replace('.s-1', ' s$^{-1}$')

    if zunit[0] in ('e', 'V', 'A'):
        key_to_zunit = {-4: 'p', -3: 'n', -2: r'\xb5', -1: 'm',
                        0: '', 1: 'k', 2: 'M', 3: 'G', 4: 'T'}
        max_value = max(abs(vmin), abs(vmax))
        key = min(4, max(-4, log10(max_value) // 3))

        return 1000 ** key, key_to_zunit[key] + zunit

    return 1, zunit


def set_norm(zscale: str, vmin: float, vmax: float):
    """
    Set data-range normalization
    """
    if zscale == 'log':
        return mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)  # clip=True

    if zscale == 'diff':
        mid_val = (vmin + vmax) / 2
        if vmin < 0 < vmax:
            tmp1, tmp2 = (vmin, vmax)
            vmin = -max(-tmp1, tmp2)
            vmax = max(-tmp1, tmp2)
            mid_val = 0.
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=mid_val, vmax=vmax)

    if zscale == 'ratio':
        mid_val = (vmin + vmax) / 2
        if vmin < 1 < vmax:
            tmp1, tmp2 = (vmin, vmax)
            vmin = min(tmp1, 1 / tmp2)
            vmax = max(1 / tmp1, tmp2)
            mid_val = 1.
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=mid_val, vmax=vmax)

    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def adjust_img_ticks(axx, xarr):
    """
    Adjust ticks of the image axis
    """
    if (xarr.shape[1] % 10) == 0:
        axx.set_xticks(np.linspace(0, xarr.shape[1], 6, dtype=int))
        axx.set_xticks(np.linspace(0, xarr.shape[1], 21, dtype=int),
                       minor=True)
    elif (xarr.shape[1] % 8) == 0:
        axx.set_xticks(np.linspace(0, xarr.shape[1], 5, dtype=int))
        axx.set_xticks(np.linspace(0, xarr.shape[1], 17, dtype=int),
                       minor=True)

    if (xarr.shape[0] % 10) == 0:
        axx.set_yticks(np.linspace(0, xarr.shape[0], 6, dtype=int))
        axx.set_yticks(np.linspace(0, xarr.shape[0], 21, dtype=int),
                       minor=True)
    elif (xarr.shape[0] % 8) == 0:
        axx.set_yticks(np.linspace(0, xarr.shape[0], 5, dtype=int))
        axx.set_yticks(np.linspace(0, xarr.shape[0], 17, dtype=int),
                       minor=True)


def fig_draw_panels(fig, xarr, side_panels: str):
    """
    Draw two side-panels, one left and one under the main image panel

    Parameters
    ----------
    fig :  matplotlib
    xarr :  xarray.DataArray
       Object holding measurement data and attributes
    side_panels :  str
       Show row and column statistics in side plots.

    Returns
    -------
    matplotlib.Axes
       axes object of the image panel
    """
    # aspect of image data
    aspect = min(4, max(1, int(round(xarr.shape[1] / xarr.shape[0]))))

    # get numpy function to apply on image rows and columns for side pannels
    func_panels = {
        'median': np.median,
        'nanmedian': np.nanmedian,
        'mean': np.mean,
        'nanmean': np.nanmean,
        'quality': 'quality',
        'std': np.std,
        'nanstd': np.nanstd}.get(side_panels, None)
    if func_panels is None:
        raise KeyError('unknown function for side_panels')

    cset = tol_cset('bright')
    if aspect == 1:
        axx = fig.add_gridspec(left=0.25, bottom=0.25).subplots()
        ax_panelx = axx.inset_axes([0, -0.25, 1, 0.2], sharex=axx)
        ax_panely = axx.inset_axes([-0.25, 0, 0.2, 1], sharey=axx)
    elif aspect == 2:
        axx = fig.add_gridspec(left=0.125, bottom=0.25).subplots()
        ax_panelx = axx.inset_axes([0, -0.25, 1, 0.2], sharex=axx)
        ax_panely = axx.inset_axes([-0.125, 0, 0.1, 1], sharey=axx)
    else:
        axx = fig.add_gridspec(left=0.0625, bottom=0.25).subplots()
        ax_panelx = axx.inset_axes([0, -0.25, 1, 0.2], sharex=axx)
        ax_panely = axx.inset_axes([-0.0625, 0, 0.05, 1], sharey=axx)

    # draw panel below the image pannel
    xdata = np.arange(xarr.shape[1])
    if side_panels == 'quality':
        ydata = np.sum(((xarr.values == 1) | (xarr.values == 2)), axis=0)
        ax_panelx.step(xdata, ydata, linewidth=0.75, color=cset.yellow)
        ydata = np.sum((xarr.values == 1), axis=0)          # worst
        ax_panelx.step(xdata, ydata, linewidth=0.75, color=cset.red)
        if len(xarr.attrs['flag_values']) == 6:
            ydata = np.sum((xarr.values == 4), axis=0)      # to_good
            ax_panelx.step(xdata, ydata, linewidth=0.75, color=cset.green)
    else:
        ax_panelx.plot(xdata, func_panels(xarr.values, axis=0),
                       linewidth=0.75, color=cset.blue)
    ax_panelx.grid()
    for xtl in axx.get_xticklabels():
        xtl.set_visible(False)
    ax_panelx.set_xlabel('column')

    # draw panel left of the image pannel
    ydata = np.arange(xarr.shape[0])
    if side_panels == 'quality':
        xdata = np.sum(((xarr.values == 1) | (xarr.values == 2)), axis=1)
        ax_panely.step(xdata, ydata, linewidth=0.75, color=cset.yellow)
        xdata = np.sum(xarr.values == 1, axis=1)            # worst
        ax_panely.step(xdata, ydata, linewidth=0.75, color=cset.red)
        if len(xarr.attrs['flag_values']) == 6:
            xdata = np.sum(xarr.values == 4, axis=1)        # to_good
            ax_panely.step(xdata, ydata, linewidth=0.75, color=cset.green)
    else:
        ax_panely.plot(func_panels(xarr.values, axis=1), ydata,
                       linewidth=0.75, color=cset.blue)
    ax_panely.grid()
    for ytl in axx.get_yticklabels():
        ytl.set_visible(False)
    ax_panely.set_ylabel('row')

    return axx


# - main functions ---------------------------------
def fig_data_to_xarr(data, zscale=None, vperc=None, vrange=None):
    """
    Prepare image data for plotting.

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

    Notes
    -----
    The input data should have two dimensions. If the input data is array-like,
    it will be converted into a xarray DataArray using the function data_to_xr
    or pys5p.s5p_xarray.

    The returned xarray DataArray has addition attributes to facilitate the
    plotting methods of S5Pplot with names '_cmap' or start with '_z'.

    The default values of vperc are used when both vrange and vperc are None.
    When vrange and vperc are provided, then vrange is used.

    Returns
    -------
    xarray.DataArray
    """
    # make sure that we are working with a xarray DataArray
    xarr = data.copy() if isinstance(data, xr.DataArray) else data_to_xr(data)

    if zscale is None:
        zscale = 'linear'
    if zscale not in ('diff', 'linear', 'log', 'ratio'):
        raise RuntimeError(f'unknown zscale: {zscale}')
    xarr.attrs['_zscale'] = zscale

    # obtain image-data range
    if vrange is None and vperc is None:
        vmin, vmax = np.nanpercentile(xarr.values, (1., 99.))
    elif vrange is None:
        if len(vperc) != 2:
            raise TypeError('keyword vperc requires two values')
        vmin, vmax = np.nanpercentile(xarr.values, vperc)
    else:
        if len(vrange) != 2:
            raise TypeError('keyword vrange requires two values')
        vmin, vmax = vrange

    # set data units and scaling
    dscale, zunits = adjust_zunit(xarr.attrs['units'], vmin, vmax)
    xarr.values[np.isfinite(xarr.values)] *= dscale
    xarr.attrs['_zunits'] = zunits
    xarr.attrs['_zrange'] = (vmin / dscale, vmax / dscale)

    # set data label
    if zscale == 'ratio' or xarr.attrs['_zunits'] == '1':
        xarr.attrs['_zlabel'] = {
            'ratio': 'ratio',
            'diff': 'difference'}.get(zscale, 'value')
    elif zscale == 'diff':
        xarr.attrs['_zlabel'] = f'difference [{xarr.attrs["_zunits"]}]'
    else:       # zscale in ('linear', 'log')
        xarr.attrs['_zlabel'] = f'value [{xarr.attrs["_zunits"]}]'

    # set matplotlib colormap
    xarr.attrs['_cmap'] = {'linear': tol_cmap('rainbow_PuRd'),
                           'log': tol_cmap('rainbow_WhBr'),
                           'diff': tol_cmap('sunset'),
                           'ratio': tol_cmap('sunset')}.get(zscale)

    # set matplotlib data normalization
    xarr.attrs['_znorm'] = set_norm(zscale, vmin, vmax)
    return xarr


def fig_qdata_to_xarr(data, ref_data=None,
                      thres_worst=0.1, thres_bad=0.8, qlabels=None):
    """
    Prepare pixel-quality data for plotting

    Parameters
    ----------
    data :  array-like or xarray.DataArray
        Object holding detector pixel-quality data and attributes.
    ref_data :  numpy.ndarray, default=None
        Numpy array holding reference data, for example pixel quality
        reference map taken from the CKD. Shown are the changes with
        respect to the reference data.
    thres_worst :  float, default=0.1
        Threshold to reject only the worst of the bad pixels, intended
        for CKD derivation.
    thres_bad :  float, default=0.8
        Threshold for bad pixels.
    qlabels : tuple of strings
        Labels for the pixel-quality classes, see below

    Notes
    -----
    Without a reference dataset, the default quality ranking labels are:
    - 'unusable' :  pixels outside the illuminated region
    - 'worst'    :  0 <= value < thres_worst
    - 'bad'      :  0 <= value < thres_bad
    - 'good'     :  thres_bad <= value <= 1

    Otherwise the default quality ranking labels are:
    - 'unusable'    :  pixels outside the illuminated region
    - 'to worst'    :  from good or bad to worst
    - 'good to bad' :  from good to bad
    - 'to good'     :  from any rank to good
    - 'unchanged'   :  no change in rank

    Returns
    -------
    xarray.DataArray
    """
    def float_to_quality(arr):
        """
        Convert float value [0, 1] to quality classes
        """
        res = np.empty(arr.shape, dtype='i1')
        buff = arr.values if isinstance(arr, xr.DataArray) else arr
        res[buff >= thres_bad] = 4
        res[(buff > thres_worst) & (buff < thres_bad)] = 2
        res[buff <= thres_worst] = 1
        res[~swir_region.mask()] = 0
        return res

    qval = float_to_quality(data)
    if ref_data is not None:
        qdiff = float_to_quality(ref_data) - qval
        qval = np.full_like(qdiff, 8)
        qval[(qdiff == -2) | (qdiff == -3)] = 4
        qval[qdiff == 2] = 2
        qval[(qdiff == 1) | (qdiff == 3)] = 1
        qval[~swir_region.mask()] = 0

    # make sure that we are working with a xarray DataArray
    xarr = data_to_xr(qval)
    xarr.attrs['long_name'] = 'Pixel Quality'
    xarr.attrs['thres_bad'] = thres_bad
    xarr.attrs['thres_worst'] = thres_worst
    xarr.attrs['_zscale'] = 'quality'

    # define colors, data-range
    cset = tol_cset('bright')
    if ref_data is None:
        if qlabels is None:
            xarr.attrs['flag_meanings'] = ("unusable", "worst", "bad", "good")
        elif len(qlabels) != 4:
            raise TypeError('keyword qlabels requires four labels')
        else:
            xarr.attrs['flag_meanings'] = qlabels
        # define colors for resp. unusable, worst, bad and good
        ctuple = (cset.grey, cset.red, cset.yellow, '#FFFFFF')
        xarr.attrs['valid_range'] = np.array([0, 8], dtype='i1')
        xarr.attrs['flag_values'] = np.array([0, 1, 2, 4, 8], dtype='i1')
    else:
        if qlabels is None:
            xarr.attrs['flag_meanings'] = ("unusable", "to worst",
                                           "good to bad ",
                                           "to good", "unchanged")
        elif len(qlabels) != 5:
            raise TypeError('keyword qlabels requires five labels')
        else:
            xarr.attrs['flag_meanings'] = qlabels
        # define colors for resp. unusable, worst, bad, good and unchanged
        ctuple = (cset.grey, cset.red, cset.yellow, cset.green, '#FFFFFF')
        xarr.attrs['valid_range'] = np.array([0, 16], dtype='i1')
        xarr.attrs['flag_values'] = np.array([0, 1, 2, 4, 8, 16], dtype='i1')

    xarr.attrs['_cmap'] = mcolors.ListedColormap(ctuple)
    xarr.attrs['_znorm'] = mcolors.BoundaryNorm(xarr.attrs['flag_values'],
                                                xarr.attrs['_cmap'].N)
    return xarr
