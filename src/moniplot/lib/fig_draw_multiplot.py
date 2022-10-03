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

import numpy as np


def get_xylabels(gridspec, data_tuple):
    """Define xlabel and ylabel for each subplot panel.

    Parameters
    ----------
    gridspec : matplotlib.gridspec.GridSpec
       Matplotlib gridspec object
    data_tuple : tuple
       X,Y data for each panel

    Returns
    -------
    numpy.ndarray
       X,Y labels for each panel
    """
    res = ()
    data_iter = iter(data_tuple)
    for _ in range(gridspec.nrows):
        for ix in range(gridspec.ncols):
            data = next(data_iter)
            if isinstance(data, np.ndarray):
                res += (['', ''],)
                continue

            xlabel = data.dims[0]
            ylabel = 'value'
            if 'units' in data.attrs and data.attrs['units'] != '1':
                ylabel += f' [{data.attrs["units"]}]'
            if ix > 0 and ylabel == res[-ix][1]:
                res += ([xlabel, ''],)
            else:
                res += ([xlabel, ylabel],)

    res = np.array(res).reshape((gridspec.nrows, gridspec.ncols, 2))
    for ix in range(gridspec.ncols):
        if np.all(res[:, ix, 0] == res[0, ix, 0]):
            res[:-1, ix, 0] = ''
    return res


def draw_subplot(axx, xarr, xylabels) -> None:
    """Draw a subplot figure.

    Parameters
    ----------
    axx : matplotlib.Axes
       Matplotlib Axes object of subplot
    xarr
      Data with attrubutes of subplot
    xylabels
      X,Y labels of subplot
    """
    if '_plot' in xarr.attrs:
        kwargs = xarr.attrs['_plot']
    else:
        kwargs = {'color': '#4477AA'}

    label = xarr.attrs['long_name'] \
        if 'long_name' in xarr.attrs else None
    xlabel = xarr.dims[0]
    axx.plot(xarr.coords[xlabel], xarr.values, label=label, **kwargs)
    if label is not None:
        _ = axx.legend(fontsize='small', loc='upper right')
    axx.set_xlabel(xylabels[0])
    axx.set_ylabel(xylabels[1])

    if '_title' in xarr.attrs:
        axx.set_title(xarr.attrs['_title'])
    if '_yscale' in xarr.attrs:
        axx.set_yscale(xarr.attrs['_yscale'])
    if '_xlim' in xarr.attrs:
        axx.set_xlim(xarr.attrs['_xlim'])
    if '_ylim' in xarr.attrs:
        axx.set_ylim(xarr.attrs['_ylim'])
    if '_text' in xarr.attrs:
        axx.text(0.05, 0.985, kwargs['text'],
                 transform=axx.transAxes,
                 fontsize='small', verticalalignment='top',
                 bbox=dict(boxstyle='round',
                           facecolor='#FFFFFF',
                           edgecolor='#BBBBBB',
                           alpha=0.5))
