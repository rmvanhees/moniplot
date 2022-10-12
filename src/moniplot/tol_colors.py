#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2019-2022, Paul Tol
# All rights reserved
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
This module contains the definitions of color schemes and color set provided
by `Paul Tol <https://personal.sron.nl/~pault/>`_.

Contains the functions: `tol_cmap` and `tol_cset`
"""

from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, to_rgba_array


# Dictionary with all color maps as tuple of strings.
# The last element contains the color to indicate bad or out-of-range values.
_cmap_dict = {
    # Diverging color schemes
    'sunset': (
        '#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF',
        '#EAECCC', '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D',
        '#A50026', '#FFFFFF'),
    'nightfall': (
        '#125A56', '#00767B', '#238F9D', '#42A7C6', '#60BCE9',
        '#9DCCEF', '#C6DBED', '#DEE6E7', '#ECEADA', '#F0E6B2',
        '#F9D576', '#FFB954', '#FD9A44', '#F57634', '#E94C1F',
        '#D11807', '#A01813', '#FFFFFF'),
    'BuRd': (
        '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
        '#FDDBC7', '#F4A582', '#D6604D', '#B2182B', '#FFEE99'),
    'PRGn': (
        '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
        '#D9F0D3', '#ACD39E', '#5AAE61', '#1B7837', '#FFEE99'),
    # Sequential color schemes
    'YlOrBr': (
        '#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
        '#EC7014', '#CC4C02', '#993404', '#662506', '#888888'),
    'WhOrBr': (
        '#FFFFFF', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
        '#EC7014', '#CC4C02', '#993404', '#662506', '#888888'),
    'iridescent': (
        '#FEFBE9', '#FCF7D5', '#F5F3C1', '#EAF0B5', '#DDECBF',
        '#D0E7CA', '#C2E3D2', '#B5DDD8', '#A8D8DC', '#9BD2E1',
        '#8DCBE4', '#81C4E7', '#7BBCE7', '#7EB2E4', '#88A5DD',
        '#9398D2', '#9B8AC4', '#9D7DB2', '#9A709E', '#906388',
        '#805770', '#684957', '#46353A', '#999999'),
    'rainbow_PuRd': (
        '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
        '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
        '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
        '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
        '#DF4828', '#DA2222', '#FFFFFF'),
    'rainbow_PuBr': (
        '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
        '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
        '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
        '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
        '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17',
        '#521A13', '#FFFFFF'),
    'rainbow_WhRd': (
        '#E8ECFB', '#DDD8EF', '#D1C1E1', '#C3A8D1', '#B58FC2',
        '#A778B4', '#9B62A7', '#8C4E99', '#6F4C9B', '#6059A9',
        '#5568B8', '#4E79C5', '#4D8AC6', '#4E96BC', '#549EB3',
        '#59A5A9', '#60AB9E', '#69B190', '#77B77D', '#8CBC68',
        '#A6BE54', '#BEBC48', '#D1B541', '#DDAA3C', '#E49C39',
        '#E78C35', '#E67932', '#E4632D', '#DF4828', '#DA2222',
        '#666666'),
    'rainbow_WhBr': (
        '#E8ECFB', '#DDD8EF', '#D1C1E1', '#C3A8D1', '#B58FC2',
        '#A778B4', '#9B62A7', '#8C4E99', '#6F4C9B', '#6059A9',
        '#5568B8', '#4E79C5', '#4D8AC6', '#4E96BC', '#549EB3',
        '#59A5A9', '#60AB9E', '#69B190', '#77B77D', '#8CBC68',
        '#A6BE54', '#BEBC48', '#D1B541', '#DDAA3C', '#E49C39',
        '#E78C35', '#E67932', '#E4632D', '#DF4828', '#DA2222',
        '#B8221E', '#95211B', '#721E17', '#521A13', '#666666'),
    'rainbow_discrete': (
        '#E8ECFB', '#D9CCE3', '#D1BBD7', '#CAACCB', '#BA8DB4',
        '#AE76A3', '#AA6F9E', '#994F88', '#882E72', '#1965B0',
        '#437DBF', '#5289C7', '#6195CF', '#7BAFDE', '#4EB265',
        '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F6C141',
        '#F4A736', '#F1932D', '#EE8026', '#E8601C', '#E65518',
        '#DC050C', '#A5170E', '#72190E', '#42150A', '#777777')
}


# Dictionary with all color sets as tuple of strings.
# A good default for qualitative data is the bright scheme
_cset_dict = {
    'bright': {
        'names': ('blue', 'red', 'green', 'yellow', 'cyan',
                  'purple', 'grey', 'black'),
        'hexclrs': ('#4477AA', '#EE6677', '#228833', '#CCBB44',
                    '#66CCEE', '#AA3377', '#BBBBBB', '#000000')},
    'high-contrast': {
        'names': ('blue', 'yellow', 'red', 'black'),
        'hexclrs': ('#004488', '#DDAA33', '#BB5566', '#000000')},
    'medium-contrast': {
        'names': ('light_blue', 'dark_blue', 'light_yellow',
                  'dark_red', 'dark_yellow', 'light_red', 'black'),
        'hexclrs': ('#6699CC', '#004488', '#EECC66', '#994455',
                    '#997700', '#EE99AA', '#000000')},
    'vibrant': {
        'names': ('orange', 'blue', 'cyan', 'magenta', 'red', 'teal',
                  'grey', 'black'),
        'hexclrs': ('#EE7733', '#0077BB', '#33BBEE', '#EE3377',
                    '#CC3311', '#009988', '#BBBBBB', '#000000')},
    'muted': {
        'names': ('rose', 'indigo', 'sand', 'green', 'cyan', 'wine',
                  'teal', 'olive', 'purple', 'pale_grey', 'black'),
        'hexclrs': ('#CC6677', '#332288', '#DDCC77', '#117733',
                    '#88CCEE', '#882255', '#44AA99', '#999933',
                    '#AA4499', '#DDDDDD', '#000000')},
    'light': {
        'names': ('light_blue', 'orange', 'light_yellow', 'pink',
                  'light_cyan', 'mint', 'pear', 'olive', 'pale_grey', 'black'),
        'hexclrs': ('#77AADD', '#EE8866', '#EEDD88', '#FFAABB',
                    '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00',
                    '#DDDDDD', '#000000')}
}


# This is a special case os a discrete colormap
def _rainbow_discrete(cname: str, lut=0):
    """Define colormap 'rainbow_discrete'.
    """
    if lut < 1 or lut > 23:
        lut = 22

    hexclrs = _cmap_dict[cname][:-1]
    bad_hexclr = _cmap_dict[cname][-1]

    indexes = [[9], [9, 25], [9, 17, 25], [9, 14, 17, 25],
               [9, 13, 14, 17, 25], [9, 13, 14, 16, 17, 25],
               [8, 9, 13, 14, 16, 17, 25], [8, 9, 13, 14, 16, 17, 22, 25],
               [8, 9, 13, 14, 16, 17, 22, 25, 27],
               [8, 9, 13, 14, 16, 17, 20, 23, 25, 27],
               [8, 9, 11, 13, 14, 16, 17, 20, 23, 25, 27],
               [2, 5, 8, 9, 11, 13, 14, 16, 17, 20, 23, 25],
               [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 20, 23, 25],
               [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25],
               [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27],
               [2, 4, 6, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27],
               [2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27],
               [2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 26,
                27],
               [1, 3, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25,
                26, 27],
               [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 21, 23,
                25, 26, 27],
               [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22,
                24, 25, 26, 27],
               [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22,
                24, 25, 26, 27, 28],
               [0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20,
                22, 24, 25, 26, 27, 28]]

    def discretemap(colormap, hexclrs):
        """Produce a colormap from a list of discrete colors
        without interpolation.

        Parameters
        ---------
        colormap : matplotlib.colormaps
           Original colormap
        hexclrs : array_like
           Matplotlib color or array of colors

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
        """
        clrs = to_rgba_array(hexclrs)
        clrs = np.vstack([clrs[0], clrs, clrs[-1]])
        cdict = {}
        for ii, key in enumerate(('red', 'green', 'blue')):
            cdict[key] = [(jj / (len(clrs)-2.), clrs[jj, ii], clrs[jj+1, ii])
                          for jj in range(len(clrs)-1)]
        return LinearSegmentedColormap(colormap, cdict)

    cmap = discretemap('rainbow_discrete',
                       [hexclrs[i] for i in indexes[lut-1]])
    cmap.set_bad(bad_hexclr if lut == 23 else '#FFFFFF')
    return cmap


def _get_cmap(cname: str, discrete=False):
    """Return requested color map.
    """
    if cname not in _cmap_dict:
        print('[WARNING]: unknown colormap name,'
              f' using "rainbow_PuRd" not {cname}')
        cname = 'rainbow_PuRd'

    hexclrs = _cmap_dict[cname][:-1]
    bad_hexclr = _cmap_dict[cname][-1]

    if discrete:
        clrs = to_rgba_array(hexclrs)
        clrs = np.vstack([clrs[0], clrs, clrs[-1]])
        cdict = {}
        for ii, key in enumerate(('red', 'green', 'blue')):
            cdict[key] = [(jj / (len(clrs)-2.), clrs[jj, ii], clrs[jj+1, ii])
                          for jj in range(len(clrs)-1)]
        cmap = LinearSegmentedColormap(cname, cdict)
    else:
        cmap = LinearSegmentedColormap.from_list(cname, hexclrs)

    cmap.set_bad(bad_hexclr)
    return cmap


def tol_cmap(colormap=None, lut=0):
    """Continuous and discrete color sets for ordered data.

    Definition of colour schemes for lines which also work for colour-blind
    people. See `https://personal.sron.nl/~pault/`_ for background information
    and best usage of the schemes.

    Parameters
    ----------
    colormap : str, optional
       Return predefined colormap with given name.
       If not given, all possible values for colormap.
    lut : int, default=0
       Number of discrete colors in colormap.
       Parameter lut is ignored for all colormaps except 'rainbow_discrete'.

    Returns
    -------
    matplotlib.colormaps
    """
    if colormap is None:
        return _cmap_dict.keys()

    if colormap == 'rainbow_discrete':
        return _rainbow_discrete(colormap, lut)

    # do we need to return a discrete color-map?
    indx = colormap.find('_discrete')
    if indx > 0:
        cname = colormap[:indx]
        discrete = True
    else:
        cname = colormap
        discrete = False

    return _get_cmap(cname, discrete)


def tol_cset(colorset=None):
    """Discrete color sets for qualitative data.

    Definition of colour schemes for lines which also work for colour-blind
    people. See `https://personal.sron.nl/~pault/`_ for background information
    and best usage of the schemes.

    Parameters
    ----------
    colorset : str
       return color sets with name colorset. If not given, all possible values
       for colorset.

    Returns
    -------
    NamedTuple:
       a namedtuple instance with the colors.

    Notes
    -----

    * cset.red and cset[1] give the same color (in default 'bright' colorset)
    * cset._fields gives a tuple with all color names
    * list(cset) gives a list with all colors

    Examples
    --------

    >>> cset = tol_cset(<scheme>)

    """
    if colorset is None:
        return _cset_dict.keys()

    if colorset not in _cset_dict:
        print(f'[WARNING]: unknown colorset name, using bright not {colorset}')
        colorset = 'bright'

    cset = NamedTuple(colorset.replace('-', '_'),
                      [(x, str) for x in _cset_dict[colorset]['names']])
    return cset(*_cset_dict[colorset]['hexclrs'])


def __show():
    """Show all available colormaps and colorsets
    """
    # Change default colorset (for lines) and colormap (for maps).
#    plt.rc('axes', prop_cycle=plt.cycler('color', list(tol_cset('bright'))))
#    plt.cm.register_cmap('rainbow_PuRd', tol_cmap('rainbow_PuRd'))
#    plt.rc('image', cmap='rainbow_PuRd')

    # Show colorsets tol_cset(<scheme>).
    schemes = tol_cset()
    fig, axes = plt.subplots(ncols=len(schemes), figsize=(9, 3))
    fig.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.92)
    for axx, scheme in zip(axes, schemes):
        cset = tol_cset(scheme)
        names = cset._fields
        colors = list(cset)
        for name, color in zip(names, colors):
            axx.scatter([], [], c=color, s=80, label=name)
        axx.set_axis_off()
        axx.legend(loc=2)
        axx.set_title(scheme)
    plt.show()

    # Show colormaps tol_cmap(<scheme>).
    schemes = tol_cmap()
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, axes = plt.subplots(nrows=len(schemes))
    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.29, right=0.99)
    for axx, scheme in zip(axes, schemes):
        pos = list(axx.get_position().bounds)
        axx.set_axis_off()
        axx.imshow(gradient, aspect=4, cmap=tol_cmap(scheme))
        fig.text(pos[0] - 0.01, pos[1] + pos[3]/2.,
                 scheme, va='center', ha='right', fontsize=10)
    plt.show()

    # Show colormaps tol_cmap('rainbow_discrete', <lut>).
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, axes = plt.subplots(nrows=23)
    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.25, right=0.99)
    for lut, axx in enumerate(axes, start=1):
        pos = list(axx.get_position().bounds)
        axx.set_axis_off()
        axx.imshow(gradient, aspect=4, cmap=tol_cmap('rainbow_discrete', lut))
        fig.text(pos[0] - 0.01, pos[1] + pos[3]/2.,
                 f'rainbow_discrete, {lut}',
                 va='center', ha='right', fontsize=10)
    plt.show()


if __name__ == '__main__':
    __show()
