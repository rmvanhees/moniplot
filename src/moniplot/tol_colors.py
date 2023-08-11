#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2019-2022, Paul Tol
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

Routines in this module::

   tol_cmap(colormap=None, lut=0)
   tol_cset(colorset=None)
   tol_rgba(cname, cnum=None)
"""
__all__ = ['tol_cmap', 'tol_cset', 'tol_rgba']

from dataclasses import astuple, dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba_array


# pylint: disable=too-many-instance-attributes
# - Define color maps -------------------------
@dataclass
class SunSet:
    """Defines diverging color scheme 'SunSet'."""

    bad_color: str = '#FFFFFF'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF',
            '#EAECCC', '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D', '#A50026')


@dataclass
class NightFall:
    """Defines diverging color scheme 'NightFall'."""

    bad_color: str = '#FFFFFF'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#125A56', '#00767B', '#238F9D', '#42A7C6', '#60BCE9',
            '#9DCCEF', '#C6DBED', '#DEE6E7', '#ECEADA', '#F0E6B2',
            '#F9D576', '#FFB954', '#FD9A44', '#F57634', '#E94C1F',
            '#D11807', '#A01813')


@dataclass
class BuRd:
    """Defines diverging color scheme 'BuRd'."""

    bad_color: str = '#FFEE99'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
            '#FDDBC7', '#F4A582', '#D6604D', '#B2182B')


@dataclass
class PRGn:
    """Defines diverging color scheme 'PRGn'."""

    bad_color: str = '#FFEE99'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
            '#D9F0D3', '#ACD39E', '#5AAE61', '#1B7837')


@dataclass
class YlOrBr:
    """Defines sequential color scheme 'YlOrBr'."""

    bad_color: str = '#888888'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
            '#EC7014', '#CC4C02', '#993404', '#662506')


@dataclass
class WhOrBr:
    """Defines sequential color scheme 'WhOrBr'."""

    bad_color: str = '#888888'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#FFFFFF', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
            '#EC7014', '#CC4C02', '#993404', '#662506')


@dataclass
class IriDescent:
    """Defines sequential color scheme 'IriDescent'."""

    bad_color: str = '#999999'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#FEFBE9', '#FCF7D5', '#F5F3C1', '#EAF0B5', '#DDECBF',
            '#D0E7CA', '#C2E3D2', '#B5DDD8', '#A8D8DC', '#9BD2E1',
            '#8DCBE4', '#81C4E7', '#7BBCE7', '#7EB2E4', '#88A5DD',
            '#9398D2', '#9B8AC4', '#9D7DB2', '#9A709E', '#906388',
            '#805770', '#684957', '#46353A')


@dataclass
class RainbowPuRd:
    """Defines sequential color scheme 'RainbowPuRd'."""

    bad_color: str = '#FFFFFF'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
            '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
            '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
            '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
            '#DF4828', '#DA2222')


@dataclass
class RainbowPuBr:
    """Defines sequential color scheme 'RainbowPuBr'."""

    bad_color: str = '#FFFFFF'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
            '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
            '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
            '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
            '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17',
            '#521A13')


@dataclass
class RainbowWhRd:
    """Defines sequential color scheme 'RainbowWhRd'."""

    bad_color: str = '#666666'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#E8ECFB', '#DDD8EF', '#D1C1E1', '#C3A8D1', '#B58FC2',
            '#A778B4', '#9B62A7', '#8C4E99', '#6F4C9B', '#6059A9',
            '#5568B8', '#4E79C5', '#4D8AC6', '#4E96BC', '#549EB3',
            '#59A5A9', '#60AB9E', '#69B190', '#77B77D', '#8CBC68',
            '#A6BE54', '#BEBC48', '#D1B541', '#DDAA3C', '#E49C39',
            '#E78C35', '#E67932', '#E4632D', '#DF4828', '#DA2222')


@dataclass
class RainbowWhBr:
    """Defines sequential color scheme 'RainbowWhBr'."""

    bad_color: str = '#666666'
    colors: tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.colors += (
            '#E8ECFB', '#DDD8EF', '#D1C1E1', '#C3A8D1', '#B58FC2',
            '#A778B4', '#9B62A7', '#8C4E99', '#6F4C9B', '#6059A9',
            '#5568B8', '#4E79C5', '#4D8AC6', '#4E96BC', '#549EB3',
            '#59A5A9', '#60AB9E', '#69B190', '#77B77D', '#8CBC68',
            '#A6BE54', '#BEBC48', '#D1B541', '#DDAA3C', '#E49C39',
            '#E78C35', '#E67932', '#E4632D', '#DF4828', '#DA2222',
            '#B8221E', '#95211B', '#721E17', '#521A13')


@dataclass
class RainbowDiscrete:
    """Defines sequential color scheme 'RainbowDiscrete'."""

    bad_color: str = '#777777'
    colors: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.set_lut()

    def set_lut(self, lut: int = 22):
        """Define list of colors of 'rainbow_discrete'."""
        hexclrs = (
            '#E8ECFB', '#D9CCE3', '#D1BBD7', '#CAACCB', '#BA8DB4',
            '#AE76A3', '#AA6F9E', '#994F88', '#882E72', '#1965B0',
            '#437DBF', '#5289C7', '#6195CF', '#7BAFDE', '#4EB265',
            '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F6C141',
            '#F4A736', '#F1932D', '#EE8026', '#E8601C', '#E65518',
            '#DC050C', '#A5170E', '#72190E', '#42150A')

        indexes = [
            [9],
            [9, 25],
            [9, 17, 25],
            [9, 14, 17, 25],
            [9, 13, 14, 17, 25],
            [9, 13, 14, 16, 17, 25],
            [8, 9, 13, 14, 16, 17, 25],
            [8, 9, 13, 14, 16, 17, 22, 25],
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
             22, 24, 25, 26, 27, 28]
        ]

        if not 0 < lut <= len(indexes):
            lut = 23

        self.bad_color = '#FFFFFF' if lut < 23 else '#777777'
        self.colors = [hexclrs[i] for i in indexes[lut-1]]


# - Define color sets -------------------------
@dataclass
class Bright:
    """Defines a qualitative colour scheme that is colour-blind safe.
    The main scheme for lines and their labels.
    """

    blue: str = '#4477AA'
    cyan: str = '#66CCEE'
    green: str = '#228833'
    yellow: str = '#CCBB44'
    red: str = '#EE6677'
    purple: str = '#AA3377'
    grey: str = '#BBBBBB'
    black: str = '#000000'

    @property
    def colors(self):
        """Return tuple with colors."""
        return astuple(self)


@dataclass
class HighContrast:
    """Defines a qualitative colour scheme, an alternative to the bright scheme
    that is colour-blind safe and optimized for contrast. The samples
    underneath are shades of grey with the same luminance; this scheme also
    works well for people with monochrome vision and in a monochrome printout.
    """

    blue: str = '#004488'
    yellow: str = '#DDAA33'
    red: str = '#BB5566'
    black: str = '#000000'

    @property
    def colors(self):
        """Return tuple with colors."""
        return astuple(self)


@dataclass
class MediumContrast:
    """Defines a qualitative colour scheme, an alternative to the high-contrast
    scheme that is colour-blind safe with more colours. It is also optimized
    for contrast to work in a monochrome printout, but the differences are
    inevitably smaller. It is designed for situations needing colour pairs,
    shown by the three rectangles, with the lower half in the greyscale
    equivalent.
    """

    light_yellow: str = '#EECC66'
    light_red: str = '#EE99AA'
    light_blue: str = '#6699CC'
    dark_yellow: str = '#997700'
    dark_red: str = '#994455'
    dark_blue: str = '#004488'
    black: str = '#000000'

    @property
    def colors(self):
        """Return tuple with colors."""
        return astuple(self)


@dataclass
class Vibrant:
    """Defines a qualitative colour scheme, an alternative to the bright scheme
    that is equally colour-blind safe. It has been designed for data
    visualization framework TensorBoard, built around their signature orange
    FF7043. That colour has been replaced here to make it print-friendly.
    """

    blue: str = '#0077BB'
    cyan: str = '#33BBEE'
    teal: str = '#009988'
    orange: str = '#EE7733'
    red: str = '#CC3311'
    magenta: str = '#EE3377'
    grey: str = '#BBBBBB'
    black: str = '#000000'

    @property
    def colors(self):
        """Return tuple with colors."""
        return astuple(self)


@dataclass
class Muted:
    """Defines a qualitative colour scheme, an alternative to the bright scheme
    that is equally colour-blind safe with more colours, but lacking a clear
    red or medium blue. Pale grey is meant for bad data in maps.
    """

    indigo: str = '#332288'
    cyan: str = '#88CCEE'
    teal: str = '#44AA99'
    green: str = '#117733'
    olive: str = '#999933'
    sand: str = '#DDCC77'
    rose: str = '#CC6677'
    wine: str = '#882255'
    purple: str = '#AA4499'
    pale_grey: str = '#DDDDDD'
    black: str = '#000000'

    @property
    def colors(self):
        """Return tuple with colors."""
        return astuple(self)


@dataclass
class Light:
    """Defines a qualitative colour scheme that is reasonably distinct in both
    normal and colour-blind vision. It was designed to fill labelled cells with
    more and lighter colours than contained in the bright scheme, using more
    distinct colours than that in the pale scheme, but keeping black labels
    clearly readable. However, it can also be used for general qualitative
    maps.
    """

    light_blue: str = '#77AADD'
    light_cyan: str = '#99DDFF'
    mint: str = '#44BB99'
    pear: str = '#BBCC33'
    olive: str = '#AAAA00'
    light_yellow: str = '#EEDD88'
    orange: str = '#EE8866'
    pink: str = '#FFAABB'
    pale_grey: str = '#DDDDDD'
    black: str = '#000000'

    @property
    def colors(self):
        """Return tuple with colors."""
        return astuple(self)


_cset_dict = {
    'bright': Bright(),
    'high-contrast': HighContrast(),
    'medium-contrast': MediumContrast(),
    'vibrant': Vibrant(),
    'muted': Muted(),
    'light': Light()
}

_cmap_dict = {
    'sunset': SunSet(),
    'nightfall': NightFall(),
    'BuRd': BuRd(),
    'PRGn': PRGn(),
    'YlOrBr': YlOrBr(),
    'WhOrBr': WhOrBr(),
    'iridescent': IriDescent(),
    'rainbow_PuRd': RainbowPuRd(),
    'rainbow_PuBr': RainbowPuBr(),
    'rainbow_WhRd': RainbowWhRd(),
    'rainbow_WhBr': RainbowWhBr()
}


# - functions --------------------------------------
def tol_cmap(colormap: str = None, lut: int = 0):
    """Continuous and discrete color sets for ordered data.

    Definition of colour schemes for lines which also work for colour-blind
    people. See `https://personal.sron.nl/~pault/` for background information
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

    Examples
    --------
    Typical usage::

    > cmap = tol_cmap('nightfall')
    """
    if colormap is None:
        return _cmap_dict.keys()

    if colormap == 'rainbow_discrete':
        cclass = RainbowDiscrete()
        cclass.set_lut(lut)
        clrs = to_rgba_array(cclass.colors)
        clrs = np.vstack([clrs[0], clrs, clrs[-1]])
        cdict = {}
        for ii, key in enumerate(('red', 'green', 'blue')):
            cdict[key] = [(jj / (len(clrs)-2.), clrs[jj, ii], clrs[jj+1, ii])
                          for jj in range(len(clrs)-1)]

        cmap = LinearSegmentedColormap('rainbow_discrete', cdict)
        cmap.set_bad(cclass.bad_color)
        return cmap

    cclass = _cmap_dict.get(colormap, 'nightfall')
    cmap = LinearSegmentedColormap.from_list(colormap, cclass.colors)
    cmap.set_bad(cclass.bad_color)
    return cmap


def tol_cset(colorset: str = None) -> dataclass:
    """Discrete color sets for qualitative data.

    Definition of colour schemes for lines which also work for colour-blind
    people. See `https://personal.sron.nl/~pault/` for background information
    and best usage of the schemes.

    Parameters
    ----------
    colorset : str
       if None then the names of all color-sets are returned.

    Returns
    -------
    dataclass:
       a dataclass instance with colors as hexadecimal values.

    Examples
    --------
    Typical usage::

    > cset = tol_cset('bright')
    """
    if colorset is None:
        return _cset_dict.keys()

    return _cset_dict.get(colorset, Bright())


def tol_rgba(cname: str, lut: int | None = None) -> list[str, ...]:
    """
    Parameters
    ----------
    cname :  str
       Name of a colormap or colorset.
    lut : int, default=0
       Number of discrete colors in colormap (*not colorset*).
    """
    if cname == 'rainbow_discrete':
        cclass = RainbowDiscrete()
        cclass.set_lut(lut)
    elif cname in _cset_dict:
        cclass = _cset_dict.get(cname)
    else:
        cclass = _cmap_dict.get(cname, 'nightfall')

    if lut is None:
        return to_rgba_array(cclass.colors)

    cmap = tol_cmap(cname)
    cnorm = Normalize(vmin=0, vmax=lut-1)
    scalar_map = ScalarMappable(norm=cnorm, cmap=cmap)
    return [scalar_map.to_rgba(i) for i in range(lut)]


# - test functions -------------------------
def __show_cset():
    """Show colormaps tol_cset()."""
    schemes = tol_cset()
    fig, axes = plt.subplots(ncols=len(schemes), figsize=(9, 3))
    fig.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.92)
    for axx, scheme in zip(axes, schemes):
        cset = tol_cset(scheme)
        for name, color in cset.__dict__.items():
            axx.scatter([], [], c=color, s=80, label=name)
        axx.set_axis_off()
        axx.legend(loc=2)
        axx.set_title(scheme)
    plt.show()


def __show_cmap():
    """Show colormaps tol_cmap(<non-discrete>)."""
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


def __show_discrete():
    """Show colormaps tol_cmap('rainbow_discrete', <lut>)."""
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


def __show_rgba():
    """Show usage of tol_rgba()."""
    cname = 'sunset'
    print(cname, clrs := tol_rgba(cname, 16))
    fig = plt.figure()
    axx = fig.add_subplot(111)
    axx.set_prop_cycle(color=clrs)
    for ii in range(len(clrs)):
        axx.plot(np.arange(10)*(ii+1))
    plt.show()

    cname = 'muted'
    print(cname, clrs := tol_rgba(cname))
    fig = plt.figure()
    axx = fig.add_subplot(111)
    axx.set_prop_cycle(color=clrs)
    for ii in range(len(clrs)):
        axx.plot(np.arange(10)*(ii+1))
    plt.show()

    cname = 'rainbow_discrete'
    print(cname, clrs := tol_rgba(cname, 17))
    if clrs is not None:
        fig = plt.figure()
        axx = fig.add_subplot(111)
        axx.set_prop_cycle(color=clrs)
        for ii in range(len(clrs)):
            axx.plot(np.arange(10)*(ii+1))
        plt.show()


if __name__ == '__main__':
    __show_cset()
    __show_cmap()
    __show_discrete()
    __show_rgba()
