"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Defines function fig_draw_qhist used by MONplot::draw_qhist()

Copyright (c) 2022 SRON - Netherlands Institute for Space Research

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

License:  GPLv3
"""
import numpy as np

from ..tol_colors import tol_cset
from .fig_legend import blank_legend_key


def fig_draw_qhist(axx, qdata, label: str, density: bool):
    """
    Add a subplot showing pixel-quality data as a histogram

    Parameters
    ----------
    axx :  matplotlib.Axes
    qdata :  numpy.ndarray
       Object holding pixel-quality data and attributes
    label :  str
       Name describing qdata, displayed in the upper left corner
    density :  bool
       See method MONplot::draw_qhist for a description
    """
    # define colors
    cset = tol_cset('bright')

    qdata[np.isnan(qdata)] = 0.
    axx.hist(qdata, bins=11, range=[-.1, 1.],
             histtype='stepfilled', log=True,
             density=density, color=cset.blue)
    # axx.set_yscale('log', nonpositive='clip')
    axx.set_xlim([0, 1])
    axx.set_ylabel('density' if density else 'count')
    axx.set_ylim([1e-4, 10])
    axx.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    axx.grid(which='major', color='#BBBBBB',
                   lw=0.75, ls=(0, (1, 5)))
    legenda = axx.legend([blank_legend_key()],
                         [label], loc='upper left')
    legenda.draw_frame(False)
