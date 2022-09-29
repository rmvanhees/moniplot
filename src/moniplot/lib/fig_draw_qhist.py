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

from matplotlib.ticker import AutoMinorLocator
import numpy as np

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
    qdata[np.isnan(qdata)] = 0.
    # draw histogram
    axx.hist(qdata, bins=10, range=[0, 1], density=density,
             histtype='bar', align='mid', log=True, fill=True,
             edgecolor='#4477AA', facecolor='#77AADD', linewidth=1.5)
    # add label
    legenda = axx.legend([blank_legend_key()],
                         [label], loc='upper left')
    legenda.draw_frame(False)
    # add decoration
    axx.grid(which='major', axis='y', color='#AAAAAA', ls='--')
    axx.set_xlim([0, 1])
    axx.xaxis.set_minor_locator(AutoMinorLocator(2))
    axx.set_ylabel('density' if density else 'count')
    axx.set_ylim([1e-4, 10])
    axx.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
