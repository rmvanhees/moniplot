"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Defines function fig_draw_qhist used by MONplot::draw_qhist()

Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
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
