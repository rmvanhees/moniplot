"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Script used to get the optimum layout for the MONplot methods draw_signal and
draw_quality.

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
import numpy as np

import matplotlib.pyplot as plt

def draw_figure(aspect: int, side_panels='one'):
    """
    """
    figsize = {1: (9.5, 8.5),
               2: (12, 6),
               3: (12, 5),
               4: (15.5, 5)}.get(aspect)

    fig = plt.figure(figsize=figsize)
    fig.suptitle('test of matplotlib gridspec')

    # Add a gridspec
    wratio = (1., 4. * aspect, 0.25, 0.75)
    gspec = fig.add_gridspec(2, 4, left=0.1, right=0.9,
                             width_ratios=wratio, height_ratios=(4, 1),
                             bottom=0.1 if aspect in (1, 2) else 0.125,
                             top=0.9 if aspect in (1, 2) else 0.825,
                             wspace=0.05 / aspect, hspace=0.025)

    axx = fig.add_subplot(gspec[0, 1])
    pcm = axx.pcolormesh(np.random.randn(30, aspect * 30),
                        vmin=-2.5, vmax=2.5)
    # image panel
    axx.set_title(f'aspect={aspect}')
    axx.grid(True)

    # side_panels
    if side_panels != 'none':
        for xtl in axx.get_xticklabels():
            xtl.set_visible(False)
        for ytl in axx.get_yticklabels():
            ytl.set_visible(False)
        axx_px = fig.add_subplot(gspec[1, 1], sharex=axx)
        axx_px.set_xlabel('column')
        axx_px.grid(True)
        axx_py = fig.add_subplot(gspec[0, 0], sharey=axx)
        axx_py.set_ylabel('row')
        axx_py.grid(True)
    else:
        axx.set_xlabel('column')
        axx.set_ylabel('row')

    # add colorbar
    axx_c = fig.add_subplot(gspec[0, 2])
    _ = plt.colorbar(pcm, cax=axx_c, label='value')

    # text panel
    axx_c.text(1 if aspect == 1 else 0,
               1.025 if aspect in (1, 2) else 1.05,
               "(1) This is the info location\n"
               "(2) This the next line with info\n"
               "(3) This the next line with info\n"
               "(4) This the next line with info",
               transform=axx_c.transAxes,
               fontsize='x-small', style='normal',
               verticalalignment='bottom',
               horizontalalignment='center',
               multialignment='left',
               bbox={'facecolor': 'white', 'pad': 5})
    axx_c.text(0.1 * aspect,
               -0.03 - (aspect-1) * 0.005,
               "(0) This is the info location\n"
               "(1) This the next line with info\n"
               "(2) This the next line with info\n"
               "(3) This the next line with info\n"
               "(4) This the next line with info\n"
               "(5) This the next line with info\n"
               "(6) This the next line with info\n"
               "(7) This the next line with info\n"
               "(8) This the next line with info\n"
               "(9) This the next line with info",
               transform=axx_c.transAxes,
               fontsize='x-small', style='normal',
               verticalalignment='top',
               horizontalalignment='left',
               multialignment='left',
               bbox={'facecolor': 'white', 'pad': 5})
    plt.show()


def main():
    """
    """
    draw_figure(1)
    draw_figure(2)
    draw_figure(3)
    draw_figure(4)

# --------------------------------------------------
if __name__ == '__main__':
    main()
