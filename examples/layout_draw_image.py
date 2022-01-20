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

    wratio = {1: (1, 4, 0.25, 0.75),
              2: (1, 8, 0.25, 0.75),
              3: (1, 12, 0.25, 0.75),
              4: (1, 16, 0.25, 0.75)}.get(aspect)

    fig = plt.figure(figsize=figsize)
    fig.suptitle('test of matplotlib gridspec')

    # Add a gridspec with two rows and two columns and a ratio of 1 to 5 
    # between the size of the marginal axes and the main axes in both 
    # directions. Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 4,  width_ratios=wratio, height_ratios=(4, 1),
                          left=0.1, right=0.9,
                          bottom=0.1 if aspect in (1, 2) else 0.125,
                          top=0.9 if aspect in (1, 2) else 0.825,
                          wspace=0.05 / aspect, hspace=0.025)

    axx = fig.add_subplot(gs[0, 1])
    pc = axx.pcolormesh(np.random.randn(30, aspect * 30),
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
        ax_panelx = fig.add_subplot(gs[1, 1], sharex=axx)
        ax_panelx.set_xlabel('column')
        ax_panelx.grid(True)
        ax_panely = fig.add_subplot(gs[0, 0], sharey=axx)
        ax_panely.set_ylabel('row')
        ax_panely.grid(True)
    else:
        axx.set_xlabel('column')
        axx.set_ylabel('row')

    # text panel
    ax_panelz = fig.add_subplot(gs[0, 2])
    _ = plt.colorbar(pc, cax=ax_panelz, label='value')

    if 1 == 0:
        ax_panelz.text(1 if aspect == 1 else 0,
                       1.025 if aspect in (1, 2) else 1.05,
                       "(1) This is the info location\n"
                       "(2) This the next line with info\n"
                       "(3) This the next line with info\n"
                       "(4) This the next line with info",
                       transform=ax_panelz.transAxes,
                       fontsize='x-small', style='normal',
                       verticalalignment='bottom',
                       horizontalalignment='center',
                       multialignment='left',
                       bbox={'facecolor': 'white', 'pad': 5})
    else:
        ax_info = fig.add_subplot(gs[1, 2:])
        ax_info.axis('off')
        ax_info.text(0.075 if aspect in (1, 2) else 0.15,
                     0.95 - (aspect-1) * 0.025,
                     "(1) This is the info location\n"
                     "(2) This the next line with info\n"
                     "(3) This the next line with info\n"
                     "(4) This the next line with info\n"
                     "(5) This the next line with info\n"
                     "(6) This the next line with info\n"
                     "(7) This the next line with info\n"
                     "(8) This the next line with info",
                     fontsize='xx-small', style='normal',
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
