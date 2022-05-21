"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Script used to get the optimum layout for the MONplot methods draw_signal and
draw_quality.

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

import matplotlib.pyplot as plt

from moniplot.lib.fig_info import FIGinfo

def add_img_fig_box(axx_c, aspect: int, fig_info: FIGinfo) -> None:
    """
    Add a box with meta information for draw_signal and draw_quality

    Parameters
    ----------
    axx_c :  Matplotlib Axes instance of the colorbar
    aspect :  int
    fig_info :  FIGinfo
        instance of moniplot.lib.fig_info to be displayed
    """
    if fig_info is None or fig_info.location != 'above':
        return

    # put text above colorbar
    if len(fig_info) <= (7 if aspect == 1 else 6):
        if aspect <= 2:
            halign = 'center'
            fontsize = 'xx-small' if len(fig_info) > 5 else 'x-small'
        else:
            halign = 'right'
            fontsize = 'xx-small' if len(fig_info) == 6 else 'x-small'
                    
        axx_c.text(0 if aspect == 2 else 1,
                   1.04 + (aspect-1) * 0.0075,
                   fig_info.as_str(), fontsize=fontsize,
                   transform=axx_c.transAxes,
                   multialignment='left',                       
                   verticalalignment='bottom',
                   horizontalalignment=halign,
                   bbox={'facecolor': 'white', 'pad': 4})
    else:                     # put text below colorbar
        fontsize = 'xx-small' if aspect in (3, 4) else 'x-small'
        axx_c.text(0.125 + (aspect-1) * 0.2,
                   -0.03 - (aspect-1) * 0.005,
                   fig_info.as_str(), fontsize=fontsize,
                   transform=axx_c.transAxes,
                   multialignment='left',
                   verticalalignment='top',
                   horizontalalignment='left',
                   bbox={'facecolor': 'white', 'pad': 4})

def draw_figure(aspect: int, side_panels='one') -> None:
    """
    Show figure with given aspect ratio
    """
    attrs = {1: {'figsize': (10, 8),
                 'w_ratios': (1. , 7. , 0.5, 1.5),
                 'h_ratios': (7., 1.)},                  # 7 x 7
             2: {'figsize': (13, 6.25),
                 'w_ratios': (1. , 10. , 0.5, 1.5),
                 'h_ratios': (5., 1.)},                  # 10 x 5
             3: {'figsize': (15, 5.5),
                 'w_ratios': (1. , 12. , 0.5, 1.5),
                 'h_ratios': (4., 1.)},                  # 12 x 4
             4: {'figsize': (17, 4.75),
                 'w_ratios': (1. , 14. , 0.5, 1.5),
                 'h_ratios': (3.5, 1.)}}.get(aspect)     # 14 x 3.5
    
    fig = plt.figure(figsize=attrs['figsize'])
    fig.suptitle('test of matplotlib gridspec', fontsize='x-large',
                 position=(0.5, 1 - 0.4 / fig.get_figheight()))
    #
    # Define a grid layout to place subplots within the figure.
    #
    # Parameters:
    #    nrows, ncols :  int
    #        The number of rows and columns of the grid.
    #    left, right, top, bottom :  float, optional
    #        Extent of the subplots as a fraction of figure width or height.
    #        Left cannot be larger than right, and bottom cannot be larger
    #        than top. If not given, the values will be inferred from a figure
    #        or rcParams at draw time. See also GridSpec.get_subplot_params.
    #    wspace :  float, optional
    #        The amount of width reserved for space between subplots, expressed
    #        as a fraction of the average axis width. If not given, the values
    #        will be inferred from a figure or rcParams when necessary.
    #        See also GridSpec.get_subplot_params.
    #    hspace :  float, optional
    #        The amount of height reserved for space between subplots,
    #        expressed as a fraction of the average axis height. If not given,
    #        the values will be inferred from a figure or rcParams when
    #        necessary. See also GridSpec.get_subplot_params.
    #    width_ratios :  array-like of length ncols, optional
    #        Defines the relative widths of the columns. Each column gets a
    #        relative width of width_ratios[i] / sum(width_ratios). If not
    #        given, all columns will have the same width.
    #    height_ratios :  array-like of length nrows, optional
    #        Defines the relative heights of the rows. Each row gets a
    #        relative height of height_ratios[i] / sum(height_ratios).
    #        If not given, all rows will have the same height.
    gspec = fig.add_gridspec(2, 4, wspace=0.05 / aspect, hspace=0.03,
                             width_ratios=attrs['w_ratios'],
                             height_ratios=attrs['h_ratios'],
                             left=.135, right=.9, bottom=.115,
                             top=.865 - .05 * (aspect - 1) )
    # else:
    #    gspec = fig.add_gridspec(2, 4, wspace=0.05 / aspect, hspace=0.025,
    #                             width_ratios=(1., 4. * aspect, .3, .7),
    #                             height_ratios=(4, 1),
    #                             bottom=.1 if aspect == 2 else .125,
    #                             top=.85 if aspect == 2 else 0.825)

    axx = fig.add_subplot(gspec[0, 1])
    pcm = axx.pcolormesh(np.random.randn(30, aspect * 30),
                         vmin=-2, vmax=2)
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

    fig_info = FIGinfo()
    # use 5 to test x-small font (above)
    # use 7 to test xx-small font (above)
    # use 9 to test x-small font (below)
    for ii in range(5):
        fig_info.add(f'text line {ii+1}', 'blah blah blah')
    add_img_fig_box(axx_c, aspect, fig_info)

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
