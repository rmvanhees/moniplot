#
# This file is part of Python package: `moniplot`
#
#     https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2025 SRON
#    All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Perform a unit test on several aspect ratios and image sizes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest

from moniplot.lib.fig_info import FIGinfo

if TYPE_CHECKING:
    from matplotlib import Axes


def add_fig_box(axx_c: Axes, aspect: int, fig_info: FIGinfo) -> None:
    """Add a box with meta information for draw_signal and draw_quality.

    Parameters
    ----------
    axx_c :  Axes
       Matplotlib Axes instance of the colorbar
    aspect :  int
       Provide aspect ratio
    fig_info :  FIGinfo
       instance of moniplot.lib.fig_info to be displayed

    """
    if fig_info is None:
        return

    # put text above colorbar
    if fig_info.location == "above":
        if aspect <= 2:
            halign = "left" if aspect == 1 else "center"
            fontsize = "x-small"
        else:
            halign = "right"
            fontsize = "xx-small" if len(fig_info) > 6 else "x-small"

        axx_c.text(
            0 if aspect <= 2 else 1,
            1.04 + (aspect - 1) * 0.0075,
            fig_info.as_str(),
            fontsize=fontsize,
            transform=axx_c.transAxes,
            multialignment="left",
            verticalalignment="bottom",
            horizontalalignment=halign,
            bbox={"facecolor": "white", "pad": 4},
        )
        return

    if fig_info.location == "below":
        fontsize = "xx-small" if aspect in (3, 4) else "x-small"
        axx_c.text(
            0.125 + (aspect - 1) * 0.2,
            -0.03 - (aspect - 1) * 0.005,
            fig_info.as_str(),
            fontsize=fontsize,
            transform=axx_c.transAxes,
            multialignment="left",
            verticalalignment="top",
            horizontalalignment="left",
            bbox={"facecolor": "white", "pad": 4},
        )


# -------------------------
@pytest.mark.parametrize("aspect", [1, 2, 3, 4])
def test_layout(aspect: int) -> None:
    """Show figure with given aspect ratio."""
    attrs = {
        1: {
            "figsize": (10, 8),
            "w_ratios": (1.0, 7.0, 0.5, 1.5),
            "h_ratios": (7.0, 1.0),
        },  # 7 x 7
        2: {
            "figsize": (13, 6.25),
            "w_ratios": (1.0, 10.0, 0.5, 1.5),
            "h_ratios": (5.0, 1.0),
        },  # 10 x 5
        3: {
            "figsize": (15, 5.375),
            "w_ratios": (1.0, 12.0, 0.5, 1.5),
            "h_ratios": (4.0, 1.0),
        },  # 12 x 4
        4: {
            "figsize": (17, 5.125),
            "w_ratios": (1.0, 14.0, 0.5, 1.5),
            "h_ratios": (3.5, 1.0),
        },
    }.get(aspect)  # 14 x 3.5

    fig = plt.figure(figsize=attrs["figsize"])
    fig.suptitle(
        "test of matplotlib gridspec",
        fontsize="x-large",
        position=(0.5, 1 - 0.4 / fig.get_figheight()),
    )
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
    gspec = fig.add_gridspec(
        2,
        4,
        wspace=0.1 / max(2, aspect - 1),
        hspace=0.05,
        width_ratios=attrs["w_ratios"],
        height_ratios=attrs["h_ratios"],
        left=0.135 + 0.005 * (aspect - 1),
        right=0.9 - 0.005 * (aspect - 1),
        bottom=0.115 + 0.01 * (aspect - 1),
        top=0.865 - 0.025 * (aspect - 1),
    )
    # else:
    #    gspec = fig.add_gridspec(2, 4, wspace=0.05 / aspect, hspace=0.025,
    #                             width_ratios=(1., 4. * aspect, .3, .7),
    #                             height_ratios=(4, 1),
    #                             bottom=.1 if aspect == 2 else .125,
    #                             top=.85 if aspect == 2 else 0.825)

    axx = fig.add_subplot(gspec[0, 1])
    rng = np.random.default_rng()
    pcm = axx.pcolormesh(rng.standard_normal((30, aspect * 30)), vmin=-2, vmax=2)
    # image panel
    axx.set_title(f"aspect={aspect}")
    axx.grid(True)

    # side_panels
    for xtl in axx.get_xticklabels():
        xtl.set_visible(False)
    for ytl in axx.get_yticklabels():
        ytl.set_visible(False)
    axx_px = fig.add_subplot(gspec[1, 1], sharex=axx)
    axx_px.set_xlabel("column")
    axx_px.grid(True)
    axx_py = fig.add_subplot(gspec[0, 0], sharey=axx)
    axx_py.set_ylabel("row")
    axx_py.grid(True)

    # add colorbar
    axx_c = fig.add_subplot(gspec[0, 2])
    _ = plt.colorbar(pcm, cax=axx_c, label="value")

    fig_info = FIGinfo()
    for ii in range(5):
        fig_info.add(f"text line {ii + 1}", "blah blah blah")
    add_fig_box(axx_c, aspect, fig_info)

    plt.show()
