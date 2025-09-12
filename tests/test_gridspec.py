#
# This file is part of Python package: `moniplot`
#
#     https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2025 SRON
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
"""Check setting for subplots, used by class `DrawMulti`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from moniplot.lib.fig_info import FIGinfo

FIG_SIZES_1COL = [(10, 3), (10, 5), (10, 7), (10, 9), (10, 11)]
FIG_SIZES_NCOL = (
    [(5, 4.5), (10, 4.5), (15, 4.5), (6.5, 5)] + 2 * [(10, 5)] + 3 * [(10, 7.5)]
)
PANELS_NCOL = [(1, 1), (1, 2), (1, 3), (2, 2)] + 2 * [(2, 3)] + 3 * [(3, 3)]


def subplots(
    n_panel: int,
    *,
    one_column: bool = False,
    sharex: bool = False,
    sharey: bool = False,
    fig_info: FIGinfo | None = None,
) -> None:
    """Create the figure and a set of Axes."""
    if one_column:
        n_row = n_panel
        n_col = 1
        fig_size = FIG_SIZES_1COL[n_panel - 1]
    else:
        n_row, n_col = PANELS_NCOL[n_panel - 1]
        fig_size = FIG_SIZES_NCOL[n_panel - 1]

    # calculate space at bottom and top in inches (at n_row=2)
    bottom_inch = 0.11 * (5 if one_column else 3)
    top_inch = 0.12 * (5 if one_column else 3)

    # make room for the figinfo box
    hght = 0
    if fig_info is not None:
        hght = (0.0 if one_column else 0.2) + 0.15 * (min(5, len(fig_info)) - 1)
        top_inch += hght

    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(
        n_row,
        n_col,
        figure=fig,
        bottom=bottom_inch / fig.get_figheight(),
        top=1 - top_inch / fig.get_figheight(),
        hspace=0.08 if sharex else None,
        wspace=0.1 if sharey else None,
    )
    fig.set_figheight(fig_size[1] + hght)
    axx_arr = ()
    ip = 0
    for yy in range(n_row):
        for xx in range(n_col):
            if ip >= n_panel:
                continue
            axx = fig.add_subplot(gs[yy, xx])
            axx.grid(which="major", color="#AAAAAA", linestyle="--")
            axx_arr += (axx,)

            # decoration X-axis
            # show_label = True
            if (
                sharex
                and ip < (n_row - 1) * n_col
                and not (
                    (n_panel == 5 and ip == 2)
                    or (n_panel == 7 and ip >= 4)
                    or (n_panel == 8 and ip == 5)
                )
            ):
                axx.set_xticklabels("")
                # show_label = False

            # self.show_xlabel += (show_label,)

            # decoration Y-axis
            # show_label = True
            if sharey and ip % n_col:
                axx.set_yticklabels("")
                # show_label = False

            # self.show_ylabel += (show_label,)
            ip += 1

        # draw figinfo box
        if fig_info is not None:
            fig_info.draw(fig)

    return fig, np.array(axx_arr)


def test_grid01() -> None:
    """..."""
    for ii in range(1, 10):
        fig, _ = subplots(ii)
        fig.suptitle(f"GridSpec (n={ii})")
    plt.show()


def test_grid11() -> None:
    """..."""
    for ii in range(1, 10):
        fig, _ = subplots(ii, sharex=True, sharey=True)
        fig.suptitle(f"GridSpec (n={ii}), sharex, sharey")
    plt.show()


def test_grid02() -> None:
    """..."""
    for ii in range(1, 6):
        fig, _ = subplots(ii, one_column=True)
        fig.suptitle(f"GridSpec (n={ii}, one_column)")
    fig, _ = subplots(3, one_column=True, sharex=True)
    fig.suptitle(f"GridSpec (n={ii}, one_column, sharex)")
    plt.show()


def test_grid12() -> None:
    """..."""
    for ii in range(1, 6):
        fig, _ = subplots(ii, one_column=True, sharex=True)
        fig.suptitle(f"GridSpec (n={ii}, one_column, sharex)")
    plt.show()


def test_grid03() -> None:
    """..."""
    fig_info = FIGinfo()
    for ii in range(1, 7):
        fig_info.add(f"line_{ii:02d}", "dit is een tekst")
        fig, _ = subplots(2, fig_info=fig_info)
        fig.suptitle(f"GridSpec (n=2, one_column, fig_info[{ii}])")
    plt.show()


def test_grid13() -> None:
    """..."""
    fig_info = FIGinfo()
    for ii in range(1, 7):
        fig_info.add(f"line_{ii:02d}", "dit is een tekst")
        fig, _ = subplots(2, one_column=True, fig_info=fig_info)
        fig.suptitle(f"GridSpec (n=2, one_column, fig_info[{ii}])")
    plt.show()


def tests() -> None:
    """..."""
    # test_grid01()
    # test_grid11()
    # test_grid02()
    # test_grid12()
    # test_grid03()
    test_grid13()


if __name__ == "__main__":
    tests()
