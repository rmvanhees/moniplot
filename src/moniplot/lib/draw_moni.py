#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2025 SRON
#    All rights reserved.
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
#
"""Definition of the monitplot class `DrawMoni`."""

from __future__ import annotations

__all__ = ["DrawMoni"]

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from moniplot.tol_colors import tol_rgba

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# - global variables -------------------------------
DEFAULT_CSET = "bright"


# - class definition -------------------------------
class DrawMoni:
    """Generic class for all drawing classes."""

    def __init__(
        self: DrawMoni,
    ) -> None:
        """Create DrawMoni object."""
        self._cset = tol_rgba(DEFAULT_CSET)
        self.xdata = None
        self.xlabel = "value"
        self.ylabel = "number"
        self.zunits = "1"
        self.time_axis = False

    # pylint: disable=too-many-arguments
    def subplots(
        self: DrawMoni,
        n_panel: int,
        *,
        one_column: bool = False,
        xlim: list[float, float] | None = None,
        xlabel: str | None = None,
        ylim: list[float, float] | None = None,
        ylabel: str | None = None,
    ) -> tuple[Figure, list[Axes, ...]]:
        """Create a figure and a set of subplots for line-plots.

        Parameters
        ----------
        n_panel: int
           Number of panels, 1 <= n_panel <= 9
        one_column: bool, default=False
           Put all panels in one column, n_panel <= 5
        xlim: list[float, float], optional
           Set the X-axis view limits.
           Note will remove unnecessary xticklabels and xlabels
        xlabel: str, optional
           Set the label for the X-axis
        ylim: list[float, float], optional
           Set the Y-axis view limits
           Note will remove unnecessary yticklabels and ylabels
        ylabel: str, optional
           Set the label for the Y-axis
        # figsize: tuple[float, float], optiional
        #    Figure dimension (width, height) in inches

        Notes
        -----
        Distributes the panels as follows:
         1: X

         2: X X

         3: X X X

         4: X X
            X X

         5: X X X
            X X

         6: X X X
            X X X

         7: X X X
            X X X
            X

         8: X X X
            X X X
            X X

         9: X X X
            X X X
            X X X

        """
        if not 1 <= n_panel <= 9:
            raise ValueError("value out of range: 1 <= n_panel <= 9")

        if one_column:
            n_row = n_panel
            n_col = 1
            fig_size = ((10, 6), (10, 7), (10, 9), (10, 10), (10, 12))[n_panel - 1]
        else:
            n_row, n_col = (
                ((1, 1), (1, 2), (1, 3), (2, 2)) + 2 * ((2, 3),) + 3 * ((3, 3),)
            )[n_panel - 1]
            fig_size = (
                ((8, 8), (12, 6), (14, 6), (9, 9)) + 2 * ((12, 9),) + 3 * ((12, 12),)
            )[n_panel - 1]

        if xlabel is not None:
            self.xlabel = xlabel
        if ylabel is not None:
            self.ylabel = ylabel
            
        fig = plt.figure(figsize=fig_size)
        margin = min(1.0 / (1.65 * (n_row + 1)), 0.3)
        kw_adjust = {
            "bottom": margin,
            "top": 1 - 1.1 / fig_size[1]
        }
        axx_arr = ()
        for ii in range(n_panel):
            axx = fig.add_subplot(n_row, n_col, ii + 1)

            # decorate X-axis
            show_label = True
            if xlim is not None:
                kw_adjust["hspace"] = 0.05
                axx.set_xlim(*xlim)
                if (
                    ii < (n_row - 1) * n_col
                    and not (
                        (n_panel == 5 and ii == 2)
                        or (n_panel == 7 and ii >= 4)
                        or (n_panel == 8 and ii == 5)
                    )
                ):
                    axx.set_xticklabels("")
                    show_label = False

            if show_label and xlabel is not None:
                axx.set_xlabel(xlabel)

            # decorate Y-axis
            show_label = True
            if ylim is not None:
                kw_adjust["wspace"] = 0.1
                axx.set_ylim(*ylim)
                if ii % n_col:
                    axx.set_yticklabels("")
                    show_label = False

            if show_label and ylabel is not None:
                axx.set_ylabel(ylabel)

            axx_arr += (axx,)

        # adjust white-space around the panels
        plt.subplots_adjust(**kw_adjust)

        return fig, np.array(axx_arr)

    def set_cset(self: DrawMoni, cname: str, cnum: int | None = None) -> None:
        """Use alternative color-set through which `draw_lplot` will cycle.

        Parameters
        ----------
        cname :  str
           Name of color set. Use None to get the default matplotlib value.
        cnum : int, optional
           Number of discrete colors in colormap (*not colorset*).

        """
        if not isinstance(cname, str):
            raise ValueError("The name of a color-set should be a string.")
        self._cset = tol_rgba(cname, cnum)

    def unset_cset(self: DrawMoni) -> None:
        """Set color set to its default."""
        self._cset = tol_rgba(DEFAULT_CSET)
