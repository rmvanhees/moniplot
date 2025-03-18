#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2025 SRON - Netherlands Institute for Space Research
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
"""Definition of the monitplot class `DrawTimeSerie`."""

from __future__ import annotations

__all__ = ["DrawTimeSerie"]

from typing import TYPE_CHECKING, NotRequired, TypedDict, Unpack

import numpy as np
from matplotlib.patches import Rectangle

from .digitized_biweight import digitized_biweight
from .draw_lines import DrawLines
from .tol_colors import tol_cset

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

ONE_DAY = np.timedelta64(1, "D")


class DrawKeys(TypedDict):
    """Define keyword arguments of DrawLines.draw()."""

    title: NotRequired[str]
    subtitle: NotRequired[str]
    xlabel: NotRequired[str]
    ylabel: NotRequired[str]
    xlim: NotRequired[list[float, float]]
    ylim: NotRequired[list[float, float]]
    kwlegend: NotRequired[dict]


# - class definition -------------------------------
class DrawTimeSerie:
    """Plot time-serie data showing average and spread uding percentiels.

    Parameters
    ----------
    xdata: NDArray[np.datetime64]
       X data (timestamps).
    ydata: NDArray
       Y data.
    bin_size: int, default=1
       Size of the date-bins in days or hours.
    percentiles: tuple[int, ...], optional
       Sequence of pairs of percentages for the percentiles to compute.
       Values must be between 0 and 100 inclusive. The median will always be shown.

    Examples
    --------
    ...

    """

    def __init__(
        self: DrawTimeSerie,
        xdata: NDArray[np.datetime64],
        ydata: NDArray,
        bin_size: int = 1,
        percentiles: tuple[int, ...] | None = None,
    ) -> None:
        """Create DrawTimeSerie object."""
        self.perc = None
        self.x_mnmx = None
        self.y_mnmx = None
        res = digitized_biweight(
            xdata, ydata, bin_size=bin_size, y_in_bins=percentiles is not None
        )
        # add one point at the end because we use option steps
        self.xbin = np.append(res[0], res[0][-1] + ONE_DAY)
        self.ybin = res[1]
        if percentiles is None:
            return
        self.perc = percentiles
        y_in_bins = res[3]

        # check percentiles
        if 50 in percentiles or len(percentiles) % 2 == 1:
            raise KeyError("number of percentiles should be odd and not contain 50")

        # calculate percentiles per date-bin
        self.y_mnmx = np.zeros((len(percentiles), self.ybin.size + 1), dtype=float)
        for ii, ybuff in enumerate(y_in_bins):
            self.y_mnmx[:, ii] = np.nanpercentile(ybuff, self.perc)

        # add one point at the end (actualy, at the end of every data-gap)
        self.y_mnmx[:, -1] = self.y_mnmx[:, -2]
        self.x_mnmx = self.xbin.copy()

        # close data-gaps (date-bins without data)
        i_gaps = np.isnan(self.ybin).nonzero()[0]
        if i_gaps.size == 0:
            return

        i_diff = np.append([0], np.diff(i_gaps))
        for i_d, i_g in zip(i_diff, i_gaps, strict=True):
            if i_d == 1:
                continue
            self.x_mnmx = np.insert(self.x_mnmx, i_g, self.x_mnmx[i_g])
            self.y_mnmx = np.insert(self.y_mnmx, i_g, self.y_mnmx[:, i_g - 1], axis=1)

    # def set_colors(
    #    self: DrawTimeSerie,
    #    color_avg: str,
    #    color_fill1: str | None = None,
    #    color_fill2: str | None = None,
    # ) -> None:
    #    """..."""
    #    return

    def draw(
        self: DrawTimeSerie,
        /,
        axx: Axes,
        nolabel: bool = False,
        **kwargs: Unpack[DrawKeys],
    ) -> None:
        """Draw the actual plot and add annotations to a subplot, before closing it."""
        blue = tol_cset("bright").blue
        plain_blue = tol_cset("light").light_blue
        grey = tol_cset("plain").grey

        # add title to image panel
        if "title" in kwargs:
            axx.set_title(kwargs["title"], fontsize="large")

        if "subtitle" in kwargs:
            blank_legend_handle = Rectangle(
                (0, 0), 0, 0, fill=False, edgecolor="none", visible=False
            )
            legend = axx.legend(
                [blank_legend_handle], [kwargs["subtitle"]], loc="upper left"
            )
            legend.draw_frame(False)
            axx.add_artist(legend)

        # add X & Y label
        if "xlabel" in kwargs:
            axx.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            axx.set_ylabel(kwargs["ylabel"])

        # set the limits of the X-axis & Y-axis
        if "xlim" in kwargs:
            axx.set_xlim(*kwargs["xlim"])
        if "ylim" in kwargs:
            axx.set_ylim(*kwargs["ylim"])

        plot = DrawLines()
        if self.perc is not None:
            axx.fill_between(
                self.x_mnmx,
                self.y_mnmx[0, :],
                self.y_mnmx[-1, :],
                step="post",  # THIS IS IMPORTANT!
                color=grey if len(self.perc) == 4 else plain_blue,
                label=None if nolabel else f"[{self.perc[0]}%, {self.perc[-1]}%]",
            )
            if len(self.perc) == 4:
                axx.fill_between(
                    self.x_mnmx,
                    self.y_mnmx[1, :],
                    self.y_mnmx[2, :],
                    step="post",  # THIS IS IMPORTANT!
                    color=plain_blue,
                    label=None if nolabel else f"[{self.perc[1]}%, {self.perc[2]}%]",
                )
        plot.add_line(
            axx,
            self.ybin,
            xdata=self.xbin,
            use_steps=True,
            lw=2,
            color=blue,
            label=None if nolabel else "median",
        )
        if "kwlegend" in kwargs:
            plot.draw(axx, kwlegend=kwargs["kwlegend"])
        else:
            plot.draw(axx)
