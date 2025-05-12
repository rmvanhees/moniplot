#
# This file is part of Python package: `moniplot`
#
#     https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2019-2025 SRON
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
"""Definition of the monitplot class `DrawTimeSerie`."""

from __future__ import annotations

__all__ = ["DrawTimeSerie"]

from typing import TYPE_CHECKING, NotRequired, TypedDict, Unpack

import numpy as np
from matplotlib.patches import Rectangle

from .digitized_biweight import digitized_biweight
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
    xdata :  NDArray[np.datetime64]
       X data (timestamps).
    ydata :  NDArray
       Y data.
    bin_size :  int, default=1
       Size of the date-bins in days or hours.
    min_in_bins :  int, default=25
       Minimum number of data-samples per date-bin
    percentiles :  tuple[int, ...], optional
       Sequence of pairs of percentages for the percentiles to compute.
       Values must be between 0 and 100 inclusive, excluding 50.
       The median will always be determined.

    Examples
    --------
    ...

    """

    def __init__(
        self: DrawTimeSerie,
        xdata: NDArray[np.datetime64],
        ydata: NDArray,
        bin_size: int = 1,
        min_in_bins: int = 25,
        percentiles: tuple[int, ...] | None = None,
    ) -> None:
        """Create DrawTimeSerie object."""
        self.x_perc = None
        self.y_perc = None

        # check percentiles
        if percentiles is not None and (50 in percentiles or len(percentiles) % 2 == 1):
            raise KeyError("number of percentiles should be odd and not contain 50")
        self.perc = percentiles

        res = digitized_biweight(
            xdata, ydata, bin_size=bin_size, y_in_bins=percentiles is not None
        )
        # (x_avg, y_avg) contain the timestamps and median of the input data
        # which are shown using the matplotlib step-function.
        self.x_avg = res[0]
        self.y_avg = res[1]
        if percentiles is None:
            return
        y_in_bins = res[3]

        # calculate percentiles per date-bin
        self.x_perc = np.append(res[0], res[0][-1] + ONE_DAY)
        self.y_perc = np.full((len(percentiles), self.x_perc.size), np.nan)
        for ii, ybuff in enumerate(y_in_bins):
            if ybuff.size >= min_in_bins:
                self.y_perc[:, ii] = np.nanpercentile(ybuff, self.perc)
            else:
                self.y_avg[ii] = np.nan

        # add one point at the end (actualy, at the end of every data-gap)
        self.y_perc[:, -1] = self.y_perc[:, -2]

        # close data-gaps (date-bins without data)
        i_gaps = np.isnan(self.y_avg).nonzero()[0]
        if i_gaps.size == 0:
            return

        offs = 0
        i_diff = np.append([0], np.diff(i_gaps))
        for i_d, i_g in zip(i_diff, i_gaps, strict=True):
            if i_d == 1:
                continue
            i_g += offs
            self.x_perc = np.insert(self.x_perc, i_g, self.x_perc[i_g])
            self.y_perc = np.insert(self.y_perc, i_g, self.y_perc[:, i_g - 1], axis=1)
            offs += 1

    def draw(
        self: DrawTimeSerie,
        /,
        axx: Axes,
        nolabel: bool = False,
        **kwargs: Unpack[DrawKeys],
    ) -> None:
        """Draw the actual plot and add annotations to a subplot, before closing it.

        Parameters
        ----------
        axx :  matplotlib.Axes
            Matplotlib Axes object of plot window
        nolabel :  bool, default=False
            Do not add a Matplotlib legend
        **kwargs :  Unpack[DrawKeys]
            Keywords passed to matplotlib.pyplot

        """
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

        # add grid lines (default settings)
        axx.grid(True)

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

        # define parameters for `Axes.legend`
        kwlegend = kwargs.get("kwlegend", {"fontsize": "small", "loc": "best"})

        # draw pannel
        if self.perc is not None:
            axx.fill_between(
                self.x_perc,
                self.y_perc[0, :],
                self.y_perc[-1, :],
                step="post",
                color=grey if len(self.perc) == 4 else plain_blue,
                label=None if nolabel else f"[{self.perc[0]}%, {self.perc[-1]}%]",
            )
            if len(self.perc) == 4:
                axx.fill_between(
                    self.x_perc,
                    self.y_perc[1, :],
                    self.y_perc[2, :],
                    step="post",
                    color=plain_blue,
                    label=None if nolabel else f"[{self.perc[1]}%, {self.perc[2]}%]",
                )
        axx.step(
            self.x_avg,
            self.y_avg,
            where="post",
            linewidth=2,
            color=blue,
            label=None if nolabel else "median",
        )

        # draw legenda in figure
        if axx.get_legend_handles_labels()[1]:
            axx.legend(**kwlegend)
