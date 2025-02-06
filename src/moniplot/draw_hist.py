#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2024 SRON - Netherlands Institute for Space Research
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
"""Definition of the moniplot class `DrawHist`."""

from __future__ import annotations

__all__ = ["DrawHist"]

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .biweight import Biweight
from .lib.fig_info import FIGinfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# - global variables -------------------------------
DEFAULT_CSET = "bright"


# - class definition -------------------------------
class DrawHist:
    """Create a histogram plot.

    Parameters
    ----------
    arr :  xr.DataArray | np.ndarray
        Data array
    clip :  tuple[float | None, float | None] | None, optional
        Pass clip values to numpy.clip
    **kwargs :  int
        Pass 'bin', 'density', and/or 'weights' as parameters to numpy.histogram

    Notes
    -----
    When the input data is a xarray.DataArray then the attributes 'long_name'
    and 'units' are used in the plot decoration.

    Examples
    --------
    Generate a figure with two histogram plots.

    >>> from moniplot.mon_plot import MONplot
    >>> report = MONplot("test_monplot.pdf", "This is an example figure")
    >>> report.set_institute("SRON")
    >>> fig, axx = plt.subplots(1, 2, sharey="all", figsize=(9, 8))
    >>> plot = DrawHist(dist1, bins=20)
    >>> plot.draw(axx[0], title="Dit is het eerste histogram")
    >>> plot = DrawHist(dist2, bins=20)
    >>> plot.draw(axx[1], yticks_visible=False)
    >>> axx[1].set_title("Dit is het tweede histogram")
    >>> report.add_copyright(axx[1])
    >>> report.close_this_page(fig, plot.get_figinfo())
    >>> report.close()

    """

    def __init__(
        self: DrawHist,
        arr: xr.DataArray | np.ndarray,
        clip: tuple[float | None, float | None] | None = None,
        **kwargs: int,
    ) -> None:
        """Create a DrawHist object."""
        self.xlabel = "value"
        self.ylabel = "density" if kwargs.get("density") else "number"
        self.zunits = "1"
        if isinstance(arr, xr.DataArray):
            data = np.ravel(arr.values)
            if "long_name" in arr.attrs:
                self.xlabel = arr.attrs["long_name"]
            if "units" in arr.attrs:
                self.zunits = arr.attrs["units"]
                self.xlabel += f" [{self.zunits}]"
        else:
            data = np.ravel(arr)

        if clip is not None:
            data = np.clip(data, *clip)
            vrange = (
                data.min() if clip[0] is None else clip[0],
                data.max() if clip[1] is None else clip[1],
            )
        else:
            vrange = (data.min(), data.max())

        self.__data = data
        self.__hist, self.__edges = np.histogram(data, range=vrange, **kwargs)

    def get_figinfo(self: DrawHist, fig_info: FIGinfo | None = None) -> FIGinfo:
        """..."""
        if fig_info is None:
            fig_info = FIGinfo()

        with Biweight(self.__data) as bwght:
            if self.zunits == "1":
                fig_info.add("median", bwght.median, "{:.5g}")
                fig_info.add("spread", bwght.spread, "{:.5g}")
            else:
                fig_info.add("median", (bwght.median, self.zunits), "{:.5g} {}")
                fig_info.add("spread", (bwght.spread, self.zunits), "{:.5g} {}")

        return fig_info

    def draw(
        self: DrawHist,
        axx: Axes | None = None,
        title: str | None = None,
        xticks_visible: bool = True,
        yticks_visible: bool = True,
    ) -> None:
        """..."""
        if len(self.__hist) > 24:
            axx.stairs(
                self.__hist,
                self.__edges,
                edgecolor="#4477AA",
                facecolor="#77AADD",
                fill=True,
                linewidth=1.5,
            )
            axx.grid(which="major", color="#AAAAAA", linestyle="--")
        else:
            axx.bar(
                self.__edges[:-1],
                self.__hist,
                width=np.diff(self.__edges),
                align="edge",
                edgecolor="#4477AA",
                facecolor="#77AADD",
                linewidth=1.5,
            )
            axx.grid(which="major", axis="y", color="#AAAAAA", linestyle="--")

        if not xticks_visible:
            for xtl in axx.get_xticklabels():
                xtl.set_visible(False)
        else:
            axx.set_xlabel(self.xlabel)

        if not yticks_visible:
            for ytl in axx.get_yticklabels():
                ytl.set_visible(False)
        else:
            axx.set_ylabel(self.ylabel)

        if title is not None:
            axx.set_title(title)
