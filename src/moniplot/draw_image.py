#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2023 SRON - Netherlands Institute for Space Research
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
"""This module contains the class `DrawImage`."""

from __future__ import annotations

__all__ = ["DrawImage"]

import warnings
from math import log10
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import AutoMinorLocator

from .biweight import Biweight
from .lib.fig_info import FIGinfo
from .tol_colors import tol_cmap, tol_cset

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# - global variables -------------------------------


# - class definition -------------------------------
class DrawImage:
    """Create a detector-plot with colorbar and side-panels (optional).

    Parameters
    ----------
    arr :  np.ndarray | xr.DataArray
        2D-data array
    zscale :  str, default='linear'
        Scaling of the data values. Recognized values are: 'linear', 'log',
        'diff', 'ratio' or 'quality'.
    vperc :  list, default=[1, 99]
        Range to normalize luminance data between percentiles min and max of
        array data.
    vrange :  list, default=None
        Range to normalize luminance data between vmin and vmax.

    Examples
    --------
    Generate a figure with a detector image

    >>> report = MONplot("test_monplot.pdf", "This is an example figure")
    >>> report.set_institute("SRON")
    >>> plot = DrawImage(arr2d)
    >>> fig, axx = plot.subplots()
    >>> plot.draw(axx, fig_info=fig_info, title="Dit is een ruis plaatje")
    >>> report.add_copyright(axx["image"])
    # Note fig_info=None, already added by plot.draw
    >>> report.close_this_page(fig, None)
    """

    def __init__(
        self: DrawImage,
        arr: np.ndarray | xr.DataArray,
        zscale: str = None,
        vperc: list[int, int] | None = None,
        vrange: list[float, float] | None = None,
    ) -> None:
        """Prepare image data for plotting."""
        self._cmap = tol_cmap("rainbow_PuRd")
        self._cset = tol_cset("bright")
        self._image = arr.values if isinstance(arr, xr.DataArray) else arr
        self.attrs = {
            "long_name": "",
            "units": "1",
            "dims": ["row", "column"],
        }
        self._zlabel: str = "value"

        # check parameter 'zscale'
        if zscale is None:
            self._zscale = "linear"
        elif zscale not in ("diff", "linear", "log", "ratio", "quality"):
            raise RuntimeError(f"unknown zscale: {zscale}")
        else:
            self._zscale = zscale

        if self._zscale == "quality":
            # check DataArray
            if not isinstance(arr, xr.DataArray):
                raise ValueError("Pixel-Quality data must be a xr.DataArray")
            # check attributes
            for key in [
                "colors",
                "long_name",
                "thres_bad",
                "thres_worst",
                "flag_meanings",
                "flag_values",
            ]:
                if key not in arr.attrs:
                    raise KeyError(f"attribute {key} not present in Pixel-Quality data")
                self.attrs[key] = arr.attrs[key]

            self._zlabel = "quality"
            self._cmap = mcolors.ListedColormap(arr.attrs["colors"])
            self._znorm = mcolors.BoundaryNorm(arr.attrs["flag_values"], self._cmap.N)
            return

        # obtain image-data range
        if vrange is None and vperc is None:
            vmin, vmax = np.nanpercentile(self._image, (1.0, 99.0))
        elif vrange is None:
            if len(vperc) != 2:
                raise TypeError("keyword vperc requires two values")
            vmin, vmax = np.nanpercentile(self._image, vperc)
        else:
            if len(vrange) != 2:
                raise TypeError("keyword vrange requires two values")
            vmin, vmax = vrange

        # obtain image-data and attributes
        if isinstance(arr, xr.DataArray):
            if "long_name" in arr.attrs:
                self.attrs["long_name"] = arr.attrs["long_name"]
            if "units" in arr.attrs:
                dscale = self.set_zunit(arr.attrs["units"], vmin, vmax)
                if dscale != 1:
                    vmin /= dscale
                    vmax /= dscale
                    self._image /= dscale

        # set data-label and matplotlib colormap
        match self._zscale:
            case "linear":
                self._zlabel = "value"
                self._cmap = tol_cmap("rainbow_PuRd")
            case "log":
                self._zlabel = "value"
                self._cmap = tol_cmap("rainbow_WhBr")
            case "diff":
                self._zlabel = "difference"
                self._cmap = tol_cmap("sunset")
            case "ratio":
                self._zlabel = "ratio"
                self._cmap = tol_cmap("sunset")
        if not (zscale == "ratio" or self.attrs["units"] == "1"):
            self._zlabel += f" [{self.attrs['units']}]"

        # set matplotlib data normalization
        self._znorm = self.set_norm(vmin, vmax)

    @property
    def aspect(self: DrawImage) -> int | None:
        """Return aspect-ratio of image data."""
        if self._image is None:
            return None

        return min(4, max(1, int(round(self._image.shape[1] / self._image.shape[0]))))

    def subplots(
        self: DrawImage, side_panels: bool = True
    ) -> tuple[Figure, dict[str, Axes]]:
        """Obtain matplotlib Figure and Axes for plot-layout.

        Parameters
        ----------
        side_panels : bool, optional
           Do you want side_panels with row and column statistics
        """
        match self.aspect:
            case 1:
                width_ratios = (8, 0.5, 0.5)
                height_ratios = (0.2, 8)
            case 2:
                width_ratios = (12, 0.5, 0.5)
                height_ratios = (0.25, 6)
            case 3:
                width_ratios = (15, 0.5, 0.5)
                height_ratios = (0.35, 5)
            case 4:
                width_ratios = (16, 0.5, 0.5)
                height_ratios = (0.5, 4)
            case _:
                raise ValueError("unknown aspect-ratio")

        if side_panels:
            mosaic = [
                [".", "caption", ".", "info"],
                ["y-panel", "image", "colorbar", "."],
                [".", "x-panel", ".", "."],
            ]
            width_ratios = (1,) + width_ratios
            height_ratios += (1,)
        else:
            mosaic = [["caption", ".", "info"], ["image", "colorbar", "."]]

        fig, axx = plt.subplot_mosaic(
            mosaic,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            figsize=(np.sum(width_ratios), np.sum(height_ratios)),
        )
        plt.subplots_adjust(
            hspace=0.05 + 0.01 * (self.aspect - 1),
            wspace=0.07 - 0.01 * (self.aspect - 1),
        )

        # adjust axes
        axx["info"].set_axis_off()
        axx["caption"].set_axis_off()
        axx["colorbar"].tick_params(
            axis="x",
            bottom=False,
            top=False,
            labelbottom=False,
            labeltop=False,
        )
        axx["colorbar"].tick_params(
            axis="y",
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
        )
        axx["colorbar"].ticklabel_format(useOffset=False)
        if side_panels:
            for xtl in axx["image"].get_xticklabels():
                xtl.set_visible(False)
            for ytl in axx["image"].get_yticklabels():
                ytl.set_visible(False)
            axx["x-panel"].tick_params(axis="y", labelrotation=45, labelsize="small")
            axx["y-panel"].tick_params(
                axis="x",
                bottom=False,
                top=True,
                labelbottom=False,
                labeltop=True,
                labelrotation=-45,
                labelsize="small",
            )
            axx["x-panel"].ticklabel_format(useOffset=False)
            axx["x-panel"].sharex(axx["image"])
            axx["x-panel"].xaxis.set_minor_locator(AutoMinorLocator())
            axx["y-panel"].ticklabel_format(useOffset=False)
            axx["y-panel"].sharey(axx["image"])
            axx["y-panel"].yaxis.set_minor_locator(AutoMinorLocator())

        return fig, axx

    def set_zunit(self: DrawImage, zunits: str, vmin: float, vmax: float) -> int:
        """Adjust data units given the data range.

        Units which are renamed: electron to `e` and Volt to `V`
        Scale data-range between -1000 and 1000.

        Parameters
        ----------
        zunits :  str
            Units of the image data
        vmin, vmax : float
            image-data range

        Returns
        -------
        dscale
            Factor to scale data accoring to zunits
        """
        if zunits == "1":
            return 1

        zunits = zunits.replace("electron", "e")
        zunits = zunits.replace("Volt", "V")
        zunits = zunits.replace("-1", "$^{-1}$")
        zunits = zunits.replace("-2", "$^{-2}$")
        zunits = zunits.replace("-3", "$^{-3}$")
        zunits = zunits.replace("um", "\u03bcm")
        # 'thin space', alternative is 'mid space': '\u2005'
        zunits = zunits.replace(".", "\u2009")
        zunits = zunits.replace(" ", "\u2009")

        if zunits[0] in ("e", "V", "A", "m"):
            key_to_zunit = {
                -4: "p",
                -3: "n",
                -2: "\u03bc",
                -1: "m",
                0: "",
                1: "k",
                2: "M",
                3: "G",
                4: "T",
            }
            max_value = max(abs(vmin), abs(vmax))
            key = min(4, max(-4, int(log10(max_value)) // 3))

            self.attrs["units"] = key_to_zunit[key] + zunits
            return 1000**key

        self.attrs["units"] = zunits
        return 1

    def set_norm(self: DrawImage, vmin: float, vmax: float) -> mcolors:
        """Set data-range normalization for matplotlib.

        Parameters
        ----------
        vmin : float
            Minimum of the data range
        vmax : float
            Maximum of the data range

        Returns
        -------
        matplotlib.colors.mcolors
        """
        if self._zscale == "log":
            return mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)

        if vmin == vmax:
            scale = max(1, abs(round((vmin + vmax) / 2)))
            vmin -= 1e-3 * scale
            vmax += 1e-3 * scale
            vcntr = (vmin + vmax) / 2
            if self._zscale == "diff" and vmin < 0 < vmax:
                vcntr = 0.0
                tmp1, tmp2 = (vmin, vmax)
                vmin = -max(-tmp1, tmp2)
                vmax = max(-tmp1, tmp2)

            return mcolors.TwoSlopeNorm(vcntr, vmin=vmin, vmax=vmax)

        if self._zscale == "ratio":
            if vmin < 1 < vmax:
                vcntr = 1.0
                tmp1, tmp2 = (vmin, vmax)
                vmin = min(tmp1, 1 / tmp2)
                vmax = max(1 / tmp1, tmp2)
            else:
                vcntr = (vmin + vmax) / 2
            return mcolors.TwoSlopeNorm(vcntr, vmin=vmin, vmax=vmax)

        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def __draw_image(self: DrawImage, axx: dict, title: str | None) -> None:
        """..."""
        if self._zscale == "quality":
            cm_img = axx["image"].imshow(
                self._image,
                norm=self._znorm,
                extent=[0, self._image.shape[1], 0, self._image.shape[0]],
                aspect="auto",
                cmap=self._cmap,
                interpolation="none",
                origin="lower",
            )
        else:
            # cmap = self.cmap if self.cmap else self._cmap
            cm_img = axx["image"].imshow(
                self._image,
                norm=self._znorm,
                extent=[0, self._image.shape[1], 0, self._image.shape[0]],
                aspect="auto",
                cmap=self._cmap,
                interpolation="none",
                origin="lower",
            )

        # adjust tickmarks
        if (self._image.shape[1] % 10) == 0:
            axx["image"].set_xticks(np.linspace(0, self._image.shape[1], 6, dtype=int))
        elif (self._image.shape[1] % 8) == 0:
            axx["image"].set_xticks(np.linspace(0, self._image.shape[1], 5, dtype=int))
        if (self._image.shape[0] % 10) == 0:
            axx["image"].set_yticks(np.linspace(0, self._image.shape[0], 6, dtype=int))
        elif (self._image.shape[0] % 8) == 0:
            axx["image"].set_yticks(np.linspace(0, self._image.shape[0], 5, dtype=int))

        if self._zscale == "quality":
            bounds = [int(i) for i in self.attrs["flag_values"]]
            mbounds = [
                (bounds[ii + 1] + bounds[ii]) / 2 for ii in range(len(bounds) - 1)
            ]
            _ = plt.colorbar(
                cm_img, cax=axx["colorbar"], ticks=mbounds, boundaries=bounds
            )
            axx["colorbar"].tick_params(axis="y", which="both", length=0)
            axx["colorbar"].set_yticklabels(self.attrs["flag_meanings"])
        else:
            _ = plt.colorbar(cm_img, cax=axx["colorbar"], label=self._zlabel)

        # add title to image panel
        if title is not None:
            axx["image"].set_title(title)
        else:
            axx["image"].set_title(self.attrs["long_name"])

    def __draw_side_panels(self: DrawImage, axx: dict, side_panels: str) -> None:
        """..."""
        if self._zscale == "quality":
            # draw panel below the image panel
            xdata = np.arange(self._image.shape[1])
            ydata = np.sum(((self._image == 1) | (self._image == 2)), axis=0)
            axx["x-panel"].step(xdata, ydata, linewidth=0.75, color=self._cset.yellow)
            ydata = np.sum((self._image == 1), axis=0)  # worst
            axx["x-panel"].step(xdata, ydata, linewidth=0.75, color=self._cset.red)
            if len(self.attrs["flag_values"]) == 6:
                ydata = np.sum((self._image == 4), axis=0)  # to_good
                axx["x-panel"].step(
                    xdata, ydata, linewidth=0.75, color=self._cset.green
                )
            axx["x-panel"].grid()

            # draw panel left of the image panel
            ydata = np.arange(self._image.shape[0])
            xdata = np.sum(((self._image == 1) | (self._image == 2)), axis=1)
            axx["y-panel"].step(xdata, ydata, linewidth=0.75, color=self._cset.yellow)
            xdata = np.sum(self._image == 1, axis=1)  # worst
            axx["y-panel"].step(xdata, ydata, linewidth=0.75, color=self._cset.red)
            if len(self.attrs["flag_values"]) == 6:
                xdata = np.sum(self._image == 4, axis=1)  # to_good
                axx["y-panel"].step(
                    xdata, ydata, linewidth=0.75, color=self._cset.green
                )
            axx["y-panel"].grid()
            return

        match side_panels:
            case "nanmedian" | "default":
                func_panels = np.nanmedian
            case "nanmean":
                func_panels = np.nanmean
            case "nanstd":
                func_panels = np.nanstd
            case "median":
                func_panels = np.median
            case "mean":
                func_panels = np.mean
            case "std":
                func_panels = np.std
            case _:
                raise ValueError(f"unknown function for side_panels: {side_panels}")

        # draw panel below the image panel
        xdata = np.arange(self._image.shape[1])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            axx["x-panel"].plot(
                xdata,
                func_panels(self._image, axis=0),
                linewidth=0.75,
                color=self._cset.blue,
            )
            axx["x-panel"].grid()

        # draw panel left of the image panel
        ydata = np.arange(self._image.shape[0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            axx["y-panel"].plot(
                func_panels(self._image, axis=1),
                ydata,
                linewidth=0.75,
                color=self._cset.blue,
            )
        axx["y-panel"].grid()

    @staticmethod
    def __add_info_box(axx: Axes, fig_info: FIGinfo) -> None:
        """Add a box with meta information in the current figure.

        Parameters
        ----------
        axx :  matplotlib.axes.Axes
        fig_info :  FIGinfo
            Instance of pys5p.lib.plotlib.FIGinfo to be displayed
        """
        if fig_info is None or fig_info.location != "above":
            return

        axx.text(
            2.0,
            0.3,
            fig_info.as_str(),
            fontsize="x-small",
            transform=axx.transAxes,
            multialignment="left",
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"facecolor": "white", "pad": 4},
        )

    def draw(
        self: DrawImage,
        axx: dict[str, Axes],
        *,
        fig_info: FIGinfo | None = None,
        side_panels: str = "default",
        title: str | None = None,
    ) -> None:
        """Display 2D array as an image.

        Averaged column/row signal are optionally displayed in side-panels.

        Parameters
        ----------
        axx :  dict[str, Axes]
        fig_info :  FIGinfo, <default=None
           OrderedDict holding meta-data to be displayed in the figure.
        side_panels :  str, default='nanmedian'
           Show image row and column statistics in two side panels.
           Valid values are: 'median', 'nanmedian', 'mean', 'nanmean',
           'quality', 'std' and 'nanstd'.
        title :  str, default=None
           Title of this figure using `Axis.set_title`.

        Notes
        -----
        The information provided in the parameter `fig_info` will be displayed
        in a text box. In addition, we display the creation date and the data
        (biweight) median & spread.

        Currently, we have turned off the automatic offset notation of
        `matplotlib`. Maybe this should be the default, which the user may
        override.

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset img
        (`numpy.ndarray` or `xarray.DataArray`) with side-panels and title::

        ...
        """
        # add data statistics to fig_info
        if fig_info is None:
            fig_info = FIGinfo()

        if self._zscale == "quality":
            if "unchanged" in self.attrs["flag_meanings"]:
                fig_info.add(self.attrs["flag_meanings"][3], np.sum(self._image == 4))
                fig_info.add(self.attrs["flag_meanings"][2], np.sum(self._image == 2))
                fig_info.add(self.attrs["flag_meanings"][1], np.sum(self._image == 1))
            else:
                fig_info.add(
                    f'{self.attrs["flag_meanings"][2]}'
                    f' (quality < {self.attrs["thres_bad"]})',
                    np.sum((self._image == 1) | (self._image == 2)),
                )
                fig_info.add(
                    f'{self.attrs["flag_meanings"][1]}'
                    f' (quality < {self.attrs["thres_worst"]})',
                    np.sum(self._image == 1),
                )
        else:
            biwght = Biweight(self._image)
            if self.attrs["units"] == "1":
                fig_info.add("median", biwght.median, "{:.5g}")
                fig_info.add("spread", biwght.spread, "{:.5g}")
            else:
                fig_info.add(
                    "median", (biwght.median, self.attrs["units"]), "{:.5g} {}"
                )
                fig_info.add(
                    "spread", (biwght.spread, self.attrs["units"]), "{:.5g} {}"
                )

        # draw actual image
        self.__draw_image(axx, title)
        if "x-panel" not in axx:
            axx["image"].set_xlabel(self.attrs["dims"][1])
            axx["image"].set_ylabel(self.attrs["dims"][0])
        else:
            self.__draw_side_panels(axx, side_panels)
            axx["x-panel"].set_xlabel(self.attrs["dims"][1])
            axx["y-panel"].set_ylabel(self.attrs["dims"][0])

        self.__add_info_box(axx["info"], fig_info)
