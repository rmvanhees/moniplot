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
"""Definition of the moniplot classes `DrawDetImage` and `DrawDetQuality`."""

from __future__ import annotations

__all__ = ["DrawDetImage", "DrawDetQuality"]

import warnings
from math import log10
from typing import TYPE_CHECKING, Self

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from moniplot.biweight import Biweight
from moniplot.lib.fig_info import FIGinfo
from moniplot.tol_colors import tol_cmap, tol_cset

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from numpy.typing import ArrayLike, NDArray


# - class DrawDetGen definition ------------------------------
class DrawDetGen:
    """Generic class to display detector-data as an image."""

    def __init__(
        self: DrawDetGen,
        arr: NDArray,
        attrs: dict | None = None,
        coords: dict[str, ArrayLike] | None = None,
        side_panels: bool = True,
    ) -> None:
        """Initialize class DrawDetGen."""
        self._cmap = tol_cmap("rainbow_PuRd")
        self._cset = tol_cset("bright")
        self._znorm = None
        self._attrs = (
            attrs
            if attrs is not None
            else {
                "long_name": "",
                "units": "1",
            }
        )
        self._coords = (
            coords
            if coords is not None
            else {
                "row": np.arange(arr.shape[0]),
                "column": np.arange(arr.shape[1]),
            }
        )
        self._image = arr
        self._axx = self._subplots_(side_panels)

    def __enter__(self: DrawDetGen) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: DrawDetGen, *args: object) -> bool:
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self: DrawDetGen) -> None:
        """Finalize all panels."""
        cnames = iter(self._coords.keys())
        if "x-panel" not in self._axx:
            self._axx["image"].set_ylabel(next(cnames))
            self._axx["image"].set_xlabel(next(cnames))
        else:
            self._axx["y-panel"].set_ylabel(next(cnames))
            self._axx["x-panel"].set_xlabel(next(cnames))
        self._draw_colorbar(self._draw_image())

    def _subplots_(self: DrawDetGen, side_panels: bool) -> dict[str, Axes]:
        """Obtain matplotlib Axes for plot-layout.

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
            width_ratios = (1, *width_ratios)
            height_ratios += (1,)
        else:
            mosaic = [["caption", ".", "info"], ["image", "colorbar", "."]]

        _, axx = plt.subplot_mosaic(
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

        return axx

    def _draw_colorbar(self: DrawDetGen, _cm_img: AxesImage) -> None:
        """Add colorbar to figure."""
        return

    @property
    def aspect(self: DrawDetGen) -> int | None:
        """Return aspect-ratio of image data."""
        if self._image is None:
            return None

        return min(4, max(1, round(self._image.shape[1] / self._image.shape[0])))

    def set_caption(self: DrawDetGen, caption: str) -> None:
        """Add caption as a subtitle."""
        self._axx["caption"].text(
            0.5,
            3.5,
            caption,
            fontsize="xx-large",
            horizontalalignment="center",
            verticalalignment="top",
            transform=self._axx["caption"].transAxes,
        )
        # plt.figure.subtitle(
        #    caption,
        #    fontsize="x-large",
        #    position=(0.5, 1 - 0.3 / plt.figure.get_figheight())
        # )

    def set_title(self: DrawDetGen, title: str | None = None) -> None:
        """Add title above the image panel."""
        self._axx["image"].set_title(
            self._attrs["long_name"] if title is None else title
        )

    def _draw_side_panels(self: DrawDetGen, _not_used: str) -> None:
        """Add side-panels to figure."""
        return

    def add_side_panels(self: DrawDetGen, side_panels: str = "default") -> None:
        """Add side-panels left and below the image panel.

        Parameters
        ----------
        side_panels :  str, default='nanmedian'
           Show image row and column statistics in two side panels.
           Valid values are: 'median', 'nanmedian', 'mean', 'nanmean',
           'quality', 'std' and 'nanstd'.

        """
        if "x-panel" in self._axx:
            self._draw_side_panels(side_panels)

    def _add_info_box(self: DrawDetGen, fig_info: FIGinfo) -> None:
        """Add a box with meta information in the current figure.

        Parameters
        ----------
        fig_info :  FIGinfo
            Instance of pys5p.lib.plotlib.FIGinfo to be displayed

        """
        if fig_info is None or fig_info.location != "above":
            return

        self._axx["info"].text(
            2.0,
            0.3,
            fig_info.as_str(),
            fontsize="x-small",
            transform=self._axx["info"].transAxes,
            multialignment="left",
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"facecolor": "white", "pad": 4},
        )

    def add_copyright(self: DrawDetGen, institute: str = "SRON") -> None:
        """Display copyright statement in the lower right corner.

        Parameters
        ----------
        institute: str, default="SRON"
           name of the copyright owner

        """
        self._axx["image"].text(
            1,
            0,
            rf" $\copyright$ {institute}",
            horizontalalignment="right",
            verticalalignment="bottom",
            rotation="vertical",
            fontsize="xx-small",
            transform=self._axx["image"].transAxes,
        )

    def _draw_image(self: DrawDetGen) -> AxesImage:
        """..."""
        # adjust tickmarks
        if (self._image.shape[1] % 10) == 0:
            self._axx["image"].set_xticks(
                np.linspace(0, self._image.shape[1], 6, dtype=int)
            )
        elif (self._image.shape[1] % 8) == 0:
            self._axx["image"].set_xticks(
                np.linspace(0, self._image.shape[1], 5, dtype=int)
            )
        if (self._image.shape[0] % 10) == 0:
            self._axx["image"].set_yticks(
                np.linspace(0, self._image.shape[0], 6, dtype=int)
            )
        elif (self._image.shape[0] % 8) == 0:
            self._axx["image"].set_yticks(
                np.linspace(0, self._image.shape[0], 5, dtype=int)
            )

        return self._axx["image"].imshow(
            self._image,
            cmap=self._cmap,
            norm=self._znorm,
            extent=[0, self._image.shape[1], 0, self._image.shape[0]],
            aspect="auto",
            interpolation="none",
            origin="lower",
        )


# - class DrawDetImage definition ----------------------------
class DrawDetImage(DrawDetGen):
    """Display detector-data as an image, i.e., on a 2D regular raster."""

    # pylint: disable=too-many-arguments
    def __init__(
        self: DrawDetImage,
        arr: NDArray,
        attrs: dict[str, str] | None = None,
        coords: dict[str, ArrayLike] | None = None,
        *,
        vperc: ArrayLike[int, int] | None = None,
        vrange: ArrayLike[float, float] | None = None,
        zscale: str | None = None,
        side_panels: bool = True,
    ) -> None:
        """Initialize class DrawDetImage.

        Parameters
        ----------
        arr :  NDArray
           detector image data (must be 2-D)
        attrs :  dict[str, str], optional
           provide attributes for your data, such as 'long_name' and 'units'.
           Default: 'long_name' is empty and 'units equals '1'
        coords : dict[str, ArrayLike], optional
           provide coordinates of your detector image. Default: 'row' and 'column'
        vperc :  ArrayLike, default=[1, 99]
           range to normalize luminance between percentiles min and max
        vrange :  ArrayLike, default=None
           range to normalize luminance between vmin and vmax.
        zscale :  str, default='linear'
           scale color-map: 'linear', 'log', 'diff' or 'ratio'
        side_panels : bool, optional
           Do you want side_panels with row and column statistics

        """
        self._zlabel = None
        DrawDetGen.__init__(self, arr, attrs, coords, side_panels)

        # set zscale
        if zscale is None:
            self._zscale = "linear"
        elif zscale not in ("diff", "linear", "log", "ratio"):
            raise KeyError(f"provided unknown value for zscale: {zscale}")
        else:
            self._zscale = zscale

        # obtain image-data range
        if vrange is None and vperc is None:
            vmin, vmax = np.nanpercentile(self._image, (1.0, 99.0))
        elif vrange is None:
            if len(vperc) != 2:
                raise KeyError("keyword 'vperc' requires two values")
            vmin, vmax = np.nanpercentile(self._image, vperc)
        else:
            if len(vrange) != 2:
                raise KeyError("keyword 'vrange' requires two values")
            vmin, vmax = vrange

        # optimize data-range accordig to vmin and vmax
        dscale = self.set_zunit(self._attrs["units"], vmin, vmax)
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
        if not (zscale == "ratio" or self._attrs["units"] == "1"):
            self._zlabel += f" [{self._attrs['units']}]"

        # set matplotlib data normalization
        self._znorm = self.set_norm(vmin, vmax)

    def set_norm(self: DrawDetImage, vmin: float, vmax: float) -> mcolors:
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

        if self._zscale == "diff":
            if vmin < 0 < vmax:
                vcntr = 0.0
                tmp1, tmp2 = (vmin, vmax)
                vmin = -max(-tmp1, tmp2)
                vmax = max(-tmp1, tmp2)
            else:
                vcntr = (vmin + vmax) / 2
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

    def set_zunit(self: DrawDetImage, zunits: str, vmin: float, vmax: float) -> int:
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

            self._attrs["units"] = key_to_zunit[key] + zunits
            return 1000**key

        self._attrs["units"] = zunits
        return 1

    def add_fig_info(self: DrawDetGen, fig_info: FIGinfo | None = None) -> None:
        """Add fig_info box to the figure.

        Parameters
        ----------
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure.

        """
        if fig_info is None:
            fig_info = FIGinfo()

        with Biweight(self._image) as bwght:
            if self._attrs["units"] == "1":
                fig_info.add("median", bwght.median, "{:.5g}")
                fig_info.add("spread", bwght.spread, "{:.5g}")
            else:
                fig_info.add(
                    "median", (bwght.median, self._attrs["units"]), "{:.5g} {}"
                )
                fig_info.add(
                    "spread", (bwght.spread, self._attrs["units"]), "{:.5g} {}"
                )
        self._add_info_box(fig_info)

    def _draw_colorbar(self: DrawDetImage, cm_img: AxesImage) -> None:
        """Add colorbar to figure."""
        _ = plt.colorbar(cm_img, cax=self._axx["colorbar"], label=self._zlabel)

    def _draw_side_panels(self: DrawDetImage, side_panels: str) -> None:
        """Add side-panels to figure."""
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
            self._axx["x-panel"].plot(
                xdata,
                func_panels(self._image, axis=0),
                linewidth=0.75,
                color=self._cset.blue,
            )
            self._axx["x-panel"].grid()

        # draw panel left of the image panel
        ydata = np.arange(self._image.shape[0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            self._axx["y-panel"].plot(
                func_panels(self._image, axis=1),
                ydata,
                linewidth=0.75,
                color=self._cset.blue,
            )
        self._axx["y-panel"].grid()


# - class DrawDetQuality definition --------------------------
class DrawDetQuality(DrawDetGen):
    """Display pixel-quality of a detector as an image, i.e., on a 2D regular raster."""

    def __init__(
        self: DrawDetQuality,
        arr: NDArray,
        attrs: dict,
        coords: dict[str, ArrayLike] | None = None,
        *,
        side_panels: bool = True,
    ) -> None:
        """Initialize class DrawDetQuality."""
        DrawDetGen.__init__(self, arr, attrs, coords, side_panels)

        # obligatory attributes
        for key in [
            "colors",
            "long_name",
            "thres_bad",
            "thres_worst",
            "flag_meanings",
            "flag_values",
        ]:
            if key not in self._attrs:
                raise KeyError(f"attribute {key} should be provided")

        self._cmap = mcolors.ListedColormap(self._attrs["colors"])
        self._znorm = mcolors.BoundaryNorm(self._attrs["flag_values"], self._cmap.N)

    def _draw_colorbar(self: DrawDetQuality, cm_img: AxesImage) -> None:
        """Add colorbar to figure."""
        bounds = self._attrs["flag_values"]
        mbounds = [(bounds[ii + 1] + bounds[ii]) / 2 for ii in range(len(bounds) - 1)]
        _ = plt.colorbar(
            cm_img,
            cax=self._axx["colorbar"],
            ticks=mbounds,
            boundaries=bounds,
        )
        self._axx["colorbar"].tick_params(axis="y", which="both", length=0)
        self._axx["colorbar"].set_yticklabels(self._attrs["flag_meanings"])

    def _draw_side_panels(self: DrawDetQuality, _not_used: str) -> None:
        """..."""
        # draw panel below the image panel
        xdata = np.arange(self._image.shape[1])
        if len(self._attrs["flag_values"]) == 4:
            ydata = self._image.clip(0, None).sum(axis=0)
            self._axx["x-panel"].step(
                xdata, ydata, linewidth=0.75, color=self._cset.yellow
            )
        else:
            ydata = np.sum(((self._image == 1) | (self._image == 2)), axis=0)
            self._axx["x-panel"].step(
                xdata, ydata, linewidth=0.75, color=self._cset.yellow
            )
            ydata = np.sum((self._image == 1), axis=0)  # worst
            self._axx["x-panel"].step(
                xdata, ydata, linewidth=0.75, color=self._cset.red
            )
        if len(self._attrs["flag_values"]) == 6:
            ydata = np.sum((self._image == 4), axis=0)  # to_good
            self._axx["x-panel"].step(
                xdata, ydata, linewidth=0.75, color=self._cset.green
            )
        self._axx["x-panel"].grid()

        # draw panel left of the image panel
        ydata = np.arange(self._image.shape[0])
        if len(self._attrs["flag_values"]) == 4:
            xdata = self._image.clip(0, None).sum(axis=1)
            self._axx["y-panel"].step(
                xdata, ydata, linewidth=0.75, color=self._cset.yellow
            )
        else:
            xdata = np.sum(((self._image == 1) | (self._image == 2)), axis=1)
            self._axx["y-panel"].step(
                xdata, ydata, linewidth=0.75, color=self._cset.yellow
            )
            xdata = np.sum(self._image == 1, axis=1)  # worst
            self._axx["y-panel"].step(
                xdata, ydata, linewidth=0.75, color=self._cset.red
            )
        if len(self._attrs["flag_values"]) == 6:
            xdata = np.sum(self._image == 4, axis=1)  # to_good
            self._axx["y-panel"].step(
                xdata, ydata, linewidth=0.75, color=self._cset.green
            )
        self._axx["y-panel"].grid()

    def add_fig_info(self: DrawDetQuality, fig_info: FIGinfo | None = None) -> None:
        """Add fig_info box to the figure.

        Parameters
        ----------
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure.

        """
        if fig_info is None:
            fig_info = FIGinfo()

        if "unchanged" in self._attrs["flag_meanings"]:
            fig_info.add(self._attrs["flag_meanings"][3], np.sum(self._image == 4))
            fig_info.add(self._attrs["flag_meanings"][2], np.sum(self._image == 2))
            fig_info.add(self._attrs["flag_meanings"][1], np.sum(self._image == 1))
        elif self._attrs["thres_worst"] is None:
            fig_info.add(
                f"{self._attrs['flag_meanings'][2]} quality",
                (self._image[self._image > 0]).sum(),
            )
        else:
            fig_info.add(
                f"{self._attrs['flag_meanings'][2]}"
                f" (quality < {self._attrs['thres_bad']})",
                ((self._image == 1) | (self._image == 2)).sum(),
            )
            fig_info.add(
                f"{self._attrs['flag_meanings'][1]}"
                f" (quality < {self._attrs['thres_worst']})",
                (self._image == 1).sum(),
            )
        self._add_info_box(fig_info)
