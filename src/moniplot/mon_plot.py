#
# This file is part of moniplot
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
"""This module contains the class `MONplot`.

The methods of the class `MONplot` are:
 `draw_hist`, `draw_lplot`, `draw_multiplot`, `draw_qhist`, `draw_quality`,
 `draw_signal`, `draw_tracks`, `draw_trend`, draw_fov_ckd.
"""
from __future__ import annotations

__all__ = ["MONplot"]

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import xarray as xr

try:
    from cartopy import crs as ccrs
except ModuleNotFoundError:
    FOUND_CARTOPY = False
else:
    FOUND_CARTOPY = True
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec

from .biweight import Biweight
from .fig_draw_image import (
    adjust_img_ticks,
    fig_data_to_xarr,
    fig_draw_panels,
    fig_qdata_to_xarr,
)
from .fig_draw_lplot import close_draw_lplot, fig_draw_lplot
from .fig_draw_multiplot import draw_subplot, get_xylabels
from .fig_draw_qhist import fig_draw_qhist
from .fig_draw_trend import add_hk_subplot, add_subplot
from .lib.fig_info import FIGinfo
from .tol_colors import tol_rgba

if FOUND_CARTOPY:
    from .fig_draw_tracks import fig_draw_tracks

if TYPE_CHECKING:
    from matplotlib import colormaps
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# - global variables -------------------------------
DEFAULT_CSET = "bright"


# - local functions --------------------------------
class DictMpl(TypedDict):
    """Dataclass to hold matplotlib Figure and Axes for method `draw_lplot`."""
    fig: Figure | None
    axx: Axes | None
    time_axis: bool


# - main function ----------------------------------
class MONplot:
    """
    Generate PDF reports (or figures) for instrument calibration or monitoring.

    Parameters
    ----------
    figname :  Path | str
        Name of PDF or PNG file (extension required)
    caption :  str, optional
        Caption repeated on each page of the PDF

    Notes
    -----
    The methods of the class `MONplot` will accept `numpy` arrays as input and
    display your data without knowledge on the data units and coordinates.
    In most cases, this will be enough for a quick inspection of your data.
    However, when you use the labeled arrays and datasets of `xarray`then
    the software will use the name of the xarray class, coordinate names and
    data attributes, such as `long_name` and `units`.
    """

    def __init__(
        self: MONplot, figname: Path | str, caption: str | None = None
    ) -> None:
        """Initialize multi-page PDF document or a single-page PNG."""
        self.__cset = tol_rgba(DEFAULT_CSET)
        self.__cmap = None
        self.__caption = "" if caption is None else caption
        self.__institute = ""
        self.__mpl: DictMpl = {"fig": None, "axx": None, "time_axis": False}
        self.__pdf = None
        self.filename = Path(figname)
        if self.filename.suffix.lower() != ".pdf":
            return

        self.__pdf = PdfPages(self.filename)

        # turn-off the automatic offset notation of Matplotlib
        mpl.rcParams["axes.formatter.useoffset"] = False

    def __close_this_page(self: MONplot, fig: Figure) -> None:
        """Save the current figure and close the MONplot instance."""
        # add save figure
        if self.__pdf is None:
            plt.savefig(self.filename)
            plt.close(fig)
        else:
            self.__pdf.savefig()

    def close(self: MONplot) -> None:
        """Close PNG or (multipage) PDF document."""
        if self.__pdf is None:
            return

        # add PDF annotations
        doc = self.__pdf.infodict()
        if self.__caption is not None:
            doc["Title"] = self.__caption
        doc["Subject"] = "Generated using https://github.com/rmvanhees/moniplot.git"
        if self.__institute == "SRON":
            doc["Author"] = "(c) SRON Netherlands Institute for Space Research"
        elif self.__institute:
            doc["Author"] = f"(c) {self.__institute}"
        self.__pdf.close()
        plt.close("all")

    @property
    def caption(self: MONplot) -> str:
        """Returns caption of figure."""
        return self.__caption

    def set_caption(self: MONplot, caption: str) -> None:
        """Set caption of each page of the PDF.

        Parameters
        ----------
        caption :  str
           Default title of all pages at the top of the page.
        """
        self.__caption = caption

    def __add_caption(self: MONplot, fig: Figure) -> None:
        """Add figure caption."""
        if not self.caption:
            return

        fig.suptitle(
            self.caption,
            fontsize="x-large",
            position=(0.5, 1 - 0.3 / fig.get_figheight()),
        )

    # --------------------------------------------------
    @property
    def cmap(self: MONplot) -> colormaps:
        """Returns current Matplotlib colormap."""
        return self.__cmap

    def set_cmap(self: MONplot, cmap: colormaps) -> None:
        """Use alternative colormap for MONplot::draw_image.

        Parameters
        ----------
        cmap :  matplotlib colormap
        """
        self.__cmap = cmap

    def unset_cmap(self: MONplot) -> None:
        """Unset user supplied colormap, and use default colormap."""
        self.__cmap = None

    # --------------------------------------------------
    @property
    def cset(self: MONplot) -> np.ndarray:
        """Returns name of current color-set."""
        return self.__cset

    def set_cset(self: MONplot, cname: str, cnum: int | None = None) -> None:
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
        self.__cset = tol_rgba(cname, cnum)

    def unset_cset(self: MONplot) -> None:
        """Set color set to its default."""
        self.__cset = tol_rgba(DEFAULT_CSET)

    # --------------------------------------------------
    @property
    def institute(self: MONplot) -> str:
        """Returns name of institute."""
        return self.__institute

    def set_institute(self: MONplot, institute: str) -> None:
        """Use the name of your institute as a signature.

        Parameters
        ----------
        institute :  str
           Provide abbreviation of the name of your institute to be used in
           the copyright statement in the main panel of the figures.
        """
        self.__institute = institute

    # --------------------------------------------------
    def __add_copyright(self: MONplot, axx: Axes) -> None:
        """Display copyright statement in the lower right corner."""
        if not self.institute:
            return

        axx.text(
            1,
            0,
            rf" $\copyright$ {self.institute}",
            horizontalalignment="right",
            verticalalignment="bottom",
            rotation="vertical",
            fontsize="xx-small",
            transform=axx.transAxes,
        )

    @staticmethod
    def __add_fig_box(fig: Figure, fig_info: FIGinfo) -> None:
        """Add a box with meta information in the current figure.

        Parameters
        ----------
        fig :  matplotlib.figure.Figure
        fig_info :  FIGinfo
           instance of pys5p.lib.plotlib.FIGinfo to be displayed
        """
        if fig_info is None or fig_info.location != "above":
            return

        xpos = 1 - 0.4 / fig.get_figwidth()
        ypos = 1 - 0.25 / fig.get_figheight()

        fig.text(
            xpos,
            ypos,
            fig_info.as_str(),
            fontsize="x-small",
            style="normal",
            verticalalignment="top",
            horizontalalignment="right",
            multialignment="left",
            bbox={"facecolor": "white", "pad": 5},
        )

    # -------------------------
    def __draw_image__(
        self: MONplot,
        xarr: xr.DataArray,
        side_panels: str,
        fig_info: FIGinfo | None,
        title: str | None,
    ) -> None:
        """Display image data.

        Called by the public methods `draw_signal` and `draw_quality`.
        """

        def add_fig_box() -> None:
            """Add a box with meta information in the current figure."""
            if fig_info is None:
                return

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

        # aspect of image data
        aspect = min(4, max(1, int(round(xarr.shape[1] / xarr.shape[0]))))

        # select figure attributes
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

        # define matplotlib figure
        fig = plt.figure(figsize=attrs["figsize"])
        if self.caption:
            fig.suptitle(
                self.caption,
                fontsize="x-large",
                position=(0.5, 1 - 0.4 / fig.get_figheight()),
            )

        # Define a grid layout to place subplots within the figure.
        # - gspec[0, 1] is reserved for the image
        # - gspec[1, 1] is reserved for the x-panel
        # - gspec[0, 0] is reserved for the y-panel
        # - gspec[0, 2] is reserved for the colorbar
        # - gspec[1, 2] is used to pace the small fig_info box (max 6/7 lines)
        gspec = fig.add_gridspec(
            2,
            4,
            left=0.135 + 0.005 * (aspect - 1),
            right=0.9 - 0.005 * (aspect - 1),
            top=0.865 - 0.025 * (aspect - 1),
            bottom=0.115 + 0.01 * (aspect - 1),
            wspace=0.1 / max(2, aspect - 1),
            hspace=0.05,
            width_ratios=attrs["w_ratios"],
            height_ratios=attrs["h_ratios"],
        )

        # add image panel and draw image
        axx = fig.add_subplot(gspec[0, 1])
        if xarr.attrs["_zscale"] == "quality":
            img = axx.imshow(
                xarr.values,
                norm=xarr.attrs["_znorm"],
                aspect="auto",
                cmap=xarr.attrs["_cmap"],
                interpolation="none",
                origin="lower",
            )
        else:
            cmap = self.cmap if self.cmap else xarr.attrs["_cmap"]
            img = axx.imshow(
                xarr.values,
                norm=xarr.attrs["_znorm"],
                aspect="auto",
                cmap=cmap,
                interpolation="none",
                origin="lower",
            )

        # add title to image panel
        if title is not None:
            axx.set_title(title)
        elif "long_name" in xarr.attrs:
            axx.set_title(xarr.attrs["long_name"])
        self.__add_copyright(axx)
        # axx.grid(True)

        # add side panels
        if side_panels == "none":
            adjust_img_ticks(axx, xarr)
            axx.set_xlabel(xarr.dims[1])
            axx.set_ylabel(xarr.dims[0])
        else:
            for xtl in axx.get_xticklabels():
                xtl.set_visible(False)
            for ytl in axx.get_yticklabels():
                ytl.set_visible(False)
            axx_p = {
                "X": fig.add_subplot(gspec[1, 1], sharex=axx),
                "Y": fig.add_subplot(gspec[0, 0], sharey=axx),
            }
            fig_draw_panels(axx_p, xarr, side_panels)
            axx_p["X"].set_xlabel(xarr.dims[1])
            axx_p["Y"].set_ylabel(xarr.dims[0])

        # add colorbar
        if xarr.attrs["_zscale"] == "quality":
            axx_c = fig.add_subplot(gspec[0, 2])
            bounds = xarr.attrs["flag_values"]
            mbounds = [
                (bounds[ii + 1] + bounds[ii]) / 2 for ii in range(len(bounds) - 1)
            ]
            _ = plt.colorbar(img, cax=axx_c, ticks=mbounds, boundaries=bounds)
            axx_c.tick_params(axis="y", which="both", length=0)
            axx_c.set_yticklabels(xarr.attrs["flag_meanings"])
        else:
            axx_c = fig.add_subplot(gspec[0, 2])
            _ = plt.colorbar(img, cax=axx_c, label=xarr.attrs["_zlabel"])

        # add annotation and save the figure
        add_fig_box()
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_signal(
        self: MONplot,
        data: xr.DataArray | np.ndarray,
        *,
        fig_info: FIGinfo | None = None,
        side_panels: str = "nanmedian",
        title: str | None = None,
        **kwargs: int,
    ) -> None:
        """Display 2D array as an image.

        Averaged column/row signal are optionally displayed in side-panels.

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes.
        fig_info :  FIGinfo, <default=None
           OrderedDict holding meta-data to be displayed in the figure.
        side_panels :  str, default='nanmedian'
           Show image row and column statistics in two side panels.
           Use 'none' when you do not want the side panels.
           Other valid values are: 'median', 'nanmedian', 'mean', 'nanmean',
           'quality', 'std' and 'nanstd'.
        title :  str, default=None
           Title of this figure using `Axis.set_title`.
        **kwargs :   other keywords
           Pass keyword arguments: `zscale`, `vperc` or `vrange`
           to `moniplot.lib.fig_draw_image.fig_data_to_xarr()`.

        See also
        --------
        fig_data_to_xarr : Prepare image data for plotting.

        Notes
        -----
        When data is a xarray.DataArray then the following attributes
        are used::

        'long_name' : used as the title of the main panel when parameter \
                      `title` is not defined.
        '_cmap'     : contains the matplotlib colormap
        '_zlabel'   : contains the label of the color bar
        '_znorm'    : matplotlib class to normalize the data between zero \
                      and one.
        '_zscale'   : scaling of the data values: linear, log, diff, ratio, ...
        '_zunits'   : adjusted units of the data

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

        > plot = MONplot('test.pdf', caption='my caption')
        > plot.set_institute('SRON')
        > plot.draw_signal(img, title='my title')

        Add the same figure without side-panels::

        > plot.draw_signal(img, side_panels='none', title='my title')

        Add a figure using a fixed data-range that the colormap covers::

        > plot.draw_signal(img1, title='my title', vrange=[zmin, zmax])

        Add a figure where img2 = img - img_ref::

        > plot.draw_signal(img2, title='my title', zscale='diff')

        Add a figure where img2 = img / img_ref::

        > plot.draw_signal(img2, title='my title', zscale='ratio')

        Finalize the PDF file::

        > plot.close()
        """
        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and "_zscale" in data.attrs:
            xarr = data.copy()
        else:
            xarr = fig_data_to_xarr(data, **kwargs)

        # add data statistics to fig_info
        if fig_info is None:
            fig_info = FIGinfo()

        biwght = Biweight(xarr.values)
        if xarr.attrs["_zunits"] is None or xarr.attrs["_zunits"] == "1":
            fig_info.add("median", biwght.median, "{:.5g}")
            fig_info.add("spread", biwght.spread, "{:.5g}")
        else:
            fig_info.add("median", (biwght.median, xarr.attrs["_zunits"]), "{:.5g} {}")
            fig_info.add("spread", (biwght.spread, xarr.attrs["_zunits"]), "{:.5g} {}")

        # draw actual image
        self.__draw_image__(xarr, side_panels, fig_info, title)

    # --------------------------------------------------
    def draw_quality(
        self: MONplot,
        data: xr.DataArray | np.ndarray,
        ref_data: np.ndarray | None = None,
        *,
        side_panels: str = "quality",
        fig_info: FIGinfo | None = None,
        title: str | None = None,
        **kwargs: int,
    ) -> None:
        """Display pixel-quality 2D array as image with column/row statistics.

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes.
        ref_data :  numpy.ndarray, default=None
           Numpy array holding reference data, for example pixel quality
           reference map taken from the CKD. Shown are the changes with
           respect to the reference data.
        fig_info :  FIGinfo, default=None
           OrderedDict holding meta-data to be displayed in the figure.
        side_panels :  str, default='quality'
           Show image row and column statistics in two side panels.
           Use 'none' when you do not want the side panels.
        title :  str, default=None
           Title of this figure using `Axis.set_title`.
        **kwargs :   other keywords
           Pass keyword arguments: `data_sel`, `thres_worst`, `thres_bad`
           or `qlabels` to `moniplot.lib.fig_draw_image.fig_qdata_to_xarr`.

        See Also
        --------
        qdata_to_xarr : Prepare pixel-quality data for plotting.

        Notes
        -----
        When data is a xarray.DataArray then the following attributes
        are used::

        'long_name'     : used as the title of the main panel when parameter \
                          `title` is not defined.
        'flag_values'   : values of the flags used to qualify the pixel quality
        'flag_meanings' : description of the flag values
        'thres_bad'     : threshold between good and bad
        'thres_worst'   : threshold between bad and worst
        '_cmap'         : contains the matplotlib colormap
        '_zscale'       : should be 'quality'
        '_znorm'        : matplotlib class to normalize the data between zero \
                          and one

        The quality ranking labels are ['unusable', 'worst', 'bad', 'good'],
        when no reference dataset is provided. Where::

        'unusable'  : pixels outside the illuminated region
        'worst'     : 0 <= value < thres_worst
        'bad'       : 0 <= value < thres_bad
        'good'      : thres_bad <= value <= 1

        Otherwise, the labels for quality ranking indicate which pixels have
        changed w.r.t. reference. The labels are::

        'unusable'  : pixels outside the illuminated region
        'worst'     : from good or bad to worst
        'bad'       : from good to bad
        'good'      : from any rank to good
        'unchanged' : no change in rank

        The information provided in the parameter `fig_info` will be displayed
        in a small box. Where creation date and statistics on the number of
        bad and worst pixels are displayed.

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset img
        (`numpy.ndarray` or `xarray.DataArray`) with side-panels and title::

        > plot = MONplot('test.pdf', caption='my caption', institute='SRON')
        > plot.draw_quality(img, title='my title')

        Add the same figure without side-panels::

        > plot.draw_quality(img, side_panels='none', title='my title')

        Add a figure where img_ref is a quality map from early in the mission::

        > plot.draw_quality(img, img_ref, title='my title')

        Finalize the PDF file::

        > plot.close()
        """
        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and "_zscale" in data.attrs:
            xarr = data
        else:
            xarr = fig_qdata_to_xarr(data, ref_data, **kwargs)

        # add statistics on data quality to fig_info
        if fig_info is None:
            fig_info = FIGinfo()

        if ref_data is None:
            fig_info.add(
                f'{xarr.attrs["flag_meanings"][2]}'
                f' (quality < {xarr.attrs["thres_bad"]})',
                np.sum((xarr.values == 1) | (xarr.values == 2)),
            )
            fig_info.add(
                f'{xarr.attrs["flag_meanings"][1]}'
                f' (quality < {xarr.attrs["thres_worst"]})',
                np.sum(xarr.values == 1),
            )
        else:
            fig_info.add(xarr.attrs["flag_meanings"][3], np.sum(xarr.values == 4))
            fig_info.add(xarr.attrs["flag_meanings"][2], np.sum(xarr.values == 2))
            fig_info.add(xarr.attrs["flag_meanings"][1], np.sum(xarr.values == 1))

        # draw actual image
        self.__draw_image__(xarr, side_panels, fig_info, title)

    # --------------------------------------------------
    def draw_trend(
        self: MONplot,
        xds: xr.Dataset | None = None,
        hk_xds: xr.Dataset | None = None,
        *,
        fig_info: FIGinfo | None = None,
        title: str | None = None,
        **kwargs: int,
    ) -> None:
        """
        Display trends of measurement data and/or housekeeping data.

        Parameters
        ----------
        xds :  xarray.Dataset, optional
           Object holding measurement data and attributes.
        hk_xds :  xarray.Dataset, optional
           Object holding housekeeping data and attributes.
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure.
        title :  str, optional
           Title of this figure using `Axis.set_title`.
        **kwargs :   other keywords
           Pass keyword arguments: 'vperc' or 'vrange_last_orbits'
           to 'moniplot.lib.fig_draw_trend.add_hk_subplot`.

        See Also
        --------
        add_subplot : Add a subplot for measurement data.
        add_hk_subplot : Add a subplot for housekeeping data.

        Notes
        -----
        When data is a xarray.DataArray then the following attributes are used:

        - long_name: used as the title of the main panel when parameter 'title'\
          is not defined.
        - units: units of the data

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset 'xds'
        (`numpy.ndarray` or `xarray.DataArray`) with a title. The dataset 'xds'
        may contain multiple DataArrays with a common X-coordinate. Each
        DataArray will be displayed in a separate sub-panel::

        > plot = MONplot('test.pdf', caption='my caption', institute='SRON')
        > plot.draw_trend(xds, hk_xds=None, title='my title')

        Add a figure with the same Dataset 'xds' and a few trends of
        housekeeping data (again each parameter in a separate DataArray with
        a common X-coordinate)::

        > plot.draw_trend(xds, hk_xds, title='my title')

        Finalize the PDF file::

        > plot.close()


        The following Dataset-definitions for measurement data are accepted::

          <xarray.Dataset>
          Dimensions:           (days: 1, hours: 480)
          Coordinates:
            * days              (days) int32 1690329600
            * hours             (hours) float64 0.0 0.05 0.1 ... 23.9 23.95
          Data variables:
            value             (days, hours) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...

          <xarray.Dataset>
          Dimensions:           (days: 1, hours: 480)
          Coordinates:
            * days              (days) int32 1690329600
            * hours             (hours) float64 0.0 0.05 0.1 ... 23.9 23.95
          Data variables:
            value             (days, hours) float64 ...

          <xarray.Dataset>
          Dimensions:           (orbit: 480)
          Coordinates:
            * orbit             (orbit) uint32 12345 12346 ... 12823 12824
          Data variables:
            value             (orbit) float64 ...


        The following Dataset-definitions for housekeeping data are accepted::

         <xarray.Dataset>
         Dimensions:           (days: 1, hours: 480)
         Coordinates:
           * days              (days) int32 1690329600
           * hours             (hours) float64 0.0 0.05 0.1 ... 23.9 23.95
         Data variables:
           TS1_DEM_N_T       (days, hours) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           TS2_HOUSING_N_T   (days, hours) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           TS3_RADIATOR_N_T  (days, hours) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           TS4_DEM_R_T       (days, hours) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           TS5_HOUSING_R_T   (days, hours) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           TS6_RADIATOR_R_T  (days, hours) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...

         <xarray.Dataset>
         Dimensions:           (orbit: 480)
         Coordinates:
           * orbit             (orbit) uint32 12345 12346 ... 12823 12824
         Data variables:
           detector_temp     (orbit) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           grating_temp      (orbit) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           imager_temp       (orbit) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           obm_temp          (orbit) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...
           calib_unit_temp   (orbit) [('mean', '<f8'), ('err1', '<f8'), ('err2', '<f8')] ...

        Where each DataArray's have attributes 'long_name' and 'units'.
        """  # noqa: E501
        if xds is None and hk_xds is None:
            raise ValueError("both xds and hk_xds are None")
        if xds is not None and not isinstance(xds, xr.Dataset):
            raise ValueError("xds should be and xarray Dataset object")
        if hk_xds is not None and not isinstance(hk_xds, xr.Dataset):
            raise ValueError("hk_xds should be and xarray Dataset object")

        if fig_info is None:
            fig_info = FIGinfo()

        # determine npanels from xarray Dataset
        npanels = len(xds.data_vars) if xds is not None else 0
        npanels += len(hk_xds.data_vars) if hk_xds is not None else 0

        # initialize matplotlib using 'subplots'
        figsize = (10.0, 1 + (npanels + 1) * 1.5)
        fig, axarr = plt.subplots(npanels, sharex="all", figsize=figsize)
        if npanels == 1:
            axarr = [axarr]
        margin = min(1.0 / (1.65 * (npanels + 1)), 0.25)
        fig.subplots_adjust(bottom=margin, top=1 - margin, hspace=0.05)

        # add a centered subtitle to the figure
        self.__add_caption(fig)

        # add title to image panel
        if title is not None:
            axarr[0].set_title(title)

        # add figures with trend data
        ipanel = 0
        xlabel = None
        if xds is not None:
            if "orbit" in xds.coords:
                xlabel = "orbit"
            elif "hours" in xds.coords:
                xlabel = "time [hours]"
            else:
                xlabel = "time"
                plt.gcf().autofmt_xdate()
                plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            for name in xds.data_vars:
                add_subplot(
                    axarr[ipanel], xds[name], scatter=kwargs.get("scatter", False)
                )
                ipanel += 1

        xlabel_hk = None
        if hk_xds is not None:
            if "orbit" in hk_xds.coords:
                xlabel_hk = "orbit"
            elif "hours" in hk_xds.coords:
                xlabel_hk = "time [hours]"
            else:
                xlabel_hk = "time"
                plt.gcf().autofmt_xdate()
                plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            for name in hk_xds.data_vars:
                add_hk_subplot(
                    axarr[ipanel],
                    hk_xds[name],
                    vperc=kwargs.get("vperc", None),
                    vrange_last_orbits=kwargs.get("vrange_last_orbits", -1),
                )
                ipanel += 1

        # finally add a label for the X-coordinate
        if xlabel is not None and xlabel_hk is not None:
            if xlabel != xlabel_hk:
                raise ValueError(
                    "measurement and housekeeping data have different x-coordinates"
                )
        elif xlabel is None and xlabel_hk is None:
            xlabel = "x-axis"
        elif xlabel is None:
            xlabel = xlabel_hk
        axarr[-1].set_xlabel(xlabel)

        # add annotation and save the figure
        self.__add_copyright(axarr[-1])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_hist(
        self: MONplot,
        data: xr.DataArray | np.ndarray,
        data_sel: tuple[slice | int] | None = None,
        vrange: list[float, float] | None = None,
        fig_info: FIGinfo | None = None,
        title: str | None = None,
        **kwargs: int,
    ) -> None:
        r"""Display data as histograms.

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes.
        data_sel :  mask or index tuples for arrays, optional
           Select a region on the detector by fancy indexing (using a
           boolean/integer arrays), or using index tuples for arrays
           (generated with `numpy.s\_`).
        vrange :  list[float, float], default=[data.min(), data.max()]
           The lower and upper range of the bins.
           Note data will also be clipped according to this range.
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure.
        title :  str, optional
           Title of this figure using `Axis.set_title`.
           Default title is 'f'Histogram of {data.attrs["long_name"]}'.
        **kwargs :   other keywords
           Pass the following keyword arguments to `matplotlib.pyplot.hist`:
           'bins', 'density' or 'log'.
           Note that keywords: 'histtype', 'color', 'linewidth' and 'fill'
           are predefined.

        See Also
        --------
        matplotlib.pyplot.hist : Compute and plot a histogram.

        Notes
        -----
        When data is a xarray.DataArray then the following attributes are used:

        - long_name: used as the title of the main panel when parameter 'title'\
          is not defined.
        - units: units of the data

        Examples
        --------
        Create a PDF document 'test.pdf' with two pages.
        Both pages have the caption "My Caption", the title of the figure on
        the first page is "my title" and on the second page the title of the
        figure is `f"Histogram of {xarr.attrs['long_name']}"`, if xarr has
        attribute "long_name"::

        > plot = MONplot('test.pdf', caption='My Caption')
        > plot.set_institute('SRON')
        > plot.draw_hist(data, title='my title')
        > plot.draw_hist(xarr)
        > plot.close()
        """
        long_name = ""
        zunits = "1"
        if isinstance(data, xr.DataArray):
            if data_sel is None:
                values = data.values.reshape(-1)
            else:
                values = data.values[data_sel].reshape(-1)
            if "units" in data.attrs:
                zunits = data.attrs["units"]
            if "long_name" in data.attrs:
                long_name = data.attrs["long_name"]
        else:
            if data_sel is None:
                values = data.reshape(-1)
            else:
                values = data[data_sel].reshape(-1)

        # add data statistics to fig_info
        if fig_info is None:
            fig_info = FIGinfo()

        biwght = Biweight(values)
        if zunits == "1":
            fig_info.add("median", biwght.median, "{:.5g}")
            fig_info.add("spread", biwght.spread, "{:.5g}")
        else:
            fig_info.add("median", (biwght.median, zunits), "{:.5g} {}")
            fig_info.add("spread", (biwght.spread, zunits), "{:.5g} {}")

        # create figure
        fig, axx = plt.subplots(1, figsize=(9, 8))

        # add a centered subtitle to the figure
        self.__add_caption(fig)

        # add title to image panel and set xlabel
        xlabel = "value" if zunits == "1" else f"value [{zunits}]"
        if title is None:
            title = f"Histogram of {long_name}"
        elif long_name:
            xlabel = long_name if zunits == "1" else f"{long_name} [{zunits}]"
        axx.set_title(title)

        # add histogram
        if vrange is not None:
            values = np.clip(values, vrange[0], vrange[1])
        # Edgecolor is tol_cset('bright').blue
        if "bins" in kwargs and kwargs["bins"] > 24:
            axx.hist(
                values,
                range=vrange,
                histtype="step",
                edgecolor="#4477AA",
                facecolor="#77AADD",
                fill=True,
                linewidth=1.5,
                **kwargs,
            )
            axx.grid(which="major", color="#AAAAAA", linestyle="--")
        else:
            axx.hist(
                values,
                range=vrange,
                histtype="bar",
                edgecolor="#4477AA",
                facecolor="#77AADD",
                linewidth=1.5,
                **kwargs,
            )
            axx.grid(which="major", axis="y", color="#AAAAAA", linestyle="--")
        axx.set_xlabel(xlabel)
        if "density" in kwargs and kwargs["density"]:
            axx.set_ylabel("density")
        else:
            axx.set_ylabel("number")

        if len(fig_info) > 3:
            plt.subplots_adjust(top=0.875)

        # add annotation and save the figure
        self.__add_copyright(axx)
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_qhist(
        self: MONplot,
        xds: xr.DataArray,
        data_sel: tuple[slice, int] | None = None,
        density: bool = True,
        fig_info: FIGinfo | None = None,
        title: str | None = None,
    ) -> None:
        r"""Display pixel-quality data as histograms.

        Parameters
        ----------
        xds :  xarray.Dataset
           Object holding measurement data and attributes.
        data_sel :  mask or index tuples for arrays, optional
           Select a region on the detector by fancy indexing (using a
           boolean/interger arrays), or using index tuples for arrays
           (generated with `numpy.s\_`).
        density : bool, default=True
           If True, draw and return a probability density: each bin will
           display the bin's raw count divided by the total number of counts
           and the bin width (see `matplotlib.pyplot.hist`).
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure.
        title :  str, optional
           Title of this figure using `Axis.set_title`.

        See Also
        --------
        matplotlib.pyplot.hist : Compute and plot a histogram.

        Notes
        -----
        When data is a xarray.DataArray then the following attributes are used:

        - long_name: used as the title of the main panel when parameter 'title'\
          is not defined.
        - units: units of the data

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset 'xds'
        (`numpy.ndarray` or `xarray.DataArray`) with a title. The dataset 'xds'
        may contain multiple DataArrays with a common X-coordinate. Each
        DataArray will be displayed in a seperate sub-panel.

        >>> plot = MONplot('test.pdf', caption='my caption', institute='SRON')
        >>> plot.draw_qhist(xds, title='my title')
        >>> plot.close()

        """
        if not isinstance(xds, xr.Dataset):
            raise ValueError("xds should be and xarray Dataset object")

        if fig_info is None:
            fig_info = FIGinfo()

        # determine npanels from xarray Dataset
        npanels = len(xds.data_vars)

        # initialize matplotlib using 'subplots'
        figsize = (10.0, 1 + (npanels + 1) * 1.65)
        fig, axarr = plt.subplots(npanels, sharex="all", figsize=figsize)
        if npanels == 1:
            axarr = [axarr]
        margin = min(1.0 / (1.8 * (npanels + 1)), 0.25)
        fig.subplots_adjust(bottom=margin, top=1 - margin, hspace=0.02)

        # add a centered subtitle to the figure
        self.__add_caption(fig)

        # add title to image panel
        if title is None:
            title = "Histograms of pixel-quality"
        axarr[0].set_title(title)

        # add figures with histograms
        for ii, (key, xda) in enumerate(xds.data_vars.items()):
            if data_sel is None:
                qdata = xda.values.reshape(-1)
            else:
                qdata = xda.values[data_sel].reshape(-1)
            label = xda.attrs["long_name"] if "long_name" in xda.attrs else key
            fig_draw_qhist(axarr[ii], qdata, label, density)

        # finally add a label for the X-coordinate
        axarr[-1].set_xlabel("pixel quality")

        # add annotation and save the figure
        self.__add_copyright(axarr[-1])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_lplot(
        self: MONplot,
        xdata: np.ndarray | None = None,
        ydata: np.ndarray | None = None,
        *,
        square: bool = False,
        fig_info: FIGinfo | None = None,
        title: str | None = None,
        kwlegend: dict | None = None,
        **kwargs: int,
    ) -> None:
        """Create line-plot of y-data versus x-data.

        Can be called multiple times to add lines. Figure must be closed
        with y-data equals None.

        Parameters
        ----------
        xdata :  ndarray, optional
           ``[add line]`` X-data.
        ydata :  ndarray, optional
           ``[add line]`` Y-data, or
           ``[close figure]`` when ydata is None.
        square :  bool, default=False
           ``[add line]`` create a square figure,
           independent of number of data-points (*first call, only*).
        fig_info  :  FIGinfo, optional
           ``[close figure]`` Meta-data to be displayed in the figure.
        title :  str, optional
           ``[close figure]`` Title of figure (using `Axis.set_title`).
        kwlegend :  dict, optional
           ``[close figure]`` Provide keywords for the function `Axes.legend`.
           Default: {'fontsize': 'small', 'loc': 'best'}
        **kwargs :  other keywords
           ``[add line]`` Keywords are passed to `matplotlib.pyplot.plot`;
           ``[close figure]`` Keywords are passed to appropriate
           `matplotlib.Axes` method, and the keyword 'text' can be used to
           add addition text in the upper left corner.

        Examples
        --------
        General example::

        > plot = MONplot(fig_name)
        > for ii, xarr, yarr in enumerate(data_of_each_line):
        ...   plot.draw_lplot(xarr, yarr, label=mylabel[ii], marker='o')

        > plot.draw_lplot(xlim=[0, 0.5], ylim=[-10, 10],
        ...   xlabel='x-axis, ylabel=y-axis')
        > plot.close()

        Using a time-axis::

        > from datetime import datetime, timedelta
        > tt0 = (datetime(year=2020, month=10, day=1)
        ...      + timedelta(seconds=sec_in_day))
        > tt = [tt0 + iy * t_step for iy in range(yy.size)]
        > plot = MONplot(fig_name)
        > plot.draw_lplot(tt, yy, label='mylabel', marker='o')
        > plot.draw_lplot(ylim=[-10, 10], xlabel='t-axis', ylabel='y-axis')
        > plot.close()

        You can use different sets of colors and cycle through them.
        First, we use default colors defined by matplotlib::

        > plot = MONplot('test_lplot.pdf')
        > plot.set_cset(None)
        > for ii in range(5):
        ...   plot.draw_lplot(np.arange(10), np.arange(10)*(ii+1))
        > plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
        ...   title='draw_lplot [cset is None]')

        You can also assign colors to each line::

        > for ii, clr in enumerate('rgbym'):
        ...   plot.draw_lplot(np.arange(10), np.arange(10)*(ii+1), color=clr)
        > plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
        ...   title='draw_lplot [cset="rgbym"]')

        You can use one of the color sets as defined in ``tol_colors``::

        > plot.set_cset('mute')   # Note the default is 'bright'
        > for ii in range(5):
        ...   plot.draw_lplot(ydata=np.arange(10)*(ii+1))
        > plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
        ...   title='draw_lplot [cset="mute"]')

        Or you can use a color map as defined in ``tol_colors`` where
        you can define the number of colors you need. If you need less than 24
        colors, you can use 'rainbow_discrete' or you can choose one
        color map if you need more colors, for example::

        > plot.set_cset('rainbow_PuBr', 25)
        > for ii in range(25):
        ...   plot.draw_lplot(ydata=np.arange(10)*(ii+1))
        > plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
        ...   title='draw_lplot [cset="rainbow_PyBr"]')
        > plot.close()
        """
        if ydata is None:
            if self.__mpl["fig"] is None:
                raise ValueError("No plot defined and no data provided")
            fig = self.__mpl["fig"]
            axx = self.__mpl["axx"]

            if fig_info is None:
                fig_info = FIGinfo()

            if "text" in kwargs:
                axx.text(
                    0.05,
                    0.985,
                    f'{kwargs.pop("text")}',
                    transform=axx.transAxes,
                    fontsize="small",
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round",
                        "alpha": 0.5,
                        "facecolor": "#FFFFFF",
                        "edgecolor": "#BBBBBB",
                    },
                )

            close_draw_lplot(axx, self.__mpl["time_axis"], title, kwlegend, **kwargs)

            # add annotation and save the figure
            self.__add_copyright(axx)
            self.__add_fig_box(fig, fig_info)
            self.__close_this_page(fig)
            self.__mpl["fig"] = None
            return

        # initialize figure
        if xdata is None:
            xdata = np.arange(ydata.size)
        if self.__mpl["fig"] is None:
            if square:
                figsize = (9, 9)
            else:
                figsize = {0: (10, 7), 1: (10, 7), 2: (12, 7)}.get(
                    len(ydata) // 256, (14, 8)
                )

            fig, axx = plt.subplots(1, figsize=figsize)
            self.__mpl = {
                "fig": fig,
                "axx": axx,
                "time_axis": isinstance(xdata[0], datetime),
            }

            # add a centered subtitle to the figure
            self.__add_caption(self.__mpl["fig"])

            # set color cycle
            if self.cset is None:
                self.__mpl["axx"].set_prop_cycle(None)
            else:
                self.__mpl["axx"].set_prop_cycle(color=self.cset)

        # draw line in figure
        fig_draw_lplot(self.__mpl["axx"], xdata, ydata, **kwargs)

    # --------------------------------------------------
    def draw_multiplot(
        self: MONplot,
        data_tuple: tuple,
        gridspec: GridSpec | None = None,
        *,
        fig_info: FIGinfo | None = None,
        title: str | None = None,
        **kwargs: int,
    ) -> None:
        """Display multiple plots on one page.

        The data of each plot is defined in the parameter `data_tuple`.

        Parameters
        ----------
        data_tuple :  tuple of np.ndarray, xarray.DataArray or xarray.Dataset
           One dataset per subplot.
        gridspec :  matplotlib.gridspec.GridSpec, optional
           Instance of `matplotlib.gridspec.GridSpec`.
        fig_info  :  FIGinfo, optional
           Meta-data to be displayed in the figure.
        title :  str, optional
           Title of this figure using `Axis.set_title`.
           Ignored when data is a `xarray` data-structure.
        **kwargs :   other keywords
           The keywords are passed to `matplotlib.pyplot.plot`.
           Ignored when data is a `xarray` data-structure.

        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.

        Notes
        -----
        When data is a xarray.DataArray then the following attributes are used:

        - long_name: used as the title of the main panel when parameter 'title'\
          is not defined.
        - units: units of the data
        - _plot: dictionary with parameters for matplotlib.pyplot.plot
        - _title: title of the subplot (matplotlib: Axis.set_title)
        - _text: text shown in textbox placed in the upper left corner
        - _yscale: y-axis scale type, default 'linear'
        - _xlim: range of the x-axis
        - _ylim: range of the y-axis

        Examples
        --------
        Show two numpy arrays, each in a different panel. The subplots are
        above each other (row=2, col=1). The X-coordinates are generated
        using np.range(ndarray1) and  np.range(ndarray2)::

        > data_tuple = (ndarray1, ndarray2)
        > plot = MONplot(fig_name)
        > plot.draw_multiplot(data_tuple, title='my title',
        ...                   marker='o', linestyle='', color='r')
        > plot.close()

        Show four DataArrays, each in a different panel. The subplots
        are above each other in 2 columns (row=2, col=2). The X-coordinates
        are generated from the first dimension of the DataArrays::

        > data_tuple = (xarr1, xarr2, xarr3, xarr4)
        > plot = MONplot(fig_name)
        > plot.draw_multiplot(data_tuple, title='my title',
        ...                   marker='o', linestyle='')
        > plot.close()

        Show the DataArrays in a Dataset, each in a different panel. If there
        are 3 DataArrays preset then the subplots are above each other (row=3,
        col=1). The X-coordinates are generated from the (shared?) first
        dimension of the DataArrays::

        > plot = MONplot(fig_name)
        > plot.draw_multiplot(xds, title='my title')
        > plot.close()
        """
        # generate figure using contained layout
        fig = plt.figure(figsize=(10, 10))

        # define grid layout to place subplots within a figure
        if gridspec is None:
            geometry = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2)}.get(len(data_tuple))
            gridspec = GridSpec(*geometry, figure=fig)
        else:
            if len(data_tuple) > gridspec.nrows * gridspec.ncols:
                raise RuntimeError("grid too small for number of datasets")

        # determine xylabels
        xylabels = get_xylabels(gridspec, data_tuple)

        # add a centered subtitle to the figure
        self.__add_caption(fig)

        # add subplots, cycle the DataArrays of the Dataset
        axx = None
        data_iter = iter(data_tuple)
        for iy in range(gridspec.nrows):
            for ix in range(gridspec.ncols):
                axx = fig.add_subplot(gridspec[iy, ix])
                axx.grid(True)

                data = next(data_iter)
                if isinstance(data, np.ndarray):
                    if ix == iy == 0 and title is not None:
                        axx.set_title(title)
                    axx.plot(np.arange(data.size), data, **kwargs)
                elif isinstance(data, xr.DataArray):
                    draw_subplot(axx, data, xylabels[iy, ix, :])
                else:
                    for name in data.data_vars:
                        draw_subplot(axx, data[name], xylabels[iy, ix, :])

        # add annotation and save the figure
        self.__add_copyright(axx)
        if fig_info is None:
            fig_info = FIGinfo()
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_tracks(
        self: MONplot,
        lons: np.ndarray,
        lats: np.ndarray,
        icids: np.ndarray,
        *,
        saa_region: np.ndarray | None = None,
        fig_info: FIGinfo | None = None,
        title: str | None = None,
    ) -> None:
        """Display tracks of satellite on a world map.

        This module uses the Robinson projection.

        Parameters
        ----------
        lons :  (N, 2) array-like
           Longitude coordinates at start and end of measurement
        lats :  (N, 2) array-like
           Latitude coordinates at start and end of measurement
        icids :  (N) array-like
           ICID of measurements per (lon, lat)
        saa_region :  (N, 2) array-like, optional
           The coordinates of the vertices. When defined, then show SAA region
           as a matplotlib polygon patch
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure
        title :  str, optional
           Title of this figure using `Axis.set_title`

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        if not FOUND_CARTOPY:
            raise RuntimeError("You need Cartopy to use this method")

        if fig_info is None:
            fig_info = FIGinfo()

        # define plot layout
        # pylint: disable=abstract-class-instantiated
        myproj = {"projection": ccrs.Robinson(central_longitude=11.5)}
        fig, axx = plt.subplots(figsize=(12.85, 6), subplot_kw=myproj)

        # add a centered subtitle of the Figure
        self.__add_caption(fig)

        # add title to image panel
        if title is not None:
            axx.set_title(title)

        # draw tracks of satellite
        fig_draw_tracks(axx, lons, lats, icids, saa_region)

        # finalize figure
        self.__add_copyright(axx)
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_fov_ckd(
        self: MONplot,
        data: xr.DataArray | np.ndarray,
        *,
        vp_blocks: tuple,
        vp_labels: tuple[str] | None = None,
        fig_info: FIGinfo | None = None,
        title: str | None = None,
        **kwargs: int,
    ) -> None:
        """Display the SPEXone FOV CKD.

        The SPEXone FOV CKD consists of data from several viewports.

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes.
        vp_blocks : tuple
           Ranges of rows belonging to the data of one viewport. Each block
           is show in a separate subplot.
        vp_labels :  tuple of str
           Label for each viewport, default=('+50', '+20', '0', '-20', '-50').
        fig_info :  FIGinfo, default=None
           OrderedDict holding meta-data to be displayed in the figure.
        title :  str, default=None
           Title of this figure using `Axis.set_title`.
        **kwargs :   other keywords
           Keyword arguments: `zscale`, `vperc` or `vrange`

        Notes
        -----
        The current implementation only works for SPEXone CKD: FIELD_OF_VIEW,
        POLARIMETRIC, RADIOMETRIC and WAVELENGTH

        See also
        --------
        fig_data_to_xarr : Prepare image data for plotting.

        Examples
        --------
        Read SPEXone CKD::

        > from pyspex.ckd_io import CKDio
        > with CKDio(ckd_file) as ckd:
        >     fov_ckd = ckd.fov()
        >     rad_ckd = ckd.radiometric()

        Set row-ranges belonging to one viewport::

        > nview = fov_ckd.dims['viewports']
        > vp_blocks = ()
        > for ii in range(nview):
        >    ibgn = int(fov_ckd['fov_ifov_start_vp'][nview - ii - 1])
        >    iend = int(ibgn + fov_ckd['fov_nfov_vp'][nview - ii - 1] + 1)
        >    vp_blocks += ([ibgn, iend],)

        Create figures::

        > from moniplot.mon_plot import MONplot
        > plot = MONplot('test_spx1_fov_ckd.pdf', caption='SPEXone CKD')
        > plot.draw_fov_ckd(rad_ckd.isel(polarization_directions=0),
        >                   vp_blocks=vp_blocks,
        >                   title=rad_ckd.attrs['long_name'] + ' (S+)')
        > plot.draw_fov_ckd(rad_ckd.isel(polarization_directions=1),
        >                   vp_blocks=vp_blocks, zscale='log',
        >                   title=rad_ckd.attrs['long_name'] + ' (S-)')
        > plot.close()
        """
        if vp_labels is None:
            vp_labels = ("+50", "+20", "0", "-20", "-50")
        if fig_info is None:
            fig_info = FIGinfo()

        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and "_zscale" in data.attrs:
            xarr = data.copy()
        else:
            xarr = fig_data_to_xarr(data, **kwargs)

        # get dimensions needed to draw the data of the viewports
        nview = len(vp_labels)
        ncol = xarr.sizes["spectral_detector_pixels"]

        # define plot layout
        figsize = (
            1.75
            * (
                xarr.sizes["spectral_detector_pixels"]
                // xarr.sizes["spatial_samples_per_image"]
            ),
            4.5,
        )
        fig, axs = plt.subplots(nview, 1, figsize=figsize, sharex="all")
        fig.subplots_adjust(hspace=0, wspace=0, left=0.075, right=1.05, top=0.8)

        # add a centered subtitle of the Figure
        self.__add_caption(fig)

        # add title to image panel
        if title is not None:
            axs[0].set_title(title)
        elif "long_name" in xarr.attrs:
            axs[0].set_title(xarr.attrs["long_name"])

        ax_img = None
        cmap = self.cmap if self.cmap else xarr.attrs["_cmap"]
        for ii in range(nview):
            axs[ii].set_anchor((0.5, (nview - ii - 1) * 0.25))
            ibgn, iend = vp_blocks[ii]
            extent = (0, ncol, ibgn, iend)
            ax_img = axs[ii].imshow(
                xarr.values[ibgn:iend, :],
                norm=xarr.attrs["_znorm"],
                cmap=cmap,
                extent=extent,
                aspect=2,
                interpolation="none",
                origin="lower",
            )
            axs[ii].set_xticks([x * ncol // 8 for x in range(9)])
            axs[ii].set_yticks([ibgn, (iend + ibgn) // 2])
            yax2 = axs[ii].secondary_yaxis(-0.05)
            yax2.tick_params(left=False, labelleft=False)
            yax2.set_ylabel(vp_labels[ii])
            if ii == nview // 2:
                axs[ii].set_ylabel(xarr.dims[0])
            if ii == nview - 1:
                axs[ii].set_xlabel(xarr.dims[1])
            else:
                axs[ii].tick_params(labelbottom=False)

        # defaults: pad=0.05, aspect=20
        fig.colorbar(ax_img, ax=axs, pad=0.01, aspect=10, label=xarr.attrs["_zlabel"])

        # finalize figure
        self.__add_copyright(axs[-1])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)
