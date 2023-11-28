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
"""This module contains the class `MONplot`."""
from __future__ import annotations

__all__ = ["MONplot"]

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from .draw_image import DrawImage

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .lib.fig_info import FIGinfo

# - global variables -------------------------------


# - local functions --------------------------------


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
        self.__caption = "" if caption is None else caption
        self.__institute = ""
        self.__pdf = None
        self.filename = Path(figname)
        if self.filename.suffix.lower() != ".pdf":
            return

        self.__pdf = PdfPages(self.filename)

    # --------------------------------------------------
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

    def __add_caption(self: MONplot, fig: Figure, axx: Axes | None) -> None:
        """Add figure caption."""
        if not self.caption:
            return

        if axx is None:
            fig.suptitle(
                self.caption,
                fontsize="x-large",
                position=(0.5, 1 - 0.3 / fig.get_figheight()),
            )
        else:
            axx.set_title(self.caption, fontsize="x-large")

    @staticmethod
    def __add_fig_box(fig: Figure, fig_info: FIGinfo | None) -> None:
        """Add a box with meta information in the current figure.

        Parameters
        ----------
        fig :  matplotlib.figure.Figure
        fig_info :  FIGinfo, optional
        """
        if fig_info is None or fig_info.location != "above":
            return

        fig.text(
            1 - 0.4 / fig.get_figwidth(),
            1 - 0.25 / fig.get_figheight(),
            fig_info.as_str(),
            fontsize="x-small",
            style="normal",
            verticalalignment="top",
            horizontalalignment="right",
            multialignment="left",
            bbox={"facecolor": "white", "pad": 5},
        )

    # - Public Methods ---------------------------------
    def add_copyright(self: MONplot, axx: Axes) -> None:
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

    def close_this_page(
        self: MONplot,
        fig: Figure,
        fig_info: FIGinfo | None = None,
        *,
        axx: Axes | None = None,
    ) -> None:
        """Save the current figure and close the MONplot instance.

        Parameters
        ----------
        fig :  Figure
           Provide Figure instance, to place caption and close this figure
        fig_info :  FIGinfo, optional
           Provide text for the figure info-box
        axx: Axes, optional
           Provide axes and use set_title to place the caption (DrawImage)
        """
        self.__add_caption(fig, axx)
        self.__add_fig_box(fig, fig_info)

        # add save figure
        if self.__pdf is None:
            plt.savefig(self.filename)
        else:
            self.__pdf.savefig()
        plt.close()

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
        # plt.close("all")

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

        The information provided in the parameter `fig_info` will be displayed
        in a text box. In addition, we display the creation date and the data
        (biweight) median & spread.

        Currently, we have turned off the automatic offset notation of
        `matplotlib`. Maybe this should be the default, which the user may
        override.
        """
        if (
            title is None
            and isinstance(data, xr.DataArray)
            and "long_name" in data.attrs
        ):
            title = data.attrs["long_name"]

        plot = DrawImage(data.copy(), **kwargs)
        fig, axx = plot.subplots(side_panels=side_panels != "none")
        plot.draw(axx, fig_info=fig_info.copy(), side_panels=side_panels, title=title)
        self.add_copyright(axx["image"])
        self.close_this_page(fig, None)
