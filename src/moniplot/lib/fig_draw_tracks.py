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
"""This module contains `fig_draw_tracks` used by `draw_tracks`."""

from __future__ import annotations

__all__ = ['fig_draw_tracks']

from typing import TYPE_CHECKING

import numpy as np

try:
    from cartopy import crs as ccrs
except ModuleNotFoundError:
    FOUND_CARTOPY = False
else:
    FOUND_CARTOPY = True
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from moniplot.tol_colors import tol_cset

if TYPE_CHECKING:
    from matplotlib import Axes


# - main functions ---------------------------------
if FOUND_CARTOPY:
    def fig_draw_tracks(axx: Axes,
                        lons: np.ndarray,
                        lats: np.ndarray,
                        icids: np.ndarray,
                        saa_region: list[tuple[float, float], ...]) -> None:
        """Draw satellite tracks projected on the Earth surface.

        Parameters
        ----------
        axx : matplotlib.Axes
           Matplotlib Axes object of plot window
        lons : array_like
           Longitude coordinates
        lats : array_like
           Latitude coordinates
        icids :  array_like
           ICID of each ground pixel
        saa_region :
           Polygon of the SAA region
        """
        # draw coastlines and gridlines
        axx.set_global()
        axx.coastlines(resolution='110m')
        axx.gridlines()

        # draw satellite position(s)
        for val in np.unique(icids):
            mask = icids == val
            # pylint: disable=abstract-class-instantiated
            plt.plot(lons[mask], lats[mask], linestyle='',
                     marker='s', markersize=2, label=f'ICID: {val}',
                     transform=ccrs.PlateCarree())
        axx.legend(loc='lower left')

        # draw SAA region
        if saa_region is not None:
            # pylint: disable=abstract-class-instantiated
            saa_poly = Polygon(xy=saa_region, closed=True, alpha=1.0,
                               facecolor=tol_cset('bright').grey,
                               transform=ccrs.PlateCarree())
            axx.add_patch(saa_poly)
