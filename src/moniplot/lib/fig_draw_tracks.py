"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

This module

Functions
---------
fig_draw_tracks(axx, lons, lats, icids, saa_region)
   Add a subplot for tracks of the satellite projected on the Earth surface.

Copyright (c) 2022 SRON - Netherlands Institute for Space Research

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

License:  GPLv3
"""
import numpy as np

from cartopy import crs as ccrs

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

from ..tol_colors import tol_cset


# - main functions ---------------------------------
def fig_draw_tracks(axx, lons, lats, icids, saa_region) -> None:
    """
    Add a subplot for tracks of the satellite projected on the Earth surface
    """
    # define colors
    cset = tol_cset('bright')

    # draw coastlines and gridlines
    axx.set_global()
    axx.coastlines(resolution='110m')
    axx.gridlines()

    # draw satellite position(s)
    icolor = 0
    for val in np.unique(icids):
        mask = icids == val
        # pylint: disable=abstract-class-instantiated
        plt.plot(lons[mask], lats[mask], linestyle='',
                 marker='s', markersize=2,
                 color=cset[icolor % 6], label=f'ICID: {val}',
                 transform=ccrs.PlateCarree())
        icolor += 1
    axx.legend(loc='lower left')

    # draw SAA region
    if saa_region is not None:
        # pylint: disable=abstract-class-instantiated
        saa_poly = Polygon(xy=saa_region, closed=True, alpha=1.0,
                           facecolor=cset.grey, transform=ccrs.PlateCarree())
        axx.add_patch(saa_poly)
