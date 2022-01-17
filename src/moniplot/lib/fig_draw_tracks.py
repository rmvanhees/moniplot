"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

This module

Functions
---------
fig_draw_tracks(axx, lons, lats, icids, saa_region)
   Add a subplot for tracks of the satellite projected on the Earth surface.

Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
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

    # draw SAA region
    if saa_region is not None:
        # pylint: disable=abstract-class-instantiated
        saa_poly = Polygon(xy=saa_region, closed=True, alpha=1.0,
                           facecolor=cset.grey, transform=ccrs.PlateCarree())
        axx.add_patch(saa_poly)

    # draw satellite position(s)
    icid_found = []
    for lon, lat, icid in zip(lons, lats, icids):
        if icid not in icid_found:
            indx_color = len(icid_found)
        else:
            indx_color = icid_found.index(icid)
        # pylint: disable=abstract-class-instantiated
        line, = plt.plot(lon, lat, linestyle='-', linewidth=3,
                         color=cset[indx_color % 6],
                         transform=ccrs.PlateCarree())
        if icid not in icid_found:
            line.set_label(f'ICID: {icid}')
            icid_found.append(icid)
    axx.legend(loc='lower left')
