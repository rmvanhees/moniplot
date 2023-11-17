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

__all__ = ["DrawGeo"]

from typing import TYPE_CHECKING

import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from .lib.saa_region import saa_region
from .tol_colors import tol_cmap, tol_cset

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.figure import Figure

# - global variables -------------------------------
SPHERE_RADIUS = 6370997.0


# - helper functions -------------------------------
def set_proj_parms(lon_0: float = 0.0, lat_0: float = 0.0) -> dict:
    """Return settings for the ObliqueMercator projection."""
    return {
        "central_longitude": lon_0,
        "central_latitude": lat_0,
        "false_easting": 0.0,
        "false_northing": 0.0,
        "scale_factor": 1.0,
        "azimuth": 0.0,
        "globe": ccrs.Globe(ellipse="sphere"),
    }


# - class definition -------------------------------
class DrawGeo:
    """..."""

    def __init__(self: DrawGeo) -> None:
        """..."""
        self._cmap = tol_cmap("rainbow_PuRd")
        self._cset = {
            "water": "#DDEEFF",
            "land": "#E1C999",
            "grid": "#BBBBBB",
            "satellite": "#EE6677",
        }
        self._saa = None

    def set_saa(self: DrawGeo) -> None:
        """Define SAA region."""
        self._saa = saa_region()

    def __draw_worldmap(self: DrawGeo, axx: Axes, whole_globe: bool = True) -> None:
        """Draw worldmap."""
        if whole_globe:
            parallel_half = 0.883 * SPHERE_RADIUS
            meridian_half = 2.360 * SPHERE_RADIUS
            axx.set_xlim(-parallel_half, parallel_half)
            axx.set_ylim(-meridian_half, meridian_half)

        axx.spines["geo"].set_visible(False)
        axx.patch.set_facecolor(self._cset["water"])
        axx.add_feature(cfeature.LAND, edgecolor="none", facecolor=self._cset["land"])
        glx = axx.gridlines(linestyle="-", linewidth=0.5, color=self._cset["grid"])
        glx.xlocator = mpl.ticker.FixedLocator(np.linspace(-180, 180, 13))
        glx.ylocator = mpl.ticker.FixedLocator(np.linspace(-90, 90, 13))
        glx.xformatter = LONGITUDE_FORMATTER
        glx.yformatter = LATITUDE_FORMATTER

    # --------------------------------------------------
    def tracks(
        self: DrawGeo,
        lons: np.ndarray,
        lats: np.ndarray,
        icids: np.ndarray,
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
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
        title :  str, optional
           Title of this figure using `Axis.set_title`
        """
        myproj = {"projection": ccrs.Robinson(central_longitude=11.5)}
        fig, axx = plt.subplots(figsize=(12.85, 6), subplot_kw=myproj)

        # add title to image panel
        if title is not None:
            axx.set_title(title)

        # draw coastlines and gridlines
        axx.set_global()
        axx.coastlines(resolution="110m")
        axx.gridlines()

        # draw satellite position(s)
        for val in np.unique(icids):
            mask = icids == val
            # pylint: disable=abstract-class-instantiated
            plt.plot(
                lons[mask],
                lats[mask],
                linestyle="",
                marker="s",
                markersize=2,
                label=f"ICID: {val}",
                transform=ccrs.PlateCarree(),
            )
        axx.legend(loc="lower left")

        # show SAA region
        if self._saa is not None:
            # pylint: disable=abstract-class-instantiated
            saa_poly = Polygon(
                xy=self._saa,
                closed=True,
                alpha=1.0,
                facecolor=tol_cset("bright").grey,
                transform=ccrs.PlateCarree(),
            )
            axx.add_patch(saa_poly)
        return fig, axx

    # --------------------------------------------------
    def subsat(
        self: DrawGeo,
        lons: np.ndarray,
        lats: np.ndarray,
        *,
        whole_globe: bool = False,
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Display sub-satellite coordinates projected with TransverseMercator.

        Parameters
        ----------
        lons :  ndarray
           Longitude coordinates (N,)
        lats :  ndarray
           Latitude coordinates (N,)
        whole_globe :  bool, default=False
           Display the whole globe
        title :  str, optional
           Title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        """
        # determine central longitude
        if lons.max() - lons.min() > 180:
            if np.sum(lons > 0) > np.sum(lons < 0):
                lons[lons < 0] += 360
            else:
                lons[lons > 0] -= 360
        lon_0 = np.around(np.mean(lons), decimals=0)
        # print("lon_0: ", lon_0)

        # determine central latitude
        lat_0 = 0.0 if whole_globe else np.around(np.mean(lats), decimals=0)
        # print("lat_0: ", lat_0)

        # inititalize figure
        # pylint: disable=abstract-class-instantiated
        myproj = {"projection": ccrs.ObliqueMercator(**set_proj_parms(lon_0, lat_0))}
        fig, axx = plt.subplots(figsize=(12, 9), subplot_kw=myproj)

        # add title to image panel
        if title is not None:
            axx.set_title(title)

        # draw worldmap
        self.__draw_worldmap(axx, whole_globe=whole_globe)

        # draw sub-satellite spot(s)
        axx.scatter(
            lons,
            lats,
            4,
            transform=ccrs.PlateCarree(),
            marker="o",
            linewidth=0,
            color=self._cset["satellite"],
        )
        return fig, axx

    # --------------------------------------------------
    def mash(
        self: DrawGeo,
        lons: np.ndarray,
        lats: np.ndarray,
        data_in: np.ndarray | xr.DataArray,
        *,
        whole_globe: bool = False,
        vperc: list[float, float] | None = None,
        vrange: list[float, float] | None = None,
        title: str | None = None,
    ) -> tuple[Figure, type[Axes]]:
        """Display sub-satellite coordinates projected with TransverseMercator.

        Parameters
        ----------
        lons :  ndarray
           Longitude coordinates (N, M,)
        lats :  ndarray
           Latitude coordinates (N, M,)
        data_in :  ndarray | xr.DataArray
           Pixel values  (N, M,) or (N-1, M-1,)
        vrange :  list[vmin,vmax], optional
           Range to normalize luminance data between vmin and vmax.
        vperc :  list[vmin, vmax], optional
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.].
           keyword 'vperc' is ignored when vrange is given
        whole_globe :  bool, default=False
           Display the whole globe
        title :  str, optional
           Title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        """

        def extent_arr(aa: np.ndarray) -> np.ndarray:
            res = aa - ((aa[:, 1] - aa[:, 0]) / 2).reshape(-1, 1)
            return np.append(res, (2 * res[:, -1] - res[:, -2]).reshape(-1, 1), axis=1)

        if vrange is None and vperc is None:
            vperc = (1.0, 99.0)
        elif vrange is None:
            if len(vperc) != 2:
                raise TypeError("keyword vperc requires two values")
        else:
            if len(vrange) != 2:
                raise TypeError("keyword vrange requires two values")

        zlabel = "value"
        if isinstance(data_in, np.ndarray):
            data = data_in
        else:
            data = data_in.values
            if "units" in data_in.attrs and data_in.attrs["units"] != "1":
                zlabel = rf'value [{data_in.attrs["units"]}]'

        # determine central longitude
        if lons.max() - lons.min() > 180:
            if np.sum(lons > 0) > np.sum(lons < 0):
                lons[lons < 0] += 360
            else:
                lons[lons > 0] -= 360
        lon_0 = np.around(np.mean(lons), decimals=0)

        # determine central latitude
        lat_0 = 0.0 if whole_globe else np.around(np.mean(lats), decimals=0)

        if lons.shape == data.shape:
            lons = extent_arr(lons)
            lats = extent_arr(lats)

        # define data-range
        if vrange is None:
            vmin, vmax = np.nanpercentile(data, vperc)
        else:
            vmin, vmax = vrange

        # inititalize figure
        # pylint: disable=abstract-class-instantiated
        myproj = {"projection": ccrs.ObliqueMercator(**set_proj_parms(lon_0, lat_0))}
        fig, axx = plt.subplots(figsize=(12, 9), subplot_kw=myproj)

        # add title to image panel
        if title is not None:
            axx.set_title(title)

        # Add the colorbar axes anywhere in the figure.
        # Its position will be re-calculated at each figure resize.
        cax = fig.add_axes([0, 0, 0.1, 0.1])
        fig.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.85)

        # draw worldmap
        self.__draw_worldmap(axx, whole_globe=whole_globe)

        # draw sub-satellite spot(s)
        img = axx.pcolormesh(
            lons,
            lats,
            data,
            vmin=vmin,
            vmax=vmax,
            cmap=self._cmap,
            rasterized=False,
            transform=ccrs.PlateCarree(),
        )
        plt.draw()
        posn = axx.get_position()
        cax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.04, posn.height])

        # add color-bar
        plt.colorbar(img, cax=cax, label=zlabel)

        return fig, Axes
