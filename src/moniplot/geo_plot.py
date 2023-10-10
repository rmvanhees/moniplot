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
#
"""This module contains the class `GEOplot`.

The methods of the class `GEOplot` are:
"""
from __future__ import annotations

__all__ = ['GEOplot']

from pathlib import Path
from typing import TYPE_CHECKING

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
except Exception as exc:
    raise RuntimeError('This module require module Cartopy') from exc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .image_to_xarray import h5_to_xr
from .lib.fig_info import FIGinfo
from .tol_colors import tol_cmap

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib import colormaps
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# - constants --------------------------------------
SPHERE_RADIUS = 6370997.0


# - helper function --------------------------------
def set_proj_parms(lon_0: float = 0.0, lat_0: float = 0.0) -> dict:
    """Return settings for the ObliqueMercator projection."""
    return {'central_longitude': lon_0,
            'central_latitude': lat_0,
            'false_easting': 0.0,
            'false_northing': 0.0,
            'scale_factor': 1.0,
            'azimuth': 0.0,
            'globe': ccrs.Globe(ellipse='sphere')}


# - main function ----------------------------------
class GEOplot:
    """Generate geolocation figure(s).

    Parameters
    ----------
    figname :  Path | str
        Name of PDF or PNG file (extension required)
    caption :  str, optional
        Caption repeated on each page of the PDF
    """

    def __init__(self: GEOplot, figname: Path | str,
                 caption: str | None = None) -> None:
        """Initialize multi-page PDF document or a single-page PNG."""
        self.__caption = '' if caption is None else caption
        self.__institute = ''
        # define colors
        self.__cset = {
            'water': '#ddeeff',
            'land': '#e1c999',
            'grid': '#bbbbbb',
            'satellite': '#ee6677'}
        self.__cmap = tol_cmap('rainbow_PuRd')
        self.__zunit = None

        self.filename = Path(figname)
        if self.filename.suffix.lower() == '.pdf':
            self.__pdf = PdfPages(self.filename)
        else:
            self.__pdf = None

    def __close_this_page(self: GEOplot, fig: Figure) -> None:
        """Close current matplotlib figure or page in a PDF document."""
        # add save figure
        if self.__pdf is None:
            plt.savefig(self.filename)
            plt.close(fig)
        else:
            self.__pdf.savefig()

    def close(self: GEOplot) -> None:
        """Close multipage PDF document."""
        if self.__pdf is None:
            return

        self.__pdf.close()
        plt.close('all')

    # --------------------------------------------------
    @property
    def caption(self: GEOplot) -> str:
        """Returns caption of figure."""
        return self.__caption

    def set_caption(self: GEOplot, caption: str) -> None:
        """Set caption of each page of the PDF.

        Parameters
        ----------
        caption :  str
           Default title of all pages at the top of the page.
        """
        self.__caption = caption

    def __add_caption(self: GEOplot, fig: Figure) -> None:
        """Add figure caption."""
        if not self.caption:
            return

        fig.suptitle(self.caption, fontsize='x-large',
                     position=(0.5, 1 - 0.3 / fig.get_figheight()))

    # --------------------------------------------------
    @property
    def institute(self: GEOplot) -> str:
        """Returns name of institute."""
        return self.__institute

    def set_institute(self: GEOplot, institute: str) -> None:
        """Use the name of your institute as a signature.

        Parameters
        ----------
        institute :  str
           Provide abbreviation of the name of your institute to be used in
           the copyright statement in the main panel of the figures.
        """
        self.__institute = institute

    # --------------------------------------------------
    @property
    def cmap(self: GEOplot) -> colormaps:
        """Returns current Matplotlib colormap."""
        return self.__cmap

    def set_cmap(self: GEOplot, cmap: colormaps) -> None:
        """
        Define alternative color-map to overrule the default.

        Parameter
        ---------
         cmap :  matplotlib color-map
        """
        self.__cmap = cmap

    # --------------------------------------------------
    @property
    def zunit(self: GEOplot) -> str:
        """Returns value of zunit."""
        return self.__zunit

    def set_zunit(self: GEOplot, units: str) -> None:
        """Provide units of data to be displayed."""
        self.__zunit = units

    # -------------------------
    def __add_copyright(self: GEOplot, axx: Axes) -> None:
        """Display copyright statement in the lower right corner."""
        if not self.institute:
            return

        axx.text(1, 0, rf' $\copyright$ {self.institute}',
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 rotation='vertical', fontsize='xx-small',
                 transform=axx.transAxes)

    @staticmethod
    def __add_fig_box(fig: Figure, fig_info: FIGinfo) -> None:
        """
        Add meta-information in the current figure.

        Parameters
        ----------
        fig :  Matplotlib figure instance
        fig_info :  FIGinfo
           instance of moniplot.lib.FIGinfo to be displayed
        """
        if fig_info is None or fig_info.location == 'none':
            return

        xpos = 1 - 0.4 / fig.get_figwidth()
        ypos = 1 - 0.25 / fig.get_figheight()

        fig.text(xpos, ypos, fig_info.as_str(),
                 fontsize='small', style='normal',
                 verticalalignment='top',
                 horizontalalignment='right',
                 multialignment='left',
                 bbox={'facecolor': 'white', 'pad': 5})

    def __draw_worldmap(self: GEOplot, axx: Axes,
                        whole_globe: bool = True) -> None:
        """Draw worldmap."""
        if whole_globe:
            parallel_half = 0.883 * SPHERE_RADIUS
            meridian_half = 2.360 * SPHERE_RADIUS
            axx.set_xlim(-parallel_half, parallel_half)
            axx.set_ylim(-meridian_half, meridian_half)

        axx.spines['geo'].set_visible(False)
        axx.patch.set_facecolor(self.__cset['water'])
        axx.add_feature(cfeature.LAND, edgecolor='none',
                        facecolor=self.__cset['land'])
        glx = axx.gridlines(linestyle='-', linewidth=0.5,
                            color=self.__cset['grid'])
        glx.xlocator = mpl.ticker.FixedLocator(np.linspace(-180, 180, 13))
        glx.ylocator = mpl.ticker.FixedLocator(np.linspace(-90, 90, 13))
        glx.xformatter = LONGITUDE_FORMATTER
        glx.yformatter = LATITUDE_FORMATTER

    # --------------------------------------------------
    def draw_geo_subsat(self: GEOplot, lons: np.ndarray, lats: np.ndarray, *,
                        whole_globe: bool = False,
                        title: str | None = None,
                        fig_info: FIGinfo | None = None) -> None:
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
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        if fig_info is None:
            fig_info = FIGinfo()

        # determine central longitude
        if lons.max() - lons.min() > 180:
            if np.sum(lons > 0) > np.sum(lons < 0):
                lons[lons < 0] += 360
            else:
                lons[lons > 0] -= 360
        lon_0 = np.around(np.mean(lons), decimals=0)
        print('lon_0: ', lon_0)

        # determine central latitude
        lat_0 = 0. if whole_globe else np.around(np.mean(lats), decimals=0)
        print('lat_0: ', lat_0)

        # inititalize figure
        # pylint: disable=abstract-class-instantiated
        myproj = {'projection': ccrs.ObliqueMercator(
            **set_proj_parms(lon_0, lat_0))}
        fig, axx = plt.subplots(figsize=(12, 9), subplot_kw=myproj)

        # add a centered subtitle to the figure
        self.__add_caption(fig)
        if title is not None:
            axx.set_title(title, fontsize='large')

        # draw worldmap
        self.__draw_worldmap(axx, whole_globe=whole_globe)

        # draw sub-satellite spot(s)
        axx.scatter(lons, lats, 4, transform=ccrs.PlateCarree(),
                    marker='o', linewidth=0, color=self.__cset['satellite'])

        self.__add_copyright(axx)
        fig_info.add('lon0, lat0', [lon_0, lat_0])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_geo_mesh(self: GEOplot, lons: np.ndarray, lats: np.ndarray,
                      data_in: np.ndarray | xr.DataArray, *,
                      whole_globe: bool = False,
                      vperc: list[float, float] | None = None,
                      vrange: list[float, float] | None = None,
                      title: str | None = None,
                      fig_info: FIGinfo | None = None) -> None:
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
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        def extent_arr(aa: np.ndarray) -> np.ndarray:
            res = aa - ((aa[:, 1] - aa[:, 0]) / 2).reshape(-1, 1)
            return np.append(res,
                             (2 * res[:, -1] - res[:, -2]).reshape(-1, 1),
                             axis=1)

        if fig_info is None:
            fig_info = FIGinfo()

        if vrange is None and vperc is None:
            vperc = (1., 99.)
        elif vrange is None:
            if len(vperc) != 2:
                raise TypeError('keyword vperc requires two values')
        else:
            if len(vrange) != 2:
                raise TypeError('keyword vrange requires two values')

        zlabel = 'value'
        if isinstance(data_in, np.ndarray):
            data = data_in
        else:
            data = data_in.values
            if 'units' in data_in.attrs and data_in.attrs['units'] != '1':
                zlabel = fr'value [{data_in.attrs["units"]}]'

        # determine central longitude
        if lons.max() - lons.min() > 180:
            if np.sum(lons > 0) > np.sum(lons < 0):
                lons[lons < 0] += 360
            else:
                lons[lons > 0] -= 360
        lon_0 = np.around(np.mean(lons), decimals=0)

        # determine central latitude
        lat_0 = 0. if whole_globe else np.around(np.mean(lats), decimals=0)

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
        myproj = {'projection': ccrs.ObliqueMercator(
            **set_proj_parms(lon_0, lat_0))}
        fig, axx = plt.subplots(1, 1, figsize=(12.5, 9), subplot_kw=myproj)
        # add a centered subtitle to the figure
        self.__add_caption(fig)

        # add title to image panel
        if title is not None:
            axx.set_title(title, fontsize='large')

        # Add the colorbar axes anywhere in the figure.
        # Its position will be re-calculated at each figure resize.
        cax = fig.add_axes([0, 0, 0.1, 0.1])
        fig.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.85)

        # draw worldmap
        self.__draw_worldmap(axx, whole_globe=whole_globe)

        # draw sub-satellite spot(s)
        img = axx.pcolormesh(lons, lats, data,
                             vmin=vmin, vmax=vmax,
                             cmap=self.__cmap, rasterized=False,
                             transform=ccrs.PlateCarree())
        plt.draw()
        posn = axx.get_position()
        cax.set_position([posn.x0 + posn.width + 0.01,
                          posn.y0, 0.04, posn.height])

        plt.colorbar(img, cax=cax, label=zlabel)

        self.__add_copyright(axx)
        fig_info.add('lon0, lat0', [lon_0, lat_0])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)


# --------------------------------------------------
def __test() -> None:
    """Test module for GEOplot."""
    import h5py

    data_dir = Path('/nfs/SPEXone/ocal/pace-sds/oci_l1c/5.9/2022/03/21')
    if not data_dir.is_dir():
        data_dir = Path('/data/richardh/SPEXone/pace-sds/oci_l1c/5.9/2022/03/21')

    plot = GEOplot('test_oci_l1c.pdf')
    for oci_fl in data_dir.glob('PACE_OCI.20220321T*.L1C.nc'):
        plot.set_caption(oci_fl.stem + ' [geolocation]')
        with h5py.File(oci_fl) as fid:
            lats = fid['/geolocation_data/latitude'][:]
            lons = fid['/geolocation_data/longitude'][:]
            dset = fid['/observation_data/I']
            xarr = h5_to_xr(dset)
            xarr.attrs['long_name'] = dset.attrs['long_name']
            xarr.attrs['units'] = '1'
            xarr.attrs['_FillValue'] = -999.

        xarr.values[xarr.values == -999] = np.nan
        xarr = xarr.max(dim='number_of_views', skipna=True, keep_attrs=True)
        xarr = xarr.mean(dim='intensity_bands_per_view',
                         skipna=True, keep_attrs=True)
        i_nadir = lons.shape[1] // 2
        plot.draw_geo_subsat(lons[:, i_nadir], lats[:, i_nadir],
                             title='sub-satellite positions')
        plot.draw_geo_subsat(lons[:, i_nadir], lats[:, i_nadir],
                             whole_globe=True, title='sub-satellite positions')
        plot.draw_geo_mesh(lons, lats, xarr,
                           title=xarr.attrs['long_name'].decode())
        break

    plot.close()


if __name__ == '__main__':
    __test()
