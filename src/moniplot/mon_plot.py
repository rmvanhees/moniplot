"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

The class MONplot contains generic plot functions.

Notes
-----
The methods of the class MONplot will accept numpy arrays as input and display
your data without knowledge on the data units and coordinates. In most cases,
this will be enough for a quick inspection of your data.
However, when you use xarray labeled arrays and datasets then the software
will use the name of the xarray arrays, coordinate names and data attributes,
such as 'long_name' and 'units'.

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
from datetime import datetime
from pathlib import PurePath

import os
import numpy as np
import xarray as xr

try:
    from cartopy import crs as ccrs
except ModuleNotFoundError:
    FOUND_CARTOPY = False
else:
    FOUND_CARTOPY = True
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from .biweight import biweight

from .lib.fig_info import FIGinfo
from .lib.fig_draw_image import (adjust_img_ticks,
                                 fig_data_to_xarr,
                                 fig_qdata_to_xarr,
                                 fig_draw_panels)
from .lib.fig_draw_trend import add_subplot, add_hk_subplot
from .lib.fig_draw_qhist import fig_draw_qhist
from .lib.fig_draw_lplot import fig_draw_lplot, close_draw_lplot
from .lib.fig_draw_tracks import fig_draw_tracks


# - local functions --------------------------------

# - main function ----------------------------------
class MONplot:
    """
    Generate figure(s) for SRON onground calibration anlysis or
    inflight instrument monitoring

    Attributes
    ----------
    figname : str
       Name of the PDF or PNG file

    Methods
    -------
    close()
       Close PNG or (multipage) PDF document.
    set_caption(caption)
       Set caption of each page of the PDF.
    caption
       Return figure caption.
    set_cmap(cmap)
       Use alternative color-map for MONplot::draw_image.
    unset_cmap()
       Unset user supplied color-map, and use default color-map.
    cmap
       Return matplotlib colormap.
    set_institute(institute)
       Use the name of your institute as a signature.
    institute
       Return name of your institute.
    draw_signal(data, zscale='linear', side_panels='nanmedian', fig_info=None,
                title=None, **kwargs)
       Display 2D array data as an image and averaged column/row signal plots.
    draw_quality(data, ref_data=None, side_panels='quality', fig_info=None,
                 title=None, **kwargs)
       Display pixel-quality 2D data as an image and column/row statistics.
    draw_trend(xds=None, hk_xds=None, vrange_last_orbits=-1, fig_info=None,
               title=None, **kwargs)
       Display trends of measurement data and/or housekeeping data.
    draw_hist(data, data_sel=None, vrange=None, fig_info=None,
              title=None, **kwargs)
        Display data as histograms.
    draw_qhist(xds, data_sel=None, density=True, fig_info=None, title=None)
       Display pixel-quality data as histograms.
    draw_tracks(lons, lats, icids, saa_region=None, fig_info=None, title=None)
       Display tracks of satellite on a world map using a Robinson projection.

    Notes
    -----
    ...
    """
    def __init__(self, figname, caption=None, pdf_title=None):
        """
        Initialize multi-page PDF document or a single-page PNG

        Parameters
        ----------
        figname :  str
           Name of PDF or PNG file (extension required)
        caption :  str
           Caption repeated on each page of the PDF
        pdf_title :  string
           Title of the PDF document (attribute of the PDF document)
           Default: 'Monitor report on Tropomi SWIR instrument'
        """
        self.__cmap = None
        self.__caption = '' if caption is None else caption
        self.__institute = ''
        self.__mpl = None
        self.__pdf = None
        self.filename = figname
        if PurePath(figname).suffix.lower() != '.pdf':
            return

        self.__pdf = PdfPages(figname)
        # add PDF annotations
        doc = self.__pdf.infodict()
        if pdf_title is None:
            doc['Title'] = 'Monitor report on Tropomi SWIR instrument'
        else:
            doc['Title'] = pdf_title
        if self.__institute == 'SRON':
            doc['Author'] = '(c) SRON Netherlands Institute for Space Research'
        elif self.__institute:
            doc['Author'] = f'(c) {self.__institute}'

    def __repr__(self) -> None:
        pass

    def __close_this_page(self, fig) -> None:
        """
        Close current matplotlib figure or page in a PDF document
        """
        # add save figure
        if self.__pdf is None:
            plt.savefig(self.filename)
            plt.close(fig)
        else:
            self.__pdf.savefig()

    def close(self) -> None:
        """
        Close PNG or (multipage) PDF document
        """
        if self.__pdf is None:
            return

        self.__pdf.close()
        plt.close('all')

    # --------------------------------------------------
    @property
    def caption(self) -> str:
        """
        Return figure caption
        """
        return self.__caption

    def set_caption(self, caption: str) -> None:
        """
        Set caption of each page of the PDF

        Parameter
        ---------
        institute :  str
           Provide abbreviation of the name of your institute to be used in
           the copyright statement in the main panel of the figures.
        """
        self.__caption = caption

    def __add_caption(self, fig):
        """
        Add figure caption
        """
        if not self.caption:
            return

        fig.suptitle(self.caption, fontsize='x-large',
                     position=(0.5, 1 - 0.3 / fig.get_figheight()))

    # --------------------------------------------------
    @property
    def cmap(self):
        """
        Return matplotlib colormap
        """
        return self.__cmap

    def set_cmap(self, cmap) -> None:
        """
        Use alternative color-map for MONplot::draw_image

        Parameter
        ---------
         cmap :  matplotlib color-map
        """
        self.__cmap = cmap

    def unset_cmap(self) -> None:
        """
        Unset user supplied color-map, and use default color-map
        """
        self.__cmap = None

    # --------------------------------------------------
    @property
    def institute(self) -> str:
        """
        Return name of your institute
        """
        return self.__institute

    def set_institute(self, institute: str) -> None:
        """
        Use the name of your institute as a signature

        Parameter
        ---------
        institute :  str
           Provide abbreviation of the name of your institute to be used in
           the copyright statement in the main panel of the figures.
        """
        self.__institute = institute

    # --------------------------------------------------
    def __add_copyright(self, axx) -> None:
        """
        Show value of institute as copyright in the lower right corner
        of the current figure.
        """
        if not self.institute:
            return

        axx.text(1, 0, rf' $\copyright$ {self.institute}',
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 rotation='vertical', fontsize='xx-small',
                 transform=axx.transAxes)

    @staticmethod
    def __add_fig_box(fig, fig_info) -> None:
        """
        Add a box with meta information in the current figure

        Parameters
        ----------
        fig :  Matplotlib figure instance
        fig_info :  FIGinfo
           instance of pys5p.lib.plotlib.FIGinfo to be displayed
        """
        if fig_info is None or fig_info.location != 'above':
            return

        xpos = 1 - 0.4 / fig.get_figwidth()
        ypos = 1 - 0.25 / fig.get_figheight()

        fig.text(xpos, ypos, fig_info.as_str(),
                 fontsize='x-small', style='normal',
                 verticalalignment='top',
                 horizontalalignment='right',
                 multialignment='left',
                 bbox={'facecolor': 'white', 'pad': 5})

    @staticmethod
    def __add_img_fig_box(axx_c, aspect: int, fig_info) -> None:
        """
        Add a box with meta information for draw_signal and draw_quality

        Parameters
        ----------
        axx_c :  Matplotlib Axes instance of the colorbar
        aspect :  int
        fig_info :  FIGinfo
           instance of moniplot.lib.fig_info to be displayed
        """
        if fig_info is None or fig_info.location != 'above':
            return

        if len(fig_info) <= 5:    # put text above colorbar
            if aspect in (3, 4):
                halign = 'right'
                fontsize = 'xx-small' if len(fig_info) == 5 else 'x-small'
            else:
                halign = 'center'
                fontsize = 'x-small'
                    
            axx_c.text(0 if aspect == 2 else 1,
                       1.025 + aspect * 0.005,
                       fig_info.as_str(), fontsize=fontsize,
                       transform=axx_c.transAxes,
                       multialignment='left',                       
                       verticalalignment='bottom',
                       horizontalalignment=halign,
                       bbox={'facecolor': 'white', 'pad': 4})
        else:                     # put text below colorbar
            fontsize = 'xx-small' if aspect in (3, 4) else 'x-small'
            axx_c.text(0.125 + (aspect-1) * 0.2,
                       -0.03 - (aspect-1) * 0.005,
                       fig_info.as_str(), fontsize=fontsize,
                       transform=axx_c.transAxes,
                       multialignment='left',
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox={'facecolor': 'white', 'pad': 4})


    # --------------------------------------------------
    def draw_signal(self, data, *, fig_info=None, side_panels='nanmedian',
                    title=None, **kwargs) -> None:
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes
        fig_info :  FIGinfo, default=None
           OrderedDict holding meta-data to be displayed in the figure
        side_panels :  str, default='nanmedian'
           Show image row and column statistics in two side panels.
           Use 'none' when you do not want the side panels.
           Other valid values are: 'median', 'nanmedian', 'mean', 'nanmean',
           'quality', 'std' and 'nanstd'.
        title :  str, default=None
           Title of this figure (matplotlib: Axis.set_title)
        **kwargs :   other keywords
           Pass keyword arguments: zscale, vperc or vrange
           to moniplot.lib.fig_draw_image.fig_data_to_xarr()

        The information provided in the parameter 'fig_info' will be displayed
        in a text box. In addition, we display the creation date and the data
        (biweight) median & spread.

        xarray attributes
        -----------------
        Attributes starting with an underscore are added by fig_data_to_xarr

        long_name :  used as the title of the main panel when parameter 'title'
            is not defined.
        _cmap :  contains the matplotlib colormap
        _zlabel :  contains the label of the color bar
        _znorm :  matplotlib class to normalize the data between zero and one.
        _zscale :  scaling of the data values: linear, log, diff, ratio, ...
        _zunits :  adjusted units of the data

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset 'img'
        (np.ndarray or xr.DataArray) with side-panels and title

        >>> plot = MONplot('test.pdf', caption='my caption')
        >>> plot.set_institute('SRON')
        >>> plot.draw_signal(img, title='my title')

        Add the same figure without side-panels

        >>> plot.draw_signal(img, side_panels='none', title='my title')

        Add a figure using a fixed data-range that the colormap covers

        >>> plot.draw_signal(img1, title='my title', vrange=[zmin, zmax])

        Add a figure where img2 = img - img_ref

        >>> plot.draw_signal(img2, title='my title', zscale='diff')

        Add a figure where img2 = img / img_ref

        >>> plot.draw_signal(img2, title='my title', zscale='ratio')

        Finalize the PDF file

        >>> plot.close()

        """
        # initialize keyword parameters
        if fig_info is None:
            fig_info = FIGinfo()

        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and '_zscale' in data.attrs:
            xarr = data.copy()
        else:
            xarr = fig_data_to_xarr(data, **kwargs)

        # aspect of image data
        aspect = min(4, max(1, int(round(xarr.shape[1] / xarr.shape[0]))))

        # create a new figure
        fig_sz = {1: (10, 8),
                  2: (12, 6.15),
                  3: (13, 5),
                  4: (15, 4.65)}.get(aspect)
        fig = plt.figure(figsize=fig_sz)
        if self.caption:
            fig.suptitle(self.caption)

        # use a grid layout to place subplots within a figure, where
        # - gspec[0, 1] is reserved for the image
        # - gspec[1, 1] is reserved for the x-panel
        # - gspec[0, 0] is reserved for the y-panel
        # - gspec[0, 2] is reserved for the colorbar
        # - gspec[1, 2] is used to pace the small fig_info box (max 4 lines)
        # - gspec(:, 3] extra space for FigInfo and colorbar labels
        if aspect == 1:
            gspec = fig.add_gridspec(2, 4, wspace=0.05, hspace=0.025,
                                     width_ratios=(.5, 4., .2, .8),
                                     height_ratios=(4, .5),
                                     bottom=.1, top=.875)
        else:
            gspec = fig.add_gridspec(2, 4, wspace=0.05 / aspect, hspace=0.025,
                                     width_ratios=(1., 4. * aspect, .3, .7),
                                     height_ratios=(4, 1),
                                     bottom=.1 if aspect == 2 else .125,
                                     top=.85 if aspect == 2 else 0.825)

        # add image panel and draw image
        axx = fig.add_subplot(gspec[0, 1])
        img = axx.imshow(xarr.values, norm=xarr.attrs['_znorm'],
                         cmap=self.cmap if self.cmap else xarr.attrs['_cmap'],
                         aspect='auto', interpolation='none', origin='lower')
        # axx.grid(True)

        # add title to image panel
        if title is not None:
            axx.set_title(title)
        elif 'long_name' in xarr.attrs:
            axx.set_title(xarr.attrs['long_name'])
        self.__add_copyright(axx)

        # add colorbar
        axx_c = fig.add_subplot(gspec[0, 2])
        _ = plt.colorbar(img, cax=axx_c, label=xarr.attrs['_zlabel'])

        # add side panels
        if side_panels == 'none':
            adjust_img_ticks(axx, xarr)
            axx.set_xlabel(xarr.dims[1])
            axx.set_ylabel(xarr.dims[0])
        else:
            for xtl in axx.get_xticklabels():
                xtl.set_visible(False)
            for ytl in axx.get_yticklabels():
                ytl.set_visible(False)
            axx_p = {'X': fig.add_subplot(gspec[1, 1], sharex=axx),
                     'Y': fig.add_subplot(gspec[0, 0], sharey=axx)}
            fig_draw_panels(axx_p, xarr, side_panels)
            axx_p['X'].set_xlabel(xarr.dims[1])
            axx_p['Y'].set_ylabel(xarr.dims[0])

        # add data statistics to fig_info
        median, spread = biweight(xarr.values, spread=True)
        if xarr.attrs['_zunits'] is None or xarr.attrs['_zunits'] == '1':
            fig_info.add('median', median, '{:.5g}')
            fig_info.add('spread', spread, '{:.5g}')
        else:
            fig_info.add('median', (median, xarr.attrs['_zunits']), '{:.5g} {}')
            fig_info.add('spread', (spread, xarr.attrs['_zunits']), '{:.5g} {}')

        # add annotation and save figure
        self.__add_img_fig_box(axx_c, aspect, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_quality(self, data, ref_data=None, *, side_panels='quality',
                     fig_info=None, title=None, **kwargs) -> None:
        """
        Display pixel-quality 2D array data as image and column/row statistics

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes
        ref_data :  numpy.ndarray, default=None
           Numpy array holding reference data, for example pixel quality
           reference map taken from the CKD. Shown are the changes with
           respect to the reference data.
        fig_info :  FIGinfo, default=None
           OrderedDict holding meta-data to be displayed in the figure
        side_panels :  str, default='quality'
           Show image row and column statistics in two side panels.
           Use 'none' when you do not want the side panels.
        title :  str, default=None
           Title of this figure (matplotlib: Axis.set_title)
        **kwargs :   other keywords
           Pass keyword arguments: 'data_sel', 'thres_worst', 'thres_bad'
           or 'qlabels' to moniplot.lib.fig_draw_image.fig_qdata_to_xarr()

        xarray attributes
        -----------------
        Attributes starting with an underscore are added by fig_qdata_to_xarr

        long_name :  used as the title of the main panel when parameter 'title'
            is not defined.
        flag_values :  values of the flags used to qualify the pixel quality
        flag_meanings :  description of the flag values
        thres_bad :  threshold between good and bad
        thres_worst :  threshold between bad and worst
        _cmap :  contains the matplotlib colormap
        _znorm :  matplotlib class to normalize the data between zero and one.

        Notes
        -----
        The quality ranking labels are ['unusable', 'worst', 'bad', 'good'],
        in case nor reference dataset is provided. Where:
        - 'unusable'  : pixels outside the illuminated region
        - 'worst'     : 0 <= value < thres_worst
        - 'bad'       : 0 <= value < thres_bad
        - 'good'      : thres_bad <= value <= 1
        Otherwise the labels for quality ranking indicate which pixels have
        changed w.r.t. reference. The labels are:
        - 'unusable'  : pixels outside the illuminated region
        - 'worst'     : from good or bad to worst
        - 'bad'       : from good to bad
        - 'good'      : from any rank to good
        - 'unchanged' : no change in rank

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. Where creation date and statistics on the number of
        bad and worst pixels are displayed.

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset 'img'
        (np.ndarray or xr.DataArray) with side-panels and title

        >>> plot = MONplot('test.pdf', caption='my caption', institute='SRON')
        >>> plot.draw_quality(img, title='my title')

        Add the same figure without side-panels

        >>> plot.draw_quality(img, side_panels='none', title='my title')

        Add a figure where img_ref is a quality map from early in the mission

        >>> plot.draw_quality(img, img_ref, title='my title')

        Finalize the PDF file

        >>> plot.close()
        """
        if fig_info is None:
            fig_info = FIGinfo()

        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and '_zscale' in data.attrs:
            xarr = data
        else:
            xarr = fig_qdata_to_xarr(data, ref_data, **kwargs)

        # aspect of image data
        aspect = min(4, max(1, int(round(xarr.shape[1] / xarr.shape[0]))))

        # create a new figure
        fig_sz = {1: (10, 8),
                  2: (12, 6.15),
                  3: (13, 5),
                  4: (15, 4.65)}.get(aspect)
        fig = plt.figure(figsize=fig_sz)
        if self.caption:
            fig.suptitle(self.caption)

        # use a grid layout to place subplots within a figure, where
        # - gspec[0, 1] is reserved for the image
        # - gspec[1, 1] is reserved for the x-panel
        # - gspec[0, 0] is reserved for the y-panel
        # - gspec[0, 2] is reserved for the colorbar
        # - gspec[1, 2] is used to pace the small fig_info box (max 4 lines)
        if aspect == 1:
            gspec = fig.add_gridspec(2, 4, wspace=0.05, hspace=0.025,
                                     width_ratios=(.5, 4., .2, .8),
                                     height_ratios=(4, .5), bottom=.1, top=.875)
        else:
            gspec = fig.add_gridspec(2, 4, wspace=0.05 / aspect, hspace=0.025,
                                     width_ratios=(1., 4. * aspect, .3, .7),
                                     height_ratios=(4, 1),
                                     bottom=.1 if aspect == 2 else .125,
                                     top=.85 if aspect == 2 else 0.825)

        # add image panel and draw image
        axx = fig.add_subplot(gspec[0, 1])
        img = axx.imshow(xarr.values,
                         cmap=xarr.attrs['_cmap'], norm=xarr.attrs['_znorm'],
                         aspect='auto', interpolation='none', origin='lower')

        # add title to image panel
        if title is not None:
            axx.set_title(title)
        elif 'long_name' in xarr.attrs:
            axx.set_title(xarr.attrs['long_name'])
        self.__add_copyright(axx)

        # add colorbar
        axx_c = fig.add_subplot(gspec[0, 2])
        bounds = xarr.attrs['flag_values']
        mbounds = [(bounds[ii+1] + bounds[ii]) / 2
                   for ii in range(len(bounds)-1)]
        _ = plt.colorbar(img, cax=axx_c, ticks=mbounds, boundaries=bounds)
        axx_c.tick_params(axis='y', which='both', length=0)
        axx_c.set_yticklabels(xarr.attrs['flag_meanings'])

        # add side panels
        if side_panels == 'none':
            adjust_img_ticks(axx, xarr)
            axx.set_xlabel(xarr.dims[1])
            axx.set_ylabel(xarr.dims[0])
        else:
            for xtl in axx.get_xticklabels():
                xtl.set_visible(False)
            for ytl in axx.get_yticklabels():
                ytl.set_visible(False)
            axx_p = {'X': fig.add_subplot(gspec[1, 1], sharex=axx),
                     'Y': fig.add_subplot(gspec[0, 0], sharey=axx)}
            fig_draw_panels(axx_p, xarr, side_panels)
            axx_p['X'].set_xlabel(xarr.dims[1])
            axx_p['Y'].set_ylabel(xarr.dims[0])

        # add data statistics to fig_info
        if ref_data is None:
            fig_info.add(
                f'{xarr.attrs["flag_meanings"][2]}'
                f' (quality < {xarr.attrs["thres_bad"]})',
                np.sum((xarr.values == 1) | (xarr.values == 2)))
            fig_info.add(
                f'{xarr.attrs["flag_meanings"][1]}'
                f' (quality < {xarr.attrs["thres_worst"]})',
                np.sum(xarr.values == 1))
        else:
            fig_info.add(xarr.attrs['flag_meanings'][3],
                         np.sum(xarr.values == 4))
            fig_info.add(xarr.attrs['flag_meanings'][2],
                         np.sum(xarr.values == 2))
            fig_info.add(xarr.attrs['flag_meanings'][1],
                         np.sum(xarr.values == 1))

        # add annotation and save the figure
        self.__add_img_fig_box(axx_c, aspect, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_trend(self, xds=None, hk_xds=None, *,
                   fig_info=None, title=None, **kwargs) -> None:
        """
        Display trends of measurement data and/or housekeeping data

        Parameters
        ----------
        xds :  xarray.Dataset, optional
           Object holding measurement data and attributes
        hk_xds :  xarray.Dataset, optional
           Object holding housekeeping data and attributes
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure
        title :  str, optional
           Title of this figure (matplotlib: Axis.set_title)
        **kwargs :   other keywords
           Pass keyword arguments: vperc or vrange_last_orbits
           to moniplot.lib.fig_draw_trend.add_hk_subplot()

        xarray attributes
        -----------------
        long_name :  used as the title of the main panel when parameter 'title'
            is not defined.
        units :  units of the data

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset 'xds'
        (np.ndarray or xr.DataArray) with a title. The dataset 'xds' may
        contain multiple DataArrays with a common X-coordinate. Each DataArray
        will be displayed in a seperate sub-panel.

        >>> plot = MONplot('test.pdf', caption='my caption', institute='SRON')
        >>> plot.draw_trend(xds, hk_xds=None, title='my title')

        Add a figure with the same Dataset 'xds' and a few trends of
        housekeeping data (again each parameter in a seperate DataArray with
        with a common X-coordinate).

        >>> plot.draw_trend(xds, hk_xds, title='my title')

        Finalize the PDF file

        >>> plot.close()

        """
        if xds is None and hk_xds is None:
            raise ValueError('both xds and hk_xds are None')
        if xds is not None and not isinstance(xds, xr.Dataset):
            raise ValueError('xds should be and xarray Dataset object')
        if hk_xds is not None and not isinstance(hk_xds, xr.Dataset):
            raise ValueError('hk_xds should be and xarray Dataset object')

        if fig_info is None:
            fig_info = FIGinfo()

        # determine npanels from xarray Dataset
        npanels = len(xds.data_vars) if xds is not None else 0
        npanels += len(hk_xds.data_vars) if hk_xds is not None else 0

        # initialize matplotlib using 'subplots'
        figsize = (10., 1 + (npanels + 1) * 1.5)
        fig, axarr = plt.subplots(npanels, sharex=True, figsize=figsize)
        if npanels == 1:
            axarr = [axarr]
        margin = min(1. / (1.8 * (npanels + 1)), .25)
        fig.subplots_adjust(bottom=margin, top=1-margin, hspace=0.02)

        # add a centered suptitle to the figure
        self.__add_caption(fig)

        # add title to image panel
        if title is not None:
            axarr[0].set_title(title)

        # add figures with trend data
        ipanel = 0
        if xds is not None:
            xlabel = 'orbit' if 'orbit' in xds.coords else 'time [hours]'
            for name in xds.data_vars:
                add_subplot(axarr[ipanel], xds[name])
                ipanel += 1

        if hk_xds is not None:
            xlabel = 'orbit' if 'orbit' in hk_xds.coords else 'time [hours]'
            for name in hk_xds.data_vars:
                add_hk_subplot(axarr[ipanel], hk_xds[name], **kwargs)
                ipanel += 1

        # finally add a label for the X-coordinate
        axarr[-1].set_xlabel(xlabel)

        # add annotation and save the figure
        self.__add_copyright(axarr[-1])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_hist(self, data, data_sel=None, vrange=None,
                  fig_info=None, title=None, **kwargs) -> None:
        """
        Display data as histograms.

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes
        data_sel :  mask or index tuples for arrays, optional
           Select a region on the detector by fancy indexing (using a
           boolean/interger arrays), or using index tuples for arrays
           (generated with numpy.s_).
        vrange :  list, default=[data.min(), data.max()]
           The lower and upper range of the bins.
           Note data will also be clipped according to this range.
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure
        title :  str, optional
           Title of this figure (matplotlib: Axis.set_title)
        **kwargs :   other keywords
           Pass keyword arguments matplotlib.pyplot.hist: a.o. bins, density
           Note the keywords histtype, color, linewidth and fill are predefined.

        xarray attributes
        -----------------
        long_name :  used as the title of the main panel when parameter 'title'
            is not defined.
        units :  units of the data

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure with a histogram
        of array 'data' (np.ndarray or xr.DataArray). And a second page
        with also a histogram where data is a xarray with attribute
        'long_name' then the title is
        f'Histograms of {xarr.attrs["long_name"]} values'

        The array will be flattend the the histogram is created with
        matplotlib.pyplot.hist.

        >>> plot = MONplot('test.pdf', caption='my caption')
        >>> plot.set_institute('SRON')
        >>> plot.draw_hist(data, title='my title')
        >>> plot.draw_hist(xarr)
        >>> plot.close()
        """
        long_name = ''
        zunits = '1'
        if isinstance(data, xr.DataArray):
            if data_sel is None:
                values = data.values.reshape(-1)
            else:
                values = data.values[data_sel].reshape(-1)
            if 'long_name' in data.attrs:
                long_name = data.attrs['long_name']
            if 'units' in data.attrs:
                zunits = data.attrs['units']
        else:
            if data_sel is None:
                values = data.reshape(-1)
            else:
                values = data[data_sel].reshape(-1)

        # add data statistics to fig_info
        if fig_info is None:
            fig_info = FIGinfo()

        median, spread = biweight(values, spread=True)
        if zunits == '1':
            fig_info.add('median', median, '{:.5g}')
            fig_info.add('spread', spread, '{:.5g}')
        else:
            fig_info.add('median', (median, zunits), '{:.5g} {}')
            fig_info.add('spread', (spread, zunits), '{:.5g} {}')

        # create figure
        fig, axx = plt.subplots(1, figsize=(8, 7))

        # add a centered suptitle to the figure
        self.__add_caption(fig)

        # add title to image panel
        if title is None:
            title = f'Histograms of {long_name} values'
        axx.set_title(title)

        # add histogram
        if vrange is not None:
            values = np.clip(values, vrange[0], vrange[1])
        # Edgecolor is tol_cset('bright').blue
        if 'bins' in kwargs and kwargs['bins'] > 24:
            axx.hist(values, range=vrange, histtype='step',
                     edgecolor='#4477AA', facecolor='#77AADD',
                     fill=True, linewidth=1.5, **kwargs)
            axx.grid(which='major', color='#AAAAAA', ls='--')
        else:
            axx.hist(values, range=vrange, histtype='bar',
                     edgecolor='#4477AA', facecolor='#77AADD',
                     linewidth=1.5, **kwargs)
            axx.grid(which='major', axis='y', color='#AAAAAA', ls='--')
        axx.set_xlabel(long_name if long_name else 'value')
        if 'density' in kwargs and kwargs['density']:
            axx.set_ylabel('density')
        else:
            axx.set_ylabel('count')

        # add annotation and save the figure
        self.__add_copyright(axx)
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_qhist(self, xds, data_sel=None, density=True,
                   fig_info=None, title=None) -> None:
        """
        Display pixel-quality data as histograms.

        Parameters
        ----------
        xds :  xarray.Dataset
           Object holding measurement data and attributes
        data_sel :  mask or index tuples for arrays, optional
           Select a region on the detector by fancy indexing (using a
           boolean/interger arrays), or using index tuples for arrays
           (generated with numpy.s_).
        density : bool, default=True
           If True, draw and return a probability density: each bin will
           display the bin's raw count divided by the total number of counts
           and the bin width (see matplotlib.pyplot.hist).
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure
        title :  str, optional
           Title of this figure (matplotlib: Axis.set_title)

        xarray attributes
        -----------------
        long_name :  used as the title of the main panel when parameter 'title'
            is not defined.
        units :  units of the data

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset 'xds'
        (np.ndarray or xr.DataArray) with a title. The dataset 'xds' may
        contain multiple DataArrays with a common X-coordinate. Each DataArray
        will be displayed in a seperate sub-panel.

        >>> plot = MONplot('test.pdf', caption='my caption', institute='SRON')
        >>> plot.draw_qhist(xds, title='my title')
        >>> plot.close()

        """
        if not isinstance(xds, xr.Dataset):
            raise ValueError('xds should be and xarray Dataset object')

        if fig_info is None:
            fig_info = FIGinfo()

        # determine npanels from xarray Dataset
        npanels = len(xds.data_vars)

        # initialize matplotlib using 'subplots'
        figsize = (10., 1 + (npanels + 1) * 1.65)
        fig, axarr = plt.subplots(npanels, sharex=True, figsize=figsize)
        if npanels == 1:
            axarr = [axarr]
        margin = min(1. / (1.8 * (npanels + 1)), .25)
        fig.subplots_adjust(bottom=margin, top=1-margin, hspace=0.02)

        # add a centered suptitle to the figure
        self.__add_caption(fig)

        # add title to image panel
        if title is None:
            title = 'Histograms of pixel-quality'
        axarr[0].set_title(title)

        # add figures with histograms
        for ii, (key, xda) in enumerate(xds.data_vars.items()):
            if data_sel is None:
                qdata = xda.values.reshape(-1)
            else:
                qdata = xda.values[data_sel].reshape(-1)
            label = xda.attrs['long_name'] if 'long_name' in xda.attrs else key
            fig_draw_qhist(axarr[ii], qdata, label, density)

        # finally add a label for the X-coordinate
        axarr[-1].set_xlabel('pixel quality')

        # add annotation and save the figure
        self.__add_copyright(axarr[-1])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_lplot(self, xdata, ydata, color=0, *, square=False,
                   fig_info=None, title=None, **kwargs) -> None:
        """
        Plot y versus x lines, maybe called multiple times to add lines.
        Figure is closed when called with x is None.

        Parameters
        ----------
        xdata :  ndarray
           [add line] X data, [close figure] when xdata is None
        ydata :  ndarray
           [add line] Y data
        square :  bool
           [first call, only] create a square figure, independent of
           number of data-points.
        color :  integer, default=0
           [add line] Index to color in tol_colors.tol_cset('bright')
        fig_info  :  FIGinfo, optional
           [close figure] Meta-data to be displayed in the figure
        title :  str, default=None
           [close figure] Title of this figure (matplotlib: Axis.set_title)
        **kwargs :   other keywords
           [add line] Keywords are passed to mpl.pyplot.plot()
           [close figure] Kewords are passed to appropriate mpl.Axes method
           [close figure] keyword 'text' can be used to add addition text in
               the upper left corner.

        Examples
        --------
        General example:
        >>> plot = MONplot(fig_name)
        >>> for ii, xx, yy in enumerate(data_of_each_line):
        >>>    plot.draw_lplot(xx, yy, color=ii, label=mylabel[ii],
        >>>                    marker='o', linestyle='None')
        >>> plot.draw_lplot(None, None, xlim=[0, 0.5], ylim=[-10, 10],
        >>>                 xlabel=my_xlabel, ylabel=my_ylabel)
        >>> plot.close()

        Using a time-axis:
        >>> from datetime import datetime, timedelta
        >>> tt0 = (datetime(year=2020, month=10, day=1)
        >>>        + timedelta(seconds=sec_in_day))
        >>> tt = [tt0 + xx * t_step for xx in range(yy.size)]
        >>> plot = MONplot(fig_name)
        >>> plot.draw_lplot(tt, yy, color=1, label=mylabel,
        >>>                 marker='o', linestyle='None')
        >>> plot.draw_line(None, None, ylim=[-10, 10],
        >>>                xlabel=my_xlabel, ylabel=my_ylabel)
        >>> plot.close()
        """
        if xdata is None:
            if self.__mpl is None:
                raise ValueError('No plot defined and no data provided')
            fig = self.__mpl['fig']
            axx = self.__mpl['axx']

            if fig_info is None:
                fig_info = FIGinfo()

            if 'text' in kwargs:
                axx.text(0.05, 0.985, kwargs['text'],
                         transform=axx.transAxes,
                         fontsize='small', verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='#FFFFFF',
                                   edgecolor='#BBBBBB', alpha=0.5))

            close_draw_lplot(axx, self.__mpl['time_axis'], title, **kwargs)

            # add annotation and save the figure
            self.__add_copyright(axx)
            self.__add_fig_box(fig, fig_info)
            self.__close_this_page(fig)
            self.__mpl = None
            return

        # initialize figure
        if self.__mpl is None:
            if square:
                figsize = (9, 9)
            else:
                figsize = {0: (8, 7),
                           1: (10, 7),
                           2: (12, 7)}.get(len(xdata) // 256, (14, 7))

            self.__mpl = dict(zip(('fig', 'axx'),
                                  plt.subplots(1, figsize=figsize)))
            self.__mpl['time_axis'] = isinstance(xdata[0], datetime)

            # add a centered suptitle to the figure
            self.__add_caption(self.__mpl['fig'])

        # draw line in figure
        fig_draw_lplot(self.__mpl['axx'], xdata, ydata, color, **kwargs)

    # --------------------------------------------------
    def draw_multiplot(self, data_tuple: tuple, gridspec=None, *,
                       fig_info=None, title=None, **kwargs) -> None:
        """
        Display multiple subplots on one page using
        matplotlib.gridspec.GridSpec

        Parameters
        ----------
        data_tuple :  tuple with nparray, xarray.DataArray or xarray.Dataset
           One dataset per subplot
        gridspec :  matplotlib.gridspec.GridSpec, optional
           Instance of matplotlib.gridspec.GridSpec
        fig_info  :  FIGinfo, optional
           Meta-data to be displayed in the figure
        title :  str, default=None
           Title of this figure (matplotlib: Axis.set_title)
           Ignored when data is a xarray data structure
        **kwargs :   other keywords
           Keywords are passed to mpl.pyplot.plot().
           Ignored when data is a xarray data structure

        xarray attributes
        -----------------
        long_name :  used as the title of the main panel when parameter 'title'
            is not defined.
        units :  units of the data
        _plot :  dictionary with parameters for matplotlib.pyplot.plot
        _title :  title of the subplot (matplotlib: Axis.set_title)
        _text :  text shown in textbox placed in the upper left corner
        _yscale :  y-axis scale type, default 'linear'
        _xlim :  range of the x-axis
        _ylim :  range of the y-axis

        Notes
        -----

        Examples
        --------
        Show two numpy arrays:
        >>> data_tuple = (ndarray1, ndarray2)
        >>> plot = MONplot(fig_name)
        >>> plot.draw_multiplot(data_tuple: tuple, title='my title',
        >>>                     marker='o', linestyle='', color='r')
        >>> plot.close()
        """
        def draw_subplot(axx, xarr):
            if '_plot' in xarr.attrs:
                kwargs = xarr.attrs['_plot']
            else:
                kwargs = {'color': '#4477AA'}

            label = xarr.attrs['long_name'] \
                if 'long_name' in xarr.attrs else None
            xlabel = xarr.dims[0]
            ylabel = 'value'
            if 'units' in xarr.attrs and xarr.attrs['units'] != '1':
                ylabel += f' [{xarr.attrs["units"]}]'
            axx.plot(xarr.coords[xlabel], xarr.values, label=label, **kwargs)
            if label is not None:
                _ = axx.legend(fontsize='small', loc='upper right')
            axx.set_xlabel(xlabel)
            axx.set_ylabel(ylabel)

            if '_title' in xarr.attrs:
                axx.set_title(xarr.attrs['_title'])
            if '_yscale' in xarr.attrs:
                axx.set_yscale(xarr.attrs['_yscale'])
            if '_xlim' in xarr.attrs:
                axx.set_xlim(xarr.attrs['_xlim'])
            if '_ylim' in xarr.attrs:
                axx.set_ylim(xarr.attrs['_ylim'])
            if '_text' in xarr.attrs:
                axx.text(0.05, 0.985, kwargs['text'],
                         transform=axx.transAxes,
                         fontsize='small', verticalalignment='top',
                         bbox=dict(boxstyle='round',
                                   facecolor='#FFFFFF',
                                   edgecolor='#BBBBBB',
                                   alpha=0.5))

        # generate figure using contrained layout
        fig = plt.figure(figsize=(10, 10))

        # define grid layout to place subplots within a figure
        if gridspec is None:
            geometry = {1: (1, 1),
                        2: (2, 1),
                        3: (3, 1),
                        4: (2, 2)}.get(len(data_tuple))
            gridspec = GridSpec(*geometry, figure=fig)
        else:
            if len(data_tuple) > gridspec.nrows * gridspec.ncols:
                raise RuntimeError('grid too small for number of datasets')

        # add a centered suptitle to the figure
        self.__add_caption(fig)

        # add subplots, cycle the DataArrays of the Dataset
        data_iter = iter(data_tuple)
        for yy in range(gridspec.nrows):
            for xx in range(gridspec.ncols):
                axx = fig.add_subplot(gridspec[yy, xx])
                axx.grid(True)

                data = next(data_iter)
                if isinstance(data, np.ndarray):
                    if xx == yy == 0 and title is not None:
                        axx.set_title(title)
                    axx.plot(np.arange(data.size), data, **kwargs)
                elif isinstance(data, xr.DataArray):
                    draw_subplot(axx, data)
                else:
                    for name in data.data_vars:
                        draw_subplot(axx, data[name])

        # add annotation and save the figure
        self.__add_copyright(axx)
        if fig_info is None:
            fig_info = FIGinfo()
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_tracks(self, lons, lats, icids, *, saa_region=None,
                    fig_info=None, title=None) -> None:
        """
        Display tracks of satellite on a world map using a Robinson projection

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
           as a matplotlib polygon patch.
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        if not FOUND_CARTOPY:
            raise RuntimeError('You need Cartopy to run this function')

        if fig_info is None:
            fig_info = FIGinfo()

        # define plot layout
        # pylint: disable=abstract-class-instantiated
        myproj = {'projection': ccrs.Robinson(central_longitude=11.5)}
        fig, axx = plt.subplots(figsize=(12.85, 6), subplot_kw=myproj)

        # add a centered suptitle of the Figure
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
