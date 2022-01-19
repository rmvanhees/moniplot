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
   All Rights Reserved

License:  GNU GPL v3.0
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
def fig_size(aspect: int, side_panels: str):
    """
    Calculate size of the figure (inches)
    """
    sz_corr = {4: (2.7, 0.865 if side_panels == 'none' else 1.025),
               3: (2.3, 0.9 if side_panels == 'none' else 1.05),
               2: (2.0, 1.05 if side_panels == 'none' else 1.3),
               1: (1.75, 1.5 if side_panels == 'none' else 1.75)}.get(aspect)
    if 'FIG_SZ_X' in os.environ:
        sz_corr = (float(os.environ.get('FIG_SZ_X')), sz_corr[1])
    if 'FIG_SZ_Y' in os.environ:
        sz_corr = (sz_corr[0], float(os.environ.get('FIG_SZ_Y')))
    return (6.4 * sz_corr[0], 4.8 * sz_corr[1])


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
    draw_signal(data, zscale='linear', fig_info=None, side_panels='nanmedian',
                title=None, **kwargs)
       Display 2D array data as an image and averaged column/row signal plots.
    draw_quality(data, ref_data=None, fig_info=None, side_panels='quality',
                 title=None, **kwargs)
       Display pixel-quality 2D data as an image and column/row statistics.
    draw_trend(xds=None, hk_xds=None, vrange_last_orbits=-1, fig_info=None,
               title=None, **kwargs)
       Display trends of measurement data and/or housekeeping data.
    draw_qhist(xds, density=True, fig_info=None, title=None)
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
           instance of moniplot.lib.fig_info to be displayed
        """
        if fig_info is None or fig_info.location == 'none':
            return

        if fig_info.location == 'above':
            xpos = 1 - 0.4 / fig.get_figwidth()
            ypos = 1 - 0.25 / fig.get_figheight()

            fig.text(xpos, ypos, fig_info.as_str(),
                     fontsize='x-small', style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor': 'white', 'pad': 5})

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

        # draw figure
        fig = plt.figure(figsize=fig_size(aspect, side_panels),
                         constrained_layout=True)
        self.__add_caption(fig)

        if side_panels == 'none':
            axx = fig.add_gridspec(top=.9, right=0.9).subplots()
            axx.set_xlabel('column')
            axx.set_ylabel('row')
        else:
            axx = fig_draw_panels(fig, xarr, side_panels)

        # add title to image panel
        if title is not None:
            axx.set_title(title)
        elif 'long_name' in xarr.attrs:
            axx.set_title(xarr.attrs['long_name'])

        # draw image and add colorbar
        img = axx.imshow(xarr.values, norm=xarr.attrs['_znorm'],
                         cmap=self.cmap if self.cmap else xarr.attrs['_cmap'],
                         aspect='equal', interpolation='none', origin='lower')
        adjust_img_ticks(axx, xarr)

        # define location of colorbar
        cax = make_axes_locatable(axx).append_axes("right", size=0.2, pad=0.05)
        _ = plt.colorbar(img, cax=cax, label=xarr.attrs['_zlabel'])

        # add annotation and save the figure
        median, spread = biweight(xarr.values, spread=True)
        if xarr.attrs['_zunits'] is None or xarr.attrs['_zunits'] == '1':
            fig_info.add('median', median, '{:.5g}')
            fig_info.add('spread', spread, '{:.5g}')
        else:
            fig_info.add('median', (median, xarr.attrs['_zunits']), '{:.5g} {}')
            fig_info.add('spread', (spread, xarr.attrs['_zunits']), '{:.5g} {}')

        # add annotation and save figure
        self.__add_copyright(axx)
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_quality(self, data, ref_data=None, *, fig_info=None,
                     side_panels='quality', title=None, **kwargs) -> None:
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
           Pass keyword arguments: 'thres_worst', 'thres_bad' or 'qlabels'
           to moniplot.lib.fig_draw_image.fig_qdata_to_xarr()

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

        # draw figure
        fig = plt.figure(figsize=fig_size(aspect, side_panels),
                         constrained_layout=True)
        self.__add_caption(fig)

        if side_panels == 'none':
            axx = fig.add_gridspec(top=.9, right=0.9).subplots()
            axx.set_xlabel('column')
            axx.set_ylabel('row')
        else:
            axx = fig_draw_panels(fig, xarr, side_panels)

        # add title to image panel
        if title is not None:
            axx.set_title(title)
        elif 'long_name' in xarr.attrs:
            axx.set_title(xarr.attrs['long_name'])

        # draw image and add colorbar
        img = axx.imshow(xarr.values,
                         cmap=xarr.attrs['_cmap'], norm=xarr.attrs['_znorm'],
                         aspect='equal', interpolation='none', origin='lower')
        adjust_img_ticks(axx, xarr)

        # define location of colorbar
        cax = make_axes_locatable(axx).append_axes("right", size=0.2, pad=0.05)
        bounds = xarr.attrs['flag_values']
        mbounds = [(bounds[ii+1] + bounds[ii]) / 2
                   for ii in range(len(bounds)-1)]
        _ = plt.colorbar(img, cax=cax, ticks=mbounds, boundaries=bounds)
        cax.tick_params(axis='y', which='both', length=0)
        cax.set_yticklabels(xarr.attrs['flag_meanings'])

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
        self.__add_copyright(axx)
        self.__add_fig_box(fig, fig_info)
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
    def draw_qhist(self, xds, *, density=True, exclude_region=None,
                   fig_info=None, title=None) -> None:
        """
        Display pixel-quality data as histograms.

        Parameters
        ----------
        xds :  xarray.Dataset, optional
           Object holding measurement data and attributes
        density : bool, optional
           If True, draw and return a probability density: each bin will
           display the bin's raw count divided by the total number of counts
           and the bin width (see matplotlib.pyplot.hist). Default is True
        exclude_region : numpy.ndarray, optional
           Provide a mask to define the area on the detector which should be
           excluded.
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure
        title :  str, optional
           Title of this figure (matplotlib: Axis.set_title)

        Examples
        --------
        Create a PDF document 'test.pdf' and add figure of dataset 'xds'
        (np.ndarray or xr.DataArray) with a title. The dataset 'xds' may
        contain multiple DataArrays with a common X-coordinate. Each DataArray
        will be displayed in a seperate sub-panel.

        >>> plot = MONplot('test.pdf', caption='my caption', institute='SRON')
        >>> plot.draw_trend(xds, title='my title')
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
        if title is not None:
            title = 'Histograms of pixel-quality'
        axarr[0].set_title(title, fontsize='large')

        # add figures with histograms
        for ii, (key, xda) in enumerate(xds.data_vars.items()):
            if isinstance(exclude_region, np.ndarray):
                # pylint: disable=invalid-unary-operand-type
                qdata = xda.values[~exclude_region].reshape(-1)
            else:
                qdata = xda.values.reshape(-1)
            label = xda.attrs['long_name'] if 'long_name' in xda.attrs else key
            fig_draw_qhist(axarr[ii], qdata, label, density)

        # finally add a label for the X-coordinate
        axarr[-1].set_xlabel('pixel quality')

        # add annotation and save the figure
        self.__add_copyright(axarr[-1])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_lplot(self, xdata, ydata, color=0, *,
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
        color :  integer, default=0
           [add line] Index to color in tol_colors.tol_cset('bright')
        fig_info  :  FIGinfo, optional
           [close figure] Meta-data to be displayed in the figure
        title :  str, default=None
           [close figure] Title of this figure (matplotlib: Axis.set_title)
        **kwargs :   other keywords
           [add line] Keywords are passed to mpl.pyplot.plot()
           [close figure] Kewords are passed to appropriate mpl.Axes method

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

            if fig_info is None:
                fig_info = FIGinfo()

            close_draw_lplot(self.__mpl['axx'], self.__mpl['time_axis'],
                             title, **kwargs)

            # add annotation and save the figure
            self.__add_copyright(self.__mpl['axx'])
            self.__add_fig_box(self.__mpl['fig'], fig_info)
            self.__close_this_page(self.__mpl['fig'])
            self.__mpl = None
            return

        # initialize figure
        if self.__mpl is None:
            if len(xdata) <= 256:
                figsize = (8, 8)
            elif 256 > len(xdata) <= 512:
                figsize = (10, 8)
            elif 512 > len(xdata) <= 768:
                figsize = (12, 8)
            else:
                figsize = (14, 8)

            self.__mpl = dict(zip(('fig', 'axx'),
                                  plt.subplots(1, figsize=figsize)))
            self.__mpl['time_axis'] = isinstance(xdata[0], datetime)

            # add a centered suptitle to the figure
            self.__add_caption(self.__mpl['fig'])

        # draw line in figure
        fig_draw_lplot(self.__mpl['axx'], xdata, ydata, color, **kwargs)

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
