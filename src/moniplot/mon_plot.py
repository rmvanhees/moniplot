"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

The class MONplot contains generic plot functions

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
from pathlib import PurePath

import numpy as np
import xarray as xr

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


# - local functions --------------------------------
def fig_size(aspect: int, side_panels: str):
    """
    Calculate size of the figure (inches)
    """
    sz_corr = {4: (2.5, 0.85 if side_panels == 'none' else 1.),
               3: (2.25, 1. if side_panels == 'none' else 1.25),
               2: (2.0, 1.2 if side_panels == 'none' else 1.45),
               1: (1.75, 1.75)}.get(aspect)
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
    set_cmap(cmap)
       Define alternative color-map to overrule the default.
    unset_cmap()
       Unset user supplied color-map, and use default color-map.
    get_cmap(zscale='linear')
       Returns matplotlib colormap.
    set_zunit(units)
       Provide units of data to be displayed.
    unset_zunit()
       Unset user supplied unit definition of data.
    zunit
       Returns value of zunit (property).
    draw_signal(data, vperc=None, vrange=None, zscale='linear', fig_info=None,
                side_panels='nanmedian', title=None, sub_title=None)
       Display 2D array data as image and averaged column/row signal plots.

    Notes
    -----
    ...

    Examples
    --------
    ...
    """
    def __init__(self, figname, caption=None, institute=None, pdf_title=None):
        """
        Initialize multi-page PDF document or a single-page PNG

        Parameters
        ----------
        figname :  str
           Name of PDF or PNG file (extension required)
        caption :  str
           Caption repeated on each page of the PDF
        institute :  str, default=None
           Provide abbreviation of the name of your institute to be used in
           the copyright statement in the main panel of the figures.
        pdf_title :  string
           Title of the PDF document (attribute of the PDF document)
           Default: 'Monitor report on Tropomi SWIR instrument'
        """
        self.__cmap = None
        self.__caption = '' if caption is None else caption
        self.__institute = '' if institute is None else institute
        self.__pdf = None
        self.filename = figname
        if PurePath(figname).suffix.lower() != '.pdf':
            return

        self.__pdf = PdfPages(figname)
        # add annotation
        doc = self.__pdf.infodict()
        if pdf_title is None:
            doc['Title'] = 'Monitor report on Tropomi SWIR instrument'
        else:
            doc['Title'] = pdf_title
        if self.__institute == 'SRON':
            doc['Author'] = '(c) SRON Netherlands Institute for Space Research'
        elif self.__institute:
            doc['Author'] = f'(c) {self.__institute}'

    def __repr__(self):
        pass

    def __close_this_page(self, fig):
        """
        Close current matplotlib figure or page in a PDF document
        """
        # add save figure
        if self.__pdf is None:
            plt.savefig(self.filename)
            plt.close(fig)
        else:
            self.__pdf.savefig()

    def close(self):
        """
        Close PNG or (multipage) PDF document
        """
        if self.__pdf is None:
            return

        self.__pdf.close()
        plt.close('all')

    # --------------------------------------------------
    @property
    def caption(self):
        """
        Return figure caption
        """
        return self.__caption

    # --------------------------------------------------
    def set_cmap(self, cmap):
        """
        Define alternative color-map to overrule the default

        Parameter
        ---------
         cmap :  matplotlib color-map
        """
        self.__cmap = cmap

    def unset_cmap(self):
        """
        Unset user supplied color-map, and use default color-map
        """
        self.__cmap = None

    @property
    def cmap(self):
        """
        Returns matplotlib colormap
        """
        return self.__cmap

    # --------------------------------------------------
    def set_institute(self, institute: str):
        """
        Define name of your institute

        Parameter
        ---------
        institute :  str
        """
        self.__institute = institute

    @property
    def institute(self):
        """
        Returns name of your institute
        """
        return self.__institute

    # --------------------------------------------------
    def __add_copyright(self, axx):
        """
        Display copyright in current figure (main panel)
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
        Add meta-information in the current figure

        Parameters
        ----------
        fig :  Matplotlib figure instance
        fig_info :  FIGinfo
           instance of pys5p.lib.plotlib.FIGinfo to be displayed
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
            return

    # --------------------------------------------------
    def draw_signal(self, data, zscale=None, *, fig_info=None,
                    side_panels='nanmedian', title=None):
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes
        zscale : str, default='linear'
           Scaling of the data values. Valid values are: 'linear', 'log',
           'diff' or 'ratio'.
        fig_info :  FIGinfo, default=None
           OrderedDict holding meta-data to be displayed in the figure
        side_panels :  str, default='nanmedian'
           Show image row and column statistics in two side panels.
           Use 'none' when you do not want the side panels.
           Other valid values are: 'median', 'nanmedian', 'mean', 'nanmean',
           'quality', 'std' and 'nanstd'.
        title :  str, default=None
           Title of this figure (matplotlib: sub_title)

        The information provided in the parameter 'fig_info' will be displayed
        in a text box. In addition, we display the creation date and the data
        (biweight) median & spread.
        """
        # initialize keyword parameters
        if zscale is None:
            zscale = 'linear'
        if fig_info is None:
            fig_info = FIGinfo()

        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and '_zscale' in data.attrs:
            xarr = data.copy()
        else:
            xarr = fig_data_to_xarr(data, zscale, self.cmap)

        # aspect of image data
        aspect = min(4, max(1, int(round(xarr.shape[1] / xarr.shape[0]))))

        # draw figure
        fig = plt.figure(figsize=fig_size(aspect, side_panels),
                         constrained_layout=True)
        if title is not None:
            fig.suptitle(self.caption + '\n',
                         fontsize='x-large',
                         horizontalalignment='center')

        if side_panels == 'none':
            axx = fig.add_gridspec(top=1).subplots()
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
                     side_panels='quality', title=None):
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
           Title of this figure (matplotlib: sub_title)

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
        """
        if fig_info is None:
            fig_info = FIGinfo()

        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and '_zscale' in data.attrs:
            xarr = data
        else:
            xarr = fig_qdata_to_xarr(data, ref_data)

        # aspect of image data
        aspect = min(4, max(1, int(round(xarr.shape[1] / xarr.shape[0]))))

        # draw figure
        fig = plt.figure(figsize=fig_size(aspect, side_panels),
                         constrained_layout=True)
        if title is not None:
            fig.suptitle(self.caption + '\n',
                         fontsize='x-large',
                         horizontalalignment='center')

        if side_panels == 'none':
            axx = fig.add_gridspec(top=1).subplots()
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
    def draw_trend(self, xds=None, hk_xds=None, vrange_last_orbits=-1, *,
                   fig_info=None, title=None):
        """
        Display trends of measurement and house-keeping data

        Parameters
        ----------
        xds :  xarray.Dataset, optional
           Object holding measurement data and attributes
        hk_xds :  xarray.Dataset, optional
           Object holding housekeeping data and attributes
        vrange_last_orbits :  int, default=-1
           Use the last N orbits to determine vrange (orbit coordinate only)
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure
        title :  str, optional
           Title of this figure (matplotlib: sub_title)
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

        # draw title of the Figure
        if self.caption:
            fig.suptitle(self.caption + '\n',
                         fontsize='x-large',
                         horizontalalignment='center')

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
                add_hk_subplot(axarr[ipanel], hk_xds[name], vrange_last_orbits)
                ipanel += 1

        # finally add a label for the X-coordinate
        axarr[-1].set_xlabel(xlabel)

        # add annotation and save the figure
        self.__add_copyright(axarr[-1])
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)
