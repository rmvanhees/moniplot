#
# This file is part of moniplot
#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022 SRON - Netherlands Institute for Space Research
# All rights reserved.
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
"""
This module contains the `MONplot` class with the plotting methods:
`draw_hist`, `draw_lplot`, `draw_multiplot`, `draw_qhist`, `draw_quality`,
`draw_signal`, `draw_tracks`, `draw_trend`.

"""
from datetime import datetime
from pathlib import PurePath

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
from matplotlib.gridspec import GridSpec

from .biweight import biweight

from .tol_colors import tol_rgba
from .lib.fig_info import FIGinfo
from .lib.fig_draw_image import (adjust_img_ticks,
                                 fig_data_to_xarr,
                                 fig_qdata_to_xarr,
                                 fig_draw_panels)
from .lib.fig_draw_trend import add_subplot, add_hk_subplot
from .lib.fig_draw_qhist import fig_draw_qhist
from .lib.fig_draw_lplot import fig_draw_lplot, close_draw_lplot
from .lib.fig_draw_multiplot import get_xylabels, draw_subplot
if FOUND_CARTOPY:
    from .lib.fig_draw_tracks import fig_draw_tracks

DEFAULT_CSET = 'bright'

# - local functions --------------------------------

# - main function ----------------------------------
class MONplot:
    """
    Generate PDF reports (or figures) to facilitate instrument calibration
    or monitoring.

    Parameters
    ----------
    figname :  str
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
    def __init__(self, figname, caption=None):
        """Initialize multi-page PDF document or a single-page PNG.
        """
        self.__cset = tol_rgba(DEFAULT_CSET)
        self.__cmap = None
        self.__caption = '' if caption is None else caption
        self.__institute = ''
        self.__mpl = None
        self.__pdf = None
        self.filename = figname
        if PurePath(figname).suffix.lower() != '.pdf':
            return

        self.__pdf = PdfPages(figname)

        # turn off Matplotlib's automatic offset notation
        mpl.rcParams['axes.formatter.useoffset'] = False

    def __repr__(self) -> None:
        pass

    def __close_this_page(self, fig) -> None:
        """Save the current figure and close the MONplot instance.
        """
        # add save figure
        if self.__pdf is None:
            plt.savefig(self.filename)
            plt.close(fig)
        else:
            self.__pdf.savefig()

    def close(self) -> None:
        """Close PNG or (multipage) PDF document.
        """
        if self.__pdf is None:
            return

        # add PDF annotations
        doc = self.__pdf.infodict()
        if self.__caption is not None:
            doc['Title'] = self.__caption
        doc['Subject'] = \
            'Generated using https://github.com/rmvanhees/moniplot.git'
        if self.__institute == 'SRON':
            doc['Author'] = '(c) SRON Netherlands Institute for Space Research'
        elif self.__institute:
            doc['Author'] = f'(c) {self.__institute}'
        self.__pdf.close()
        plt.close('all')

    # --------------------------------------------------
    @property
    def caption(self) -> str:
        """Return figure caption.
        """
        return self.__caption

    def set_caption(self, caption: str) -> None:
        """Set caption of each page of the PDF.

        Parameters
        ----------
        institute :  str
           Provide abbreviation of the name of your institute to be used in
           the copyright statement in the main panel of the figures.
        """
        self.__caption = caption

    def __add_caption(self, fig):
        """Add figure caption.
        """
        if not self.caption:
            return

        fig.suptitle(self.caption, fontsize='x-large',
                     position=(0.5, 1 - 0.3 / fig.get_figheight()))

    # --------------------------------------------------
    @property
    def cmap(self):
        """Return matplotlib colormap.
        """
        return self.__cmap

    def set_cmap(self, cmap) -> None:
        """Use alternative colormap for MONplot::draw_image.

        Parameters
        ----------
        cmap :  matplotlib colormap
        """
        self.__cmap = cmap

    def unset_cmap(self) -> None:
        """Unset user supplied colormap, and use default colormap.
        """
        self.__cmap = None

    # --------------------------------------------------
    @property
    def cset(self) -> str:
        """Return name of current color-set.
        """
        return self.__cset

    def set_cset(self, cname: str, cnum=None) -> None:
        """Use alternative color-set through which `draw_lplot` will cycle.

        Parameters
        ----------
        cname :  str
           Name of color set. Use None to get the default matplotlib value.
        cnum : int, optional
           Number of discrete colors in colormap (*not colorset*).
        """
        self.__cset = tol_rgba(cname, cnum)

    def unset_cset(self) -> None:
        """Set color set to its default.
        """
        self.__cset = tol_rgba(DEFAULT_CSET)

    # --------------------------------------------------
    @property
    def institute(self) -> str:
        """Return name of your institute.
        """
        return self.__institute

    def set_institute(self, institute: str) -> None:
        """Use the name of your institute as a signature.

        Parameters
        ----------
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
        """Add a box with meta information in the current figure.

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

    # -------------------------
    def __draw_image__(self, xarr: xr.DataArray, side_panels: str,
                       fig_info: FIGinfo, title: str) -> None:
        """Does the actual drawing of the image data for the public methods
        `draw_signal` and `draw_quality`.
        """
        def add_fig_box() -> None:
            """Add a box with meta information in the current figure.
            """
            if fig_info is None:
                return

            if fig_info.location == 'above':
                if aspect <= 2:
                    halign = 'center'
                    fontsize = 'x-small'
                else:
                    halign = 'right'
                    fontsize = 'xx-small' if len(fig_info) > 6 else 'x-small'

                axx_c.text(0 if aspect == 2 else 1,
                           1.04 + (aspect-1) * 0.0075,
                           fig_info.as_str(), fontsize=fontsize,
                           transform=axx_c.transAxes,
                           multialignment='left',
                           verticalalignment='bottom',
                           horizontalalignment=halign,
                           bbox={'facecolor': 'white', 'pad': 4})
                return

            if fig_info.location == 'below':
                fontsize = 'xx-small' if aspect in (3, 4) else 'x-small'
                axx_c.text(0.125 + (aspect-1) * 0.2,
                           -0.03 - (aspect-1) * 0.005,
                           fig_info.as_str(), fontsize=fontsize,
                           transform=axx_c.transAxes,
                           multialignment='left',
                           verticalalignment='top',
                           horizontalalignment='left',
                           bbox={'facecolor': 'white', 'pad': 4})

        # aspect of image data
        aspect = min(4, max(1, int(round(xarr.shape[1] / xarr.shape[0]))))

        # select figure attributes
        attrs = {1: {'figsize': (10, 8),
                     'w_ratios': (1., 7., 0.5, 1.5),
                     'h_ratios': (7., 1.)},                  # 7 x 7
                 2: {'figsize': (13, 6.25),
                     'w_ratios': (1., 10., 0.5, 1.5),
                     'h_ratios': (5., 1.)},                  # 10 x 5
                 3: {'figsize': (15, 5.375),
                     'w_ratios': (1., 12., 0.5, 1.5),
                     'h_ratios': (4., 1.)},                  # 12 x 4
                 4: {'figsize': (17, 5.125),
                     'w_ratios': (1., 14., 0.5, 1.5),
                     'h_ratios': (3.5, 1.)}}.get(aspect)     # 14 x 3.5

        # define matplotlib figure
        fig = plt.figure(figsize=attrs['figsize'])
        if self.caption:
            fig.suptitle(self.caption, fontsize='x-large',
                         position=(0.5, 1 - 0.4 / fig.get_figheight()))

        # Define a grid layout to place subplots within the figure.
        # - gspec[0, 1] is reserved for the image
        # - gspec[1, 1] is reserved for the x-panel
        # - gspec[0, 0] is reserved for the y-panel
        # - gspec[0, 2] is reserved for the colorbar
        # - gspec[1, 2] is used to pace the small fig_info box (max 6/7 lines)
        gspec = fig.add_gridspec(2, 4,
                                 left=.135 + .005 * (aspect-1),
                                 right=.9 - .005 * (aspect-1),
                                 top=.865 - .025 * (aspect-1),
                                 bottom=.115 + .01 * (aspect-1),
                                 wspace=0.1 / max(2, aspect-1),
                                 hspace=0.05,
                                 width_ratios=attrs['w_ratios'],
                                 height_ratios=attrs['h_ratios'])

        # add image panel and draw image
        axx = fig.add_subplot(gspec[0, 1])
        if xarr.attrs['_zscale'] == 'quality':
            img = axx.imshow(xarr.values, norm=xarr.attrs['_znorm'],
                             aspect='auto', cmap=xarr.attrs['_cmap'],
                             interpolation='none', origin='lower')
        else:
            cmap = self.cmap if self.cmap else xarr.attrs['_cmap']
            img = axx.imshow(xarr.values, norm=xarr.attrs['_znorm'],
                             aspect='auto', cmap=cmap,
                             interpolation='none', origin='lower')

        # add title to image panel
        if title is not None:
            axx.set_title(title)
        elif 'long_name' in xarr.attrs:
            axx.set_title(xarr.attrs['long_name'])
        self.__add_copyright(axx)
        # axx.grid(True)

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

        # add colorbar
        if xarr.attrs['_zscale'] == 'quality':
            axx_c = fig.add_subplot(gspec[0, 2])
            bounds = xarr.attrs['flag_values']
            mbounds = [(bounds[ii+1] + bounds[ii]) / 2
                       for ii in range(len(bounds)-1)]
            _ = plt.colorbar(img, cax=axx_c, ticks=mbounds, boundaries=bounds)
            axx_c.tick_params(axis='y', which='both', length=0)
            axx_c.set_yticklabels(xarr.attrs['flag_meanings'])
        else:
            axx_c = fig.add_subplot(gspec[0, 2])
            _ = plt.colorbar(img, cax=axx_c, label=xarr.attrs['_zlabel'])

        # add annotation and save the figure
        add_fig_box()
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_signal(self, data, *, fig_info=None, side_panels='nanmedian',
                    title=None, **kwargs) -> None:
        """Display 2D array as an image and averaged column/row signal
        in the side-panels (optional).

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes.
        fig_info :  FIGinfo, default=None
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
        When data is an xarray.DataArray then the following attributes
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
        (`numpy.ndarray` or `xarray.DataArray`) with side-panels and title.

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
        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and '_zscale' in data.attrs:
            xarr = data.copy()
        else:
            xarr = fig_data_to_xarr(data, **kwargs)

        # add data statistics to fig_info
        if fig_info is None:
            fig_info = FIGinfo()

        median, spread = biweight(xarr.values, spread=True)
        if xarr.attrs['_zunits'] is None or xarr.attrs['_zunits'] == '1':
            fig_info.add('median', median, '{:.5g}')
            fig_info.add('spread', spread, '{:.5g}')
        else:
            fig_info.add('median', (median, xarr.attrs['_zunits']), '{:.5g} {}')
            fig_info.add('spread', (spread, xarr.attrs['_zunits']), '{:.5g} {}')

        # draw actual image
        self.__draw_image__(xarr, side_panels, fig_info, title)

    # --------------------------------------------------
    def draw_quality(self, data, ref_data=None, *, side_panels='quality',
                     fig_info=None, title=None, **kwargs) -> None:
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
        When data is an xarray.DataArray then the following attributes
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

        Otherwise the labels for quality ranking indicate which pixels have
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
        (`numpy.ndarray` or `xarray.DataArray`) with side-panels and title

        >>> plot = MONplot('test.pdf', caption='my caption', institute='SRON')
        >>> plot.draw_quality(img, title='my title')

        Add the same figure without side-panels

        >>> plot.draw_quality(img, side_panels='none', title='my title')

        Add a figure where img_ref is a quality map from early in the mission

        >>> plot.draw_quality(img, img_ref, title='my title')

        Finalize the PDF file

        >>> plot.close()

        """
        # convert, if necessary, input data to xarray.DataArray
        if isinstance(data, xr.DataArray) and '_zscale' in data.attrs:
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

        # draw actual image
        self.__draw_image__(xarr, side_panels, fig_info, title)

    # --------------------------------------------------
    def draw_trend(self, xds=None, hk_xds=None, *,
                   fig_info=None, title=None, **kwargs) -> None:
        """
        Display trends of measurement data and/or housekeeping data

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
        add_hk_subplot : Add a subplot for housekeeping data.

        Notes
        -----
        When data is an xarray.DataArray then the following attributes are used:

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
        margin = min(1. / (1.65 * (npanels + 1)), .25)
        fig.subplots_adjust(bottom=margin, top=1-margin, hspace=0.05)

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
        r"""Display data as histograms.

        Parameters
        ----------
        data :  numpy.ndarray or xarray.DataArray
           Object holding measurement data and attributes.
        data_sel :  mask or index tuples for arrays, optional
           Select a region on the detector by fancy indexing (using a
           boolean/interger arrays), or using index tuples for arrays
           (generated with `numpy.s\_`).
        vrange :  list, default=[data.min(), data.max()]
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
        When data is an xarray.DataArray then the following attributes are used:

        - long_name: used as the title of the main panel when parameter 'title'\
          is not defined.
        - units: units of the data

        Examples
        --------
        Create a PDF document 'test.pdf' with two pages.
        Both pages have the caption "My Caption", the title of the figure on
        the first page is "my title" and on the second page the title of the
        figure is f'Histogram of {xarr.attrs['long_name']}" when xarr has
        attribute "long_name".

        >>> plot = MONplot('test.pdf', caption='My Caption')
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
            if 'units' in data.attrs:
                zunits = data.attrs['units']
            if 'long_name' in data.attrs:
                long_name = data.attrs['long_name']
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
        fig, axx = plt.subplots(1, figsize=(9, 8))

        # add a centered suptitle to the figure
        self.__add_caption(fig)

        # add title to image panel and set xlabel
        xlabel = 'value' if zunits == '1' else f'value [{zunits}]'
        if title is None:
            title = f'Histogram of {long_name}'
        elif long_name:
            xlabel = long_name if zunits == '1' else f'{long_name} [{zunits}]'
        axx.set_title(title)

        # add histogram
        if vrange is not None:
            values = np.clip(values, vrange[0], vrange[1])
        # Edgecolor is tol_cset('bright').blue
        if 'bins' in kwargs and kwargs['bins'] > 24:
            axx.hist(values, range=vrange, histtype='step',
                     edgecolor='#4477AA', facecolor='#77AADD',
                     fill=True, linewidth=1.5, **kwargs)
            axx.grid(which='major', color='#AAAAAA', linestyle='--')
        else:
            axx.hist(values, range=vrange, histtype='bar',
                     edgecolor='#4477AA', facecolor='#77AADD',
                     linewidth=1.5, **kwargs)
            axx.grid(which='major', axis='y', color='#AAAAAA', linestyle='--')
        axx.set_xlabel(xlabel)
        if 'density' in kwargs and kwargs['density']:
            axx.set_ylabel('density')
        else:
            axx.set_ylabel('number')

        if len(fig_info) > 3:
            plt.subplots_adjust(top=.875)

        # add annotation and save the figure
        self.__add_copyright(axx)
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_qhist(self, xds, data_sel=None, density=True,
                   fig_info=None, title=None) -> None:
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
        When data is an xarray.DataArray then the following attributes are used:

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
    def draw_lplot(self, xdata, ydata, *, square=False,
                   fig_info=None, title=None, **kwargs) -> None:
        """Plot y versus x lines, maybe called multiple times to add lines.
        Figure is closed when called with xdata equals None.

        Parameters
        ----------
        xdata :  ndarray
           ``[add line]`` X data;
           ``[close figure]`` when xdata is None.
        ydata :  ndarray
           ``[add line]`` Y data.
        square :  bool
           ``[add line]`` create a square figure,
           independent of number of data-points (*first call, only*).
        fig_info  :  FIGinfo, optional
           ``[close figure]`` Meta-data to be displayed in the figure.
        title :  str, default=None
           ``[close figure]`` Title of this figure using `Axis.set_title`.
        **kwargs :   other keywords
           ``[add line]`` Keywords are passed to `matplotlib.pyplot.plot`;
           ``[close figure]`` Keywords are passed to appropriate
           `matplotlib.Axes` method;
           ``[close figure]`` Keyword 'text' can be used to add addition
           text in the upper left corner.

        Examples
        --------
        General example:

        >>> plot = MONplot(fig_name)
        >>> for ii, ix, iy in enumerate(data_of_each_line):
        >>>    plot.draw_lplot(ix, iy, label=mylabel[ii],
        >>>                    marker='o', linestyle='None')
        >>> plot.draw_lplot(None, None, xlim=[0, 0.5], ylim=[-10, 10],
        >>>                 xlabel=my_xlabel, ylabel=my_ylabel)
        >>> plot.close()

        Using a time-axis:

        >>> from datetime import datetime, timedelta
        >>> tt0 = (datetime(year=2020, month=10, day=1)
        >>>        + timedelta(seconds=sec_in_day))
        >>> tt = [tt0 + iy * t_step for iy in range(yy.size)]
        >>> plot = MONplot(fig_name)
        >>> plot.draw_lplot(tt, yy, label=mylabel,
        >>>                 marker='o', linestyle='None')
        >>> plot.draw_line(None, None, ylim=[-10, 10],
        >>>                xlabel=my_xlabel, ylabel=my_ylabel)
        >>> plot.close()

        You can use different sets of colors and cycle through them.
        First, we use default colors defined by matplotlib:

        >>> plot = MONplot('test_lplot.pdf')
        >>> plot.set_cset(None)
        >>> for i in range(5):
        >>>    plot.draw_lplot(np.arange(10), np.arange(10)*(i+1))
        >>> plot.draw_lplot(None, None)

        You can also assign colors to each line:

        >>> clr = 'rgbym'
        >>> for i in range(5):
        >>>    plot.draw_lplot(np.arange(10), np.arange(10)*(i+1), color=clr[i])
        >>> plot.draw_lplot(None, None)

        You can use one of the color sets as defined in ``tol_colors``:

        >>> plot.set_cset('mute')   # the default is 'bright'
        >>> for i in range(5):
        >>>    plot.draw_lplot(np.arange(10), np.arange(10)*(i+1))
        >>> plot.draw_lplot(None, None)

        Or you can use a color map as defined in ``tol_colors`` where
        you can define the number of colors you need. If you need less than 24
        colors, you can use 'rainbow_discrete' or you can choose an other
        color map if you need more colors, for example:

        >>> plot.set_cset('rainbow_PuBr', 25)
        >>> for i in range(25):
        >>>     plot.draw_lplot(np.arange(10), np.arange(10)*(i+1))
        >>> plot.draw_lplot(None, None)

        """
        if xdata is None:
            if self.__mpl is None:
                raise ValueError('No plot defined and no data provided')
            fig = self.__mpl['fig']
            axx = self.__mpl['axx']

            if fig_info is None:
                fig_info = FIGinfo()

            if 'text' in kwargs:
                axx.text(0.05, 0.985, kwargs.pop('text'),
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
                figsize = {0: (10, 7),
                           1: (10, 7),
                           2: (12, 7)}.get(len(xdata) // 256, (14, 8))

            self.__mpl = dict(zip(('fig', 'axx'),
                                  plt.subplots(1, figsize=figsize)))
            self.__mpl['time_axis'] = isinstance(xdata[0], datetime)

            # add a centered suptitle to the figure
            self.__add_caption(self.__mpl['fig'])

            # set color cycle
            if self.cset is None:
                self.__mpl['axx'].set_prop_cycle(None)
            else:
                self.__mpl['axx'].set_prop_cycle(color=self.cset)

        # draw line in figure
        fig_draw_lplot(self.__mpl['axx'], xdata, ydata, **kwargs)

    # --------------------------------------------------
    def draw_multiplot(self, data_tuple: tuple, gridspec=None, *,
                       fig_info=None, title=None, **kwargs) -> None:
        """Display multiple subplots on one page using
        `matplotlib.gridspec.GridSpec`.

        Parameters
        ----------
        data_tuple :  tuple of np.ndarray, xarray.DataArray or xarray.Dataset
           One dataset per subplot.
        gridspec :  matplotlib.gridspec.GridSpec, optional
           Instance of `matplotlib.gridspec.GridSpec`.
        fig_info  :  FIGinfo, optional
           Meta-data to be displayed in the figure.
        title :  str, default=None
           Title of this figure using `Axis.set_title`.
           Ignored when data is a `xarray` data-structure.
        **kwargs :   other keywords
           Keywords are passed to `matplotlib.pyplot.plot`.
           Ignored when data is a `xarray` data-structure.

        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.

        Notes
        -----
        When data is an xarray.DataArray then the following attributes are used:

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
        using np.range(ndarray1) and  np.range(ndarray2):

        >>> data_tuple = (ndarray1, ndarray2)
        >>> plot = MONplot(fig_name)
        >>> plot.draw_multiplot(data_tuple, title='my title',
        >>>                     marker='o', linestyle='', color='r')
        >>> plot.close()

        Show four DataArrays, each in a different panel. The subplots
        are above each other in 2 columns (row=2, col=2). The X-coordinates
        are generated from the first dimension of the DataArrays:

        >>> data_tuple = (xarr1, xarr2, xarr3, xarr4)
        >>> plot = MONplot(fig_name)
        >>> plot.draw_multiplot(data_tuple, title='my title',
        >>>                     marker='o', linestyle='')
        >>> plot.close()

        Show the DataArrays in a Dataset, each in a different panel. If there
        are 3 DataArrays preset then the subplots are above each other (row=3,
        col=1). The X-coordinates are generated from the (shared?) first
        dimension of the DataArrays:

        >>> plot = MONplot(fig_name)
        >>> plot.draw_multiplot(xds, title='my title')
        >>> plot.close()

        """
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

        # determine xylabels
        xylabels = get_xylabels(gridspec, data_tuple)

        # add a centered suptitle to the figure
        self.__add_caption(fig)

        # add subplots, cycle the DataArrays of the Dataset
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
    def draw_tracks(self, lons, lats, icids, *, saa_region=None,
                    fig_info=None, title=None) -> None:
        """Display tracks of satellite on a world map
        using a Robinson projection.

        Parameters
        ----------
        lons :  (N, 2) array-like
           Longitude coordinates at start and end of measurement.
        lats :  (N, 2) array-like
           Latitude coordinates at start and end of measurement.
        icids :  (N) array-like
           ICID of measurements per (lon, lat).
        saa_region :  (N, 2) array-like, optional
           The coordinates of the vertices. When defined, then show SAA region
           as a matplotlib polygon patch.
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure.

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        if not FOUND_CARTOPY:
            raise RuntimeError("You need Cartopy to use this method")

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
