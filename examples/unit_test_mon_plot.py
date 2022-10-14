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

from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from moniplot.image_to_xarray import data_to_xr
from moniplot.lib.fig_info import FIGinfo
from moniplot.mon_plot import MONplot


def get_test_data(data_sel=None, xy_min=-5, xy_max=5, delta=0.01, error=0):
    """
    Generate synthetic data to simulate a square-detector image
    """
    if data_sel is None:
        data_sel = [(), ()]

    res = np.arange(xy_min, xy_max, delta)
    xmesh, ymesh = np.meshgrid(res[data_sel[1]], res[data_sel[0]])
    zz1 = np.exp(-xmesh ** 2 - ymesh ** 2)
    zz2 = np.exp(-(xmesh - 1) ** 2 - (ymesh - 1) ** 2)
    data = (zz1 - zz2) * 2
    data += np.random.default_rng().normal(0., error, data.shape)

    return data_to_xr(data, long_name='bogus data', units='Volt')


def run_draw_signal(plot):
    """
    Run unit tests on MONplot::draw_signal
    """
    print('Run unit tests on MONplot::draw_signal')
    msm = get_test_data(error=.1)
    # msm_ref = get_test_data(error=0.025)

    # image aspect=4
    plot.draw_signal(msm, zscale='linear', side_panels='none',
                     title='method=linear; aspect=1; fig_pos=above')
    plot.draw_signal(msm, zscale='linear',
                     title='method=linear; aspect=1; fig_pos=above')
    plot.draw_signal(msm, zscale='diff',
                     title='method=diff; aspect=1; fig_pos=above')
    msm1 = msm.copy()
    msm1.values = (msm1.values + 3) / 3
    msm1.attrs['units'] = '1'
    plot.draw_signal(msm1, zscale='ratio',
                     title='method=ratio; aspect=1; fig_pos=above')
    plot.draw_signal(np.abs(msm), zscale='log', side_panels='none',
                     title='method=error; aspect=1; fig_pos=above')
    # plot.draw_signal(msm, fig_info=fig_info_in.copy(),
    #                 title='method=linear; aspect=1; fig_pos=right')

    # image aspect=2
    msm = get_test_data(data_sel=np.s_[500-250:500+250, :], error=.1)
    plot.draw_signal(msm, side_panels='none',
                     title='method=linear; aspect=2; fig_pos=above')
    plot.draw_signal(msm,
                     title='method=linear; aspect=2; fig_pos=above')
    # plot.draw_signal(msm, fig_info=fig_info_in.copy(),
    #                 title='method=linear; aspect=2; fig_pos=right')

    # image aspect=3
    msm = get_test_data(data_sel=np.s_[500-125:500+125, 500-375:500+375],
                        error=.1)
    plot.draw_signal(msm, side_panels='none',
                     title='method=linear; aspect=3; fig_pos=above')
    plot.draw_signal(msm,
                     title='method=linear; aspect=3; fig_pos=above')
    # plot.draw_signal(msm,
    #                 fig_info=fig_info_in.copy(),
    #                 title='method=linear; aspect=3; fig_pos=right')

    # image aspect=4
    msm = get_test_data(data_sel=np.s_[500-128:500+128, :], error=.1)
    plot.draw_signal(msm, side_panels='none',
                     title='method=linear; aspect=4; fig_pos=above')
    plot.draw_signal(msm,
                     title='method=linear; aspect=4; fig_pos=above')
    # plot.draw_signal(msm, fig_info=fig_info_in.copy(),
    #                 title='method=linear; aspect=4; fig_pos=right')


def run_draw_quality(plot):
    """
    Run unit tests on MONplot::draw_quality
    """
    print('Run unit tests on MONplot::draw_quality')
    row = np.linspace(0, 1., 1000)
    ref_data = np.repeat(row[None, :], 256, axis=0)

    data = ref_data.copy()
    data[125:175, 30:50] = 0.4
    data[125:175, 50:70] = 0.95
    data[75:125, 250:300] = 0.05
    data[125:175, 600:650] = 0.95
    data[75:125, 900:925] = 0.05
    data[75:125, 825:875] = 0.4

    figinfo_in = FIGinfo()
    figinfo_in.add('orbits', (17, [23662, 23707]), fmt='{} in {}')
    figinfo_in.add('coverage', ('2022-05-08', '2022-05-10'), fmt='{} / {}')

    plot.draw_quality(data, data_sel=np.s_[11:228, 16:991],
                      title='no reference', fig_info=figinfo_in.copy())
    plot.draw_quality(data, ref_data=ref_data, data_sel=np.s_[11:228, 16:991],
                      title='with reference', fig_info=figinfo_in.copy())
    plot.draw_quality(data, data_sel=np.s_[11:228, 16:991],
                      side_panels='none', title='no reference',
                      fig_info=figinfo_in.copy())


def run_draw_trend(plot):
    """
    Run unit tests on MONplot::draw_trend
    """
    print('Run unit tests on MONplot::draw_trend')
    n_elmnt = 200
    xval = np.arange(n_elmnt) / 100

    res = []
    hk_dtype = np.dtype([('mean', 'f8'), ('err1', 'f8'), ('err2', 'f8')])
    buff = np.empty(len(xval), dtype=hk_dtype)
    data = 140. + (100 - np.arange(n_elmnt)) / 1000
    buff['mean'] = data
    buff['err1'] = data - .0125
    buff['err2'] = data + .0075
    hk_attrs = {'long_name': 'SWIR detector temperature', 'units': 'K'}
    res.append(xr.DataArray(buff, name='detector_temp', attrs=hk_attrs,
                            coords={'orbit': np.arange(n_elmnt)}))

    buff = np.empty(len(xval), dtype=hk_dtype)
    data = 202.1 + (100 - np.arange(n_elmnt)) / 1000
    buff['mean'] = data
    buff['err1'] = data - .15
    buff['err2'] = data + .175
    hk_attrs = {'long_name': 'SWIR grating temperature', 'units': 'K'}
    res.append(xr.DataArray(buff, name='grating_temp', attrs=hk_attrs,
                            coords={'orbit': np.arange(n_elmnt)}))

    buff = np.empty(len(xval), dtype=hk_dtype)
    data = 208.2 + (100 - np.arange(n_elmnt)) / 1000
    buff['mean'] = data
    buff['err1'] = data - .15
    buff['err2'] = data + .175
    hk_attrs = {'long_name': 'SWIR OBM temperature', 'units': 'K'}
    res.append(xr.DataArray(buff, name='obm_temp', attrs=hk_attrs,
                            coords={'orbit': np.arange(n_elmnt)}))
    hk_ds = xr.merge(res, combine_attrs="drop_conflicts")

    msm1 = data_to_xr(np.sin(xval * np.pi), name='msm1', dims=['orbit'])
    msm2 = data_to_xr(np.cos(xval * np.pi), name='msm2', dims=['orbit'])
    msm_ds = xr.merge((msm1, msm2), combine_attrs="drop_conflicts")

    # plot.draw_trend(msm_ds, title='one dataset, no house-keeping')
    plot.draw_trend(xds=msm_ds, title='two datasets, no house-keeping')
    plot.draw_trend(hk_xds=hk_ds, title='no datasets, only house-keeping')
    plot.draw_trend(msm_ds, hk_ds, title='two datasets and house-keeping')


def run_draw_lplot(plot):
    """
    Run unit tests on MONplot::draw_lplot
    """
    print('Run unit tests on MONplot::draw_lplot')
    xval = np.arange(200) / 100
    plot.draw_lplot(xval, np.sin(xval * np.pi), label='sinus',
                    marker='o', linestyle='-')
    plot.draw_lplot(xval, np.cos(xval * np.pi), label='cosinus',
                    marker='o', linestyle='-')
    plot.draw_lplot(None, None, ylim=[-1.05, 1.05],
                    xlabel='x-axis [Pi]', ylabel='y-axis',
                    title='draw_lplot [no time_axis]')

    xval = np.arange(500) / 100
    plot.draw_lplot(xval, np.sin(xval * np.pi), label='sinus',
                    marker='o', linestyle='-')
    plot.draw_lplot(xval, np.cos(xval * np.pi), label='cosinus',
                    marker='o', linestyle='-')
    plot.draw_lplot(None, None, ylim=[-1.05, 1.05],
                    xlabel='x-axis [Pi]', ylabel='y-axis',
                    title='draw_lplot [no time_axis]')

    customdate = datetime(2016, 1, 1, 13, 0, 0)
    yval = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    xval = [customdate + timedelta(hours=i, minutes=4*i)
            for i in range(len(yval))]
    plot.draw_lplot(xval, yval, label='mydata', marker='o', linestyle='-')
    plot.draw_lplot(None, None, title='draw_lplot [time_axis]',
                    xlabel='x-axis', ylabel='y-axis')


def run_draw_qhist(plot):
    """
    Run unit tests on MONplot::draw_qhist
    """
    print('Run unit tests on MONplot::draw_qhist')
    buff0 = np.repeat(0.9 + np.random.rand(1000) / 10, 56)
    buff1 = np.repeat(np.random.rand(1000) / 10, 10)
    buff2 = 0.1 + np.random.rand(1000) / 10
    buff3 = np.repeat(0.2 + np.random.rand(1000) / 10, 2)
    buff4 = np.repeat(0.3 + np.random.rand(1000) / 10, 3)
    buff5 = np.repeat(0.4 + np.random.rand(1000) / 10, 4)
    buff6 = np.repeat(0.5 + np.random.rand(1000) / 10, 8)
    buff7 = np.repeat(0.6 + np.random.rand(1000) / 10, 12)
    buff8 = np.repeat(0.7 + np.random.rand(1000) / 10, 20)
    buff9 = np.repeat(0.8 + np.random.rand(1000) / 10, 40)
    buffa = np.repeat(0.9 + np.random.rand(1000) / 10, 100)
    frame = np.concatenate((buff0, buff1, buff2, buff3, buff4,
                            buff5, buff6, buff7, buff8, buff9,
                            buffa)).reshape(256, 1000)
    msm = xr.merge([data_to_xr(frame, name='dpqm',
                               long_name='pixel-quality map'),
                    data_to_xr(frame, name='dpqm_dark',
                               long_name='pixel-quality map (dark)'),
                    data_to_xr(frame, name='dpqm_noise',
                               long_name='pixel-quality map (noise average)'),
                    data_to_xr(frame, name='dpqm_noise_var',
                               long_name='pixel-quality map (noise variance)')])

    plot.draw_qhist(msm, data_sel=np.s_[11:228, 16:991], title='my histogram')


# --------------------------------------------------
def main():
    """
    main function
    """
    check_draw_signal = True
    check_draw_quality = True
    check_draw_qhist = True
    check_draw_trend = True
    check_draw_lplot = True

    # ---------- UNIT TEST: draw_signal ----------
    if check_draw_signal:
        plot = MONplot('mon_plot_draw_signal.pdf',
                       caption='Unit test of MONplot [draw_signal]')
        plot.set_institute('SRON')
        run_draw_signal(plot)
        plot.close()

    # ---------- UNIT TEST: draw_quality ----------
    if check_draw_quality:
        plot = MONplot('mon_plot_draw_quality.pdf',
                       caption='Unit test of MONplot [draw_quality]')
        plot.set_institute('SRON')
        run_draw_quality(plot)
        plot.close()

    # ---------- UNIT TEST: draw_qhist ----------
    if check_draw_qhist:
        plot = MONplot('mon_plot_draw_qhist.pdf',
                       caption='Unit test of MONplot [draw_qhist]')
        plot.set_institute('SRON')
        run_draw_qhist(plot)
        plot.close()

    # ---------- UNIT TEST: draw_trend ----------
    if check_draw_trend:
        plot = MONplot('mon_plot_draw_trend.pdf',
                       caption='Unit test of MONplot [draw_trend]')
        plot.set_institute('SRON')
        run_draw_trend(plot)
        plot.close()

    # ---------- UNIT TEST: draw_lplot ----------
    if check_draw_lplot:
        plot = MONplot('mon_plot_draw_lplot.pdf',
                       caption='Unit test of MONplot [draw_lplot]')
        plot.set_institute('SRON')
        run_draw_lplot(plot)
        plot.close()


# - main code --------------------------------------
if __name__ == '__main__':
    main()
