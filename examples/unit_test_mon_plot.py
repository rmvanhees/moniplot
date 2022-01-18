"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Performs unit-tests on MONplot methods: draw_signal, draw_quality,
   draw_trend and draw_lplot ## draw_cmp_images

Copyright (c) 2020-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from pys5p import swir_region

from moniplot.image_to_xarray import data_to_xr
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

    region = ~swir_region.mask()
    plot.draw_quality(data, exclude_region=region, title='no reference')
    plot.draw_quality(data, ref_data=ref_data, exclude_region=region,
                      title='with reference')


def run_draw_cmp_swir(plot):
    """
    Run unit tests on MONplot::draw_cmp_swir
    """
    print('Run unit tests on MONplot::draw_cmp_swir')

    msm = get_test_data(data_sel=np.s_[500-128:500+128, :], error=.1)
    msm_ref = get_test_data(data_sel=np.s_[500-128:500+128, :], error=.025)

    plot.draw_cmp_swir(msm, msm_ref.values, title='test image')
    plot.draw_cmp_swir(msm, msm_ref.values, add_residual=False,
                       title='test image')
    plot.draw_cmp_swir(msm, msm_ref.values, add_model=False,
                       title='test image')


def run_draw_trend(plot):
    """
    Run unit tests on MONplot::draw_trend
    """
    print('Run unit tests on MONplot::draw_trend')
    xx = np.arange(200) / 100
    hk_params = [
        ('detector_temp', 'SWIR detector temperature', 'K', np.float32),
        ('grating_temp', 'SWIR grating temperature', 'K', np.float32),
        ('obm_temp', 'SWIR OBM temperature', 'K', np.float32)]
    hk_dtype = np.dtype(
        [(parm[0], parm[3]) for parm in hk_params])
    hk_min = np.zeros(200, dtype=hk_dtype)
    hk_avg = np.zeros(200, dtype=hk_dtype)
    hk_max = np.zeros(200, dtype=hk_dtype)
    data = 140. + (100 - np.arange(200)) / 1000
    hk_min['detector_temp'][:] = data - .0125
    hk_avg['detector_temp'][:] = data
    hk_max['detector_temp'][:] = data + .0075
    data = 202.1 + (100 - np.arange(200)) / 1000
    hk_min['grating_temp'][:] = data - .15
    hk_avg['grating_temp'][:] = data
    hk_max['grating_temp'][:] = data + .175
    data = 208.2 + (100 - np.arange(200)) / 1000
    hk_min['obm_temp'][:] = data - .15
    hk_avg['obm_temp'][:] = data
    hk_max['obm_temp'][:] = data + .175
    units = [parm[2] for parm in hk_params]
    long_name = [parm[1] for parm in hk_params]

    msm_mean = data_to_xr(hk_avg, dims=['orbit'], name='hk_mean',
                          long_name=long_name, units=units)
    msm_range = data_to_xr(np.stack([hk_min, hk_max], axis=1),
                           dims=['orbit', 'range'], name='hk_range',
                           long_name=long_name, units=units)
    hk_ds = xr.merge([msm_mean, msm_range])

    msm1 = data_to_xr(np.sin(xx * np.pi), dims=['orbit'])
    msm2 = data_to_xr(np.cos(xx * np.pi), dims=['orbit'])

    plot.draw_trend(msm1, title='one dataset, no house-keeping')
    plot.draw_trend(msm1, msm2=msm2,
                    title='two datasets, no house-keeping')
    hk_keys = [parm[0] for parm in hk_params]
    plot.draw_trend(msm1, msm2=msm2,
                    hk_data=hk_ds, hk_keys=hk_keys[0:2],
                    title='two datasets and house-keeping')
    plot.draw_trend(msm1, msm2=msm2,
                    hk_data=hk_ds, hk_keys=hk_keys,
                    title='two datasets and house-keeping')


def run_draw_lplot(plot):
    """
    Run unit tests on MONplot::draw_lplot
    """
    print('Run unit tests on MONplot::draw_lplot')
    xx = np.arange(200) / 100
    plot.draw_lplot(xx, np.sin(xx * np.pi), color=0,
                    label='sinus', marker='o', linestyle='-')
    plot.draw_lplot(xx, np.cos(xx * np.pi), color=1,
                    label='cosinus', marker='o', linestyle='-')
    plot.draw_lplot(None, None, ylim=[-1.05, 1.05],
                    xlabel='x-axis [Pi]', ylabel='y-axis',
                    title='draw_lplot [no time_axis]')

    xx = np.arange(500) / 100
    plot.draw_lplot(xx, np.sin(xx * np.pi), color=0,
                    label='sinus', marker='o', linestyle='-')
    plot.draw_lplot(xx, np.cos(xx * np.pi), color=1,
                    label='cosinus', marker='o', linestyle='-')
    plot.draw_lplot(None, None, ylim=[-1.05, 1.05],
                    xlabel='x-axis [Pi]', ylabel='y-axis',
                    title='draw_lplot [no time_axis]')

    customdate = datetime(2016, 1, 1, 13, 0, 0)
    yy = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    xx = [customdate + timedelta(hours=i, minutes=4*i)
          for i in range(len(yy))]
    plot.draw_lplot(xx, yy, color=0, label='mydata',
                    marker='o', linestyle='-')
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

    region = ~swir_region.mask()
    plot.draw_qhist(msm, exclude_region=region, title='my histogram')


# --------------------------------------------------
def main():
    """
    main function
    """
    check_draw_signal = True
    check_draw_cmp_images = False
    check_draw_quality = True
    check_draw_qhist = True
    check_draw_trend = False
    check_draw_lplot = True

    # ---------- UNIT TEST: draw_signal ----------
    if check_draw_signal:
        plot = MONplot('mon_plot_draw_signal.pdf',
                       caption='Unit test of MONplot [draw_signal]')
        plot.set_institute('SRON')
        run_draw_signal(plot)
        plot.close()

    # ---------- UNIT TEST: draw_cmp_images ----------
    if check_draw_cmp_images:
        plot = MONplot('mon_plot_draw_cmp_images.pdf',
                       caption='Unit test of MONplot [draw_cmp_images]')
        plot.set_institute('SRON')
        run_draw_cmp_swir(plot)
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
