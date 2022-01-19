"""
Write here the module documentation
"""

import numpy as np

from pyS5pMon.db.eng_mon_h5 import ENGmon
from pyS5pMon.db.eng_mon_trend import trend_eng
from pyS5pMon.db.icm_mon_db import ICMmonStability
from pyS5pMon.db.s5p_mon_sql import ICMmonSQL
from pyS5pMon.db.icm_mon_trend import trend_stats

from moniplot.lib.fig_info import FIGinfo
from moniplot.mon_plot import MONplot


#--------------------------------------------------
def main():
    """
    main function
    """
    #data_dir = '/Users/richardh/'
    #data_dir = '/nfs/Tropomi/ical/monitoring/1.2/stability/swir/'
    data_dir = '/data/richardh/Tropomi/ical/monitoring/1.2/'
    eng_mon_db = 'housekeeping/swir/mon_housekeeping_swir'
    icm_mon_db = 'stability/swir/mon_stability_qvd1'

    orbit_bgn = 2815
    orbit = 21947

    stats_qvd = None
    with ICMmonStability(data_dir + icm_mon_db) as mon:
        light_source = str(mon.get_attr('light_source')).upper()
        if light_source[:3] == 'QVD':
            stats_qvd = trend_stats(mon, orbit_bgn, orbit, ('value', 'shift'))

        stats_bgr = trend_stats(mon, orbit_bgn, orbit, ('value', 'bgr_value'))

    # define fig_info
    orbit_mn = int(stats_bgr.coords['orbit'][0])
    orbit_mn -= 15
    orbit_mx = int(stats_bgr.coords['orbit'][-1])
    with ICMmonSQL(data_dir + icm_mon_db) as mon:
        coverage = mon.get_coverage(orbit_range=[orbit_mn, orbit_mx])

    fig_info = FIGinfo()
    fig_info.add('orbits', [orbit_mn, orbit_mx])
    fig_info.add('coverage',
                 (coverage[0].strftime('%Y-%m-%d'),
                  coverage[1].strftime('%Y-%m-%d')), fmt='{} / {}')

    # what to do with the not used DataArrays in stats?
    hk_keys = ('detector_temp', 'grating_temp', 'obm_temp')
    with ENGmon(data_dir + eng_mon_db) as mon:
        eng_hk = trend_eng(mon, orbit_mn, orbit_mx, hk_keys)

    # remove data when detector temperature is too high
    if 1 == 0:
        indx = np.where((eng_hk['detector_temp'].values['err1'] > 139.95)
                        & (eng_hk['detector_temp'].values['err2'] < 140.025)
                        & (eng_hk['grating_temp'].values['mean'] < 202.1))[0]
        eng_hk = eng_hk.isel(orbit=indx, drop=True)

    plot = MONplot('test_icm_trend.pdf',
                   caption='Tropomi SWIR stability (trends)')
    plot.set_institute('SRON')
    title = f'trend of median normalized {light_source} & temperatures'
    if stats_qvd is not None:
        plot.draw_trend(stats_qvd, eng_hk, fig_info=fig_info, title=title,
                        vperc=[1, 99])
    title = f'trend of median {light_source} back-ground & temperatures'
    plot.draw_trend(stats_bgr, eng_hk, fig_info=fig_info, title=title)
    plot.draw_trend(stats_bgr, eng_hk, fig_info=fig_info, title=title,
                    vperc=[1, 99])
    plot.close()

if __name__ == '__main__':
    main()
