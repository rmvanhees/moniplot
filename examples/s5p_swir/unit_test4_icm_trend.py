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
    icm_mon_db = 'offsdark/official/mon_offset_official'

    orbit_bgn = 2815
    orbit = 21947

    stats_list = []
    with ICMmonStability(data_dir + icm_mon_db) as mon:
        stats_list.append(trend_stats(mon, orbit_bgn, orbit, ['noise_plain']))
        stats_list.append(trend_stats(mon, orbit_bgn, orbit, ['noise_b1']))
        stats_list.append(trend_stats(mon, orbit_bgn, orbit, ['read_noise']))

    # define fig_info
    orbit_mn = int(stats_list[0].coords['orbit'][0])
    orbit_mn -= 15
    orbit_mx = int(stats_list[0].coords['orbit'][-1])
    with ICMmonSQL(data_dir + icm_mon_db) as mon:
        coverage = mon.get_coverage(orbit_range=[orbit_mn, orbit_mx])

    fig_info_in = FIGinfo()
    fig_info_in.add('orbits', [orbit_mn, orbit_mx])
    fig_info_in.add('coverage',
                    (coverage[0].strftime('%Y-%m-%d'),
                     coverage[1].strftime('%Y-%m-%d')), fmt='{} / {}')

    plot = MONplot('test_icm_trend.pdf', institude='SRON')
    title = 'Tropomi SWIR stability (trends)'
    for stats in stats_list:
        hk_keys = ('detector_temp', 'grating_temp', 'imager_temp', 'obm_temp')
        with ENGmon(data_dir + eng_mon_db) as mon:
            eng_hk = trend_eng(mon, orbit_mn, orbit_mx, hk_keys)

        # remove data when detector temperature is too high
        indx = np.where((eng_hk['detector_temp'].values['err1'] > 139.95)
                        & (eng_hk['detector_temp'].values['err2'] < 140.025)
                        & (eng_hk['grating_temp'].values['mean'] < 202.1))[0]
        eng_hk = eng_hk.isel(orbit=indx, drop=True)

        sub_title = f'trend of median normalized ... & temperatures'
        plot.draw_trend(stats, eng_hk, fig_info=fig_info_in.copy(),
                        title=title, sub_title=sub_title)
    plot.close()

if __name__ == '__main__':
    main()
