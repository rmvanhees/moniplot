"""
Write here the module documentation
"""

import numpy as np

from pys5p import swir_region

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
    icm_mon_db = 'quality/beta-nominal/mon_quality_beta-nominal'

    orbit_bgn = 1867
    orbit = 21952

    with ICMmonStability(data_dir + icm_mon_db) as mon:
        stats = trend_stats(mon, orbit_bgn, orbit, ['bad', 'worst'])
        stats['bad'].attrs['legend'] = 'bad (quality < 0.8)'
        stats['worst'].attrs['legend'] = 'worst (quality < 0.1)'

    # define fig_info
    orbit_mn = int(stats.coords['orbit'][0])
    orbit_mn -= 15
    orbit_mx = int(stats.coords['orbit'][-1])
    with ICMmonSQL(data_dir + icm_mon_db) as mon:
        coverage = mon.get_coverage(orbit_range=[orbit_mn, orbit_mx])

    fig_info = FIGinfo()
    fig_info.add('orbits', [orbit_mn, orbit_mx])
    fig_info.add('coverage',
                 (coverage[0].strftime('%Y-%m-%d'),
                  coverage[1].strftime('%Y-%m-%d')), fmt='{} / {}')

    hk_keys = ('detector_temp', 'grating_temp', 'obm_temp')
    with ENGmon(data_dir + eng_mon_db) as mon:
        eng_hk = trend_eng(mon, orbit_mn, orbit_mx, hk_keys)

    # remove data when detector temperature is too high
    # indx = np.where((eng_hk['detector_temp'].values['err1'] > 139.95)
    #                & (eng_hk['detector_temp'].values['err2'] < 140.025)
    #                & (eng_hk['grating_temp'].values['mean'] < 202.1))[0]
    # eng_hk = eng_hk.isel(orbit=indx, drop=True)

    plot = MONplot('test_icm_trend.pdf',
                   caption='Tropomi SWIR pixel quality (beta nominal)')
    plot.set_institute('SRON')
    title = f'trend of pixel quality & temperatures'
    plot.draw_trend(stats, eng_hk, fig_info=fig_info, title=title,
                    vperc=[1, 99])
    plot.close()

if __name__ == '__main__':
    main()
