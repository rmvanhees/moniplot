from datetime import datetime

import numpy as np
import xarray as xr

from pyS5pMon.db.eng_mon_h5 import ENGmon
from pyS5pMon.db.s5p_mon_sql import ENGmonSQL
from pyS5pMon.db.eng_mon_trend import trend_eng, trend_eng_day
from pyS5pMon.db.eng_prod_db import ENGdb

from moniplot.lib.fig_info import FIGinfo
from moniplot.mon_plot import MONplot


#--------------------------------------------------
def main():
    """
    main function
    """
    l1b_eng_db = '/data/richardh/Tropomi/share/db/sron_s5p_eng.db'
    data_dir = '/nfs/Tropomi/ical/monitoring/1.2/housekeeping/swir/'
    #data_dir = '/Users/richardh/'
    data_dir = '/data/richardh/Tropomi/ical/monitoring/1.2/housekeeping/swir/'
    eng_mon_db = 'mon_housekeeping_swir'

    proc_date = datetime(2022, 1, 7)
    orbit_range = [21472, 21956]
    proc_date = datetime(2021, 11, 20)
    orbit_range = [20791, 21275]
    proc_date = datetime(2021, 11, 21)
    orbit_range = [20806, 21290]
    proc_date = datetime(2021, 11, 22)
    orbit_range = [20820, 21304]

    with ENGdb(l1b_eng_db) as dbase:
        dbase.select(date=proc_date.strftime('%y%m%d'))
        orbits = dbase.orbits()
    orbit_day_range = [orbits[0], orbits[-1]]
    with ENGmonSQL(data_dir + eng_mon_db) as mon_sql:
        coverage = mon_sql.get_coverage(orbit_range=orbit_day_range)
    fig_info_day = FIGinfo()
    fig_info_day.add('orbits', orbit_day_range)
    fig_info_day.add('coverage',
                     (coverage[0].strftime('%Y-%m-%d'),
                      coverage[1].strftime('%Y-%m-%d')),
                     fmt='{} / {}')
    
    with ENGmonSQL(data_dir + eng_mon_db) as mon_sql:
        coverage = mon_sql.get_coverage(orbit_range=orbit_range)
    fig_info_orbit = FIGinfo()
    fig_info_orbit.add('orbits', orbit_range)
    fig_info_orbit.add('coverage',
                       (coverage[0].strftime('%Y-%m-%d'),
                        coverage[1].strftime('%Y-%m-%d')),
                       fmt='{} / {}')

    plot = MONplot('test_eng_trend.pdf',
                   caption='Tropomi SWIR housekeeping data')
    plot.set_institute('SRON')

    title = 'Temperature of sensors relevant for SWIR'
    hk_keys=('detector_temp', 'grating_temp',
             'imager_temp', 'obm_temp', 'calib_unit_temp')
    with ENGmon(data_dir + eng_mon_db) as mon:
        print(mon.get_attr('title'))
        msm_hk = trend_eng(mon, orbit_range[0], orbit_range[1], hk_keys)
        msm_hk_day = trend_eng_day(mon, proc_date, hk_keys)
    fig_info = fig_info_day.copy()
    plot.draw_trend(hk_xds=msm_hk_day, fig_info=fig_info, title=title)
    fig_info = fig_info_orbit.copy()
    plot.draw_trend(hk_xds=msm_hk, vrange_last_orbits=30,
                    fig_info=fig_info, title=title)

    title = 'Current and duty cycle of 3 heaters relevant for SWIR'
    hk_keys=('detector_heater',
             'obm_heater_cycle', 'fee_box_heater_cycle',
             'obm_heater', 'fee_box_heater')
    with ENGmon(data_dir + eng_mon_db) as mon:
        print(mon.get_attr('title'))
        msm_hk = trend_eng(mon, orbit_range[0], orbit_range[1], hk_keys)
        msm_hk_day = trend_eng_day(mon, proc_date, hk_keys)
    fig_info = fig_info_day.copy()
    plot.draw_trend(hk_xds=msm_hk_day, fig_info=fig_info, title=title)
    fig_info = fig_info_orbit.copy()
    plot.draw_trend(hk_xds=msm_hk, vrange_last_orbits=30,
                    fig_info=fig_info, title=title)

    title = 'Temperature of sensors at the SWIR front-end electronics'
    hk_keys=('fee_inner_temp', 'fee_board_temp',
             'fee_ref_volt_temp', 'fee_video_amp_temp',
             'fee_video_adc_temp')
    with ENGmon(data_dir + eng_mon_db) as mon:
        print(mon.get_attr('title'))
        msm_hk = trend_eng(mon, orbit_range[0], orbit_range[1], hk_keys)
        msm_hk_day = trend_eng_day(mon, proc_date, hk_keys)
    fig_info = fig_info_day.copy()
    plot.draw_trend(hk_xds=msm_hk_day, fig_info=fig_info, title=title)
    fig_info = fig_info_orbit.copy()
    plot.draw_trend(hk_xds=msm_hk, vrange_last_orbits=30,
                    fig_info=fig_info, title=title)
    plot.close()

if __name__ == '__main__':
    main()
