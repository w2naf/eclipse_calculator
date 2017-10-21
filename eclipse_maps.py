#!/usr/bin/env python3
import os
import datetime
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

import multiprocessing

import eclipse_calc

def plot_eclipse(date,output_dir='output'):
    map_prm = {}
    map_prm['llcrnrlon'] = -130.
    map_prm['llcrnrlat'] =   20.
    map_prm['urcrnrlon'] =  -60.
    map_prm['urcrnrlat'] =   55.

    # Eclipse Magnitude
    gs_grid     = eclipse_calc.locator.gridsquare_grid(precision=4).flatten()
    ll_grid     = eclipse_calc.locator.gridsquare2latlon(gs_grid)
    lats,lons   = ll_grid
    ecl_mags    = np.array([eclipse_calc.eclipse_mag(lat,lon,date) for lat,lon in zip(lats,lons)])
    ecl_cmap    = mpl.cm.gray_r


    fig         = plt.figure(figsize=(12,10))
    ax          = fig.add_subplot(111)
    hmap        = eclipse_calc.maps.HamMap(date,date,ax,**map_prm)
    hmap.overlay_gridsquares(label_precision=0,major_style={'color':'0.8','dashes':[1,1]})
    hmap.overlay_gridsquare_data(gs_grid,ecl_mags,plot_cbar=False,vmin=0.10,vmax=1.15,zorder=5,cmap=ecl_cmap)

    fig.tight_layout()

    date_str   = date.strftime('%Y%m%d.%H%M')
    fname       = '{!s}_eclipse.png'.format(date_str)
    fpath       = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

    return fpath

if __name__ == '__main__':
    output_dir  = 'output'
    eclipse_calc.gen_lib.clear_dir(output_dir,php=True)

    sDate   = datetime.datetime(2017,8,21,14)
    eDate   = datetime.datetime(2017,8,21,22)
    dt      = datetime.timedelta(minutes=5)

    dates   = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1]+dt)

#    for date in dates:
#        fpath = plot_eclipse(date,output_dir)
#        print(fpath)

    with multiprocessing.Pool() as pool:
        pool.map(plot_eclipse,dates)
