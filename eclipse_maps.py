#!/usr/bin/env python3
import os
import datetime
from collections import OrderedDict
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

import eclipse_calc

def plot_eclipse(date,precision=2,height=300e3,region='world',
        cmap=mpl.cm.gray_r,output_dir='output'):
    """
    region: 'us' or 'world"
    height: [km]
    """

    map_prm = {}
    if region == 'world':
        # Map boundaries for the world
        map_prm['llcrnrlon'] = -180.
        map_prm['llcrnrlat'] = -90
        map_prm['urcrnrlon'] = 180.
        map_prm['urcrnrlat'] = 90.
    else:
        # Map boundaries for the United States
        map_prm['llcrnrlon'] = -130.
        map_prm['llcrnrlat'] =   20.
        map_prm['urcrnrlon'] =  -60.
        map_prm['urcrnrlat'] =   55.

    # Eclipse Magnitude
    gs_grid     = eclipse_calc.locator.gridsquare_grid(precision=precision).flatten()
    ll_grid     = eclipse_calc.locator.gridsquare2latlon(gs_grid)
    lats,lons   = ll_grid

    ecl_obsc    = np.array([eclipse_calc.calculate_obscuration(date,lat,lon,height) for lat,lon in zip(lats,lons)])


    # Store into dataframe.
    dd          = OrderedDict()
    dd['grid']  = gs_grid
    dd['lat']   = lats
    dd['lon']   = lons
    dd['obsc']  = ecl_obsc

    df          = pd.DataFrame(dd)
    df          = df.set_index('grid')

    # Define output paths.
    date_str    = date.strftime('%Y%m%d.%H%M')
    fname       = '{!s}_{!s}km_eclipseObscuration'.format(date_str,height/1000.)
    fpath       = os.path.join(output_dir,fname)

    # Save CSV Datafile.
    csv_path    = fpath+'.csv'
    with open(csv_path,'w') as fl:
        fl.write('# Solar Eclipse Obscuration file for {!s}\n'.format(date))
    df.to_csv(csv_path,mode='a')

    # Plot data.
    fig         = plt.figure(figsize=(12,10))
    ax          = fig.add_subplot(111)
    hmap        = eclipse_calc.maps.HamMap(date,date,ax,**map_prm)
    hmap.overlay_gridsquares(label_precision=0,major_style={'color':'0.8','dashes':[1,1]})
    hmap.overlay_gridsquare_data(dd['grid'],dd['obsc'],vmin=0.10,vmax=1.15,zorder=5,cmap=cmap)
    fig.tight_layout()
    fig.savefig(fpath+'.png',bbox_inches='tight')
    plt.close(fig)

    return fpath

if __name__ == '__main__':
    output_dir  = 'output'
    eclipse_calc.gen_lib.clear_dir(output_dir,php=True)

#    sDate   = datetime.datetime(2017,8,21,14)
#    eDate   = datetime.datetime(2017,8,21,22)
    sDate   = datetime.datetime(2017,8,21,18)
    eDate   = datetime.datetime(2017,8,21,19)
    dt      = datetime.timedelta(minutes=5)

    dates   = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1]+dt)

    for date in dates:
        fpath = plot_eclipse(date,output_dir=output_dir)
        print(fpath)

#    import multiprocessing
#    with multiprocessing.Pool() as pool:
#        pool.map(plot_eclipse,dates)
