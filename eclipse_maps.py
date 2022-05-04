#!/usr/bin/env python3
import os
import datetime
from collections import OrderedDict
import multiprocessing

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

import cartopy.crs as ccrs

import eclipse_calc

def location_dict(precision=2,height=0.):
    gs_grid     = eclipse_calc.locator.gridsquare_grid(precision=precision).flatten()
    ll_grid     = eclipse_calc.locator.gridsquare2latlon(gs_grid)
    lats,lons   = ll_grid

    dd              = OrderedDict()
    dd['grid']      = gs_grid
    dd['lat']       = lats
    dd['lon']       = lons
    dd['height']    = np.ones(lats.shape)*height
    return dd

def plot_eclipse_dict(run_dict):
    return plot_eclipse(**run_dict)

def plot_eclipse(date,loc_dict,region='world',cmap=mpl.cm.gray_r,output_dir='output'):
    """
    region: 'us' or 'world"
    height: [km]
    """
    # Define output paths.
    date_str    = date.strftime('%Y%m%d.%H%M')
    fname       = '{!s}_{!s}km_eclipseObscuration'.format(date_str,height/1000.)
    fpath       = os.path.join(output_dir,fname)
    print('Processing {!s}...'.format(fpath))

    # Set up data dictionary.
    dd          = OrderedDict()
    dd['grid']  = loc_dict['grid']
    dd['lat']   = loc_dict['lat']
    dd['lon']   = loc_dict['lon']
    dd['height']= loc_dict['height']
#    locs        = loc_dict['loc']

    # Eclipse Magnitude
#    dd['obsc']  = np.array([eclipse_calc.calculate_obscuration(date,loc=loc) for loc in locs])
    dates       = np.array(len(dd['lat'])*[date])
    dd['obsc']  = eclipse_calc.calculate_obscuration(dates,dd['lat'],dd['lon'],height=dd['height'])
#    dd['obsc']  = dd['lat'] # Useful for debugging plotting.

    # Store into dataframe.
    df          = pd.DataFrame(dd)
    df          = df.set_index('grid')

    # Save CSV Datafile.
    csv_path    = fpath+'.csv'
    with open(csv_path,'w') as fl:
        fl.write('# Solar Eclipse Obscuration file for {!s}\n'.format(date))
    df.to_csv(csv_path,mode='a')

    # Plot data.
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

    vmin        = 0.
    vmax        = 1.
    cbar_ticks  = np.arange(0,1.1,0.1)

    fig         = plt.figure(figsize=(12,10))
    crs         = ccrs.PlateCarree()
    ax          = fig.add_subplot(111,projection=ccrs.PlateCarree())
    hmap        = eclipse_calc.maps.HamMap(date,date,ax,show_title=False,**map_prm)
    hmap.overlay_gridsquares(label_precision=0,major_style={'color':'0.8','linestyle':'--'})
    hmap.overlay_gridsquare_data(dd['grid'],dd['obsc'],vmin=vmin,vmax=vmax,cbar_ticks=cbar_ticks,
                zorder=5,cmap=cmap,cbar_shrink=0.5,cbar_label='Obscuration')

    title       = '{!s} Height: {!s} km'.format(date.strftime('%d %b %Y %H%M UT'),height/1000.)
    fontdict    = {'size':'x-large','weight':'bold'}
    hmap.ax.text(0.5,1.075,title,fontdict=fontdict,transform=ax.transAxes,ha='center')
    fig.tight_layout()
    fig.savefig(fpath+'.png',bbox_inches='tight')

    plt.close(fig)

    return fpath

if __name__ == '__main__':
    output_dir  = 'output'
    eclipse_calc.gen_lib.clear_dir(output_dir,php=True)

    # 21 August 2017 Total Solar Eclipse
    sDate   = datetime.datetime(2017,8,21,14)
    eDate   = datetime.datetime(2017,8,21,22)

#    # 14 October 2023 Total Solar Eclipse
#    sDate   = datetime.datetime(2023,10,14,14)
#    eDate   = datetime.datetime(2023,10,14,21)

#    # 8 April 2024 Total Solar Eclipse
#    sDate   = datetime.datetime(2024,4,8,15)
#    eDate   = datetime.datetime(2024,4,8,21)

    dt      = datetime.timedelta(minutes=5)

    precision   = 4
    height      = 300e3

    loc_dict    = location_dict(precision=precision,height=height)

    run_list    = []
    cDate       = sDate
    while cDate < eDate:
        tmp = OrderedDict()
        tmp['date']         = cDate
        tmp['loc_dict']     = loc_dict
        tmp['output_dir']   = output_dir
        run_list.append(tmp)
        cDate   += dt

    # Single Processor
#    for run_dict in run_list:
#        fpath = plot_eclipse_dict(run_dict)

    with multiprocessing.Pool() as pool:
        pool.map(plot_eclipse_dict,run_list)
