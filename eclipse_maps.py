#!/usr/bin/env python3
import os
import datetime
import bz2
import glob
import tqdm

import multiprocessing

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade

import eclipse_calc

class ScriptTimer(object):
    def __init__(self):
        self.sTime  = datetime.datetime.now()
        self.script = os.path.basename(__file__)

        print('{!s} Running...'.format(self.script))
        print('   Started: {!s}'.format(self.sTime))
        print()

    def stop(self):
        self.eTime  = datetime.datetime.now()
        total       = self.eTime - self.sTime

        print()
        print('#--------------------------------------#')
        print('{!s} Finished...'.format(self.script))
        print('   Started:  {!s}'.format(self.sTime))
        print('   Finished: {!s}'.format(self.eTime))
        print('   Duration: {!s}'.format(total))
        print('#--------------------------------------#')
        print()

def location_dict(dlat,dlon,height,lat_0=-90.,lat_1=90,lon_0=-180.,lon_1=180):
    """
    Calculates a mesh grid of latitudes and longitudes at the center points of the
    cells. Returns a flattened dataframe with these values plus a specified height.
    """
    dd              = {}

    lats            = np.arange(lat_0+dlat/2.,lat_1+dlat/2.,dlat)
    lons            = np.arange(lon_0+dlon/2.,lon_1+dlon/2.,dlon)

    LATS, LONS      = np.meshgrid(lats,lons,indexing='ij')

    dd['lat']       = LATS.flatten()
    dd['lon']       = LONS.flatten()
    dd['height']    = np.ones(dd['lat'].shape)*height
    return dd

def plot_eclipse_dict(run_dict):
    return plot_eclipse(**run_dict)

def calc_obscuration_df(date,lat,lon,height,csv_path=None,**kw_args):
    # Set up data dictionary.
    dd              = {}
    dd['lat']       = lat
    dd['lon']       = lon
    dd['height']    = height

    # Eclipse Magnitude
    dates       = np.array(len(dd['lat'])*[date])
    dd['obsc']  = eclipse_calc.calculate_obscuration(dates,dd['lat'],dd['lon'],height=dd['height'])

    # Store into dataframe.
    df          = pd.DataFrame(dd)

    # Save CSV Datafile.
    if csv_path:
        hdr = []
        hdr.append('# Solar Eclipse Obscuration file for {!s}\n'.format(date))
        hdr.append(','.join(df.columns))
        hdr = '\n'.join(hdr)
        df.to_csv(csv_path,index=False,header=hdr)

    return df


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

    csv_path    = fpath+'.csv.bz2'
    df = calc_obscuration_df(date,csv_path=csv_path,**loc_dict)

    # Calculate vectors of center lats and lons.
    center_lats = np.sort(df['lat'].unique())
    center_lons = np.sort(df['lon'].unique())

    # Find the lat/lon step size.
    dlat = center_lats[1] - center_lats[0]
    dlon = center_lons[1] - center_lons[0]

    # Calculate vectors of boundary lats and lons.
    lat_0   = center_lats.min() - dlat/2.
    lat_1   = center_lats.max() + dlat/2.
    lats    = np.arange(lat_0,lat_1+dlat,dlat)

    lon_0 = center_lons.min() - dlon/2.
    lon_1 = center_lons.max() + dlon/2.
    lons    = np.arange(lon_0,lon_1+dlon,dlon)

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
    hmap.plot_nightshade()

    cshape      = (len(center_lats),len(center_lons))
    obsc_arr    = df['obsc'].to_numpy().reshape(cshape)
    pcoll       = ax.pcolormesh(lons,lats,obsc_arr,vmin=vmin,vmax=vmax,cmap=cmap,zorder=5)

    cbar_shrink = 0.5
    cbar_label  = 'Obscuration'
    cbar        = fig.colorbar(pcoll,label=cbar_label,shrink=cbar_shrink)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    title       = '{!s} Height: {!s} km'.format(date.strftime('%d %b %Y %H%M UT'),height/1000.)
    fontdict    = {'size':'x-large','weight':'bold'}
    hmap.ax.text(0.5,1.075,title,fontdict=fontdict,transform=ax.transAxes,ha='center')
    fig.tight_layout()
    fig.savefig(fpath+'.png',bbox_inches='tight')

    plt.close(fig)
    return fpath

if __name__ == '__main__':
    timer = ScriptTimer()

    multiproc   = False
    ncpus       = multiprocessing.cpu_count()

    output_dir  = 'output'
    eclipse_calc.gen_lib.clear_dir(output_dir,php=True)

##    # 21 August 2017 Total Solar Eclipse
##    sDate   = datetime.datetime(2017,8,21,14)
##    eDate   = datetime.datetime(2017,8,21,22)
#
##    # 14 October 2023 Total Solar Eclipse
##    sDate   = datetime.datetime(2023,10,14,14)
##    eDate   = datetime.datetime(2023,10,14,21)

#    # 8 April 2024 Total Solar Eclipse
#    sDate   = datetime.datetime(2024,4,8,15)
#    eDate   = datetime.datetime(2024,4,8,21)

    sDate   = datetime.datetime(2024,4,8,17)
    eDate   = datetime.datetime(2024,4,8,18)

    dt      = datetime.timedelta(minutes=5)

    dlat        = 10.
    dlon        = 10.

#    dlat        = 0.5
#    dlon        = 0.5

    height      = 300e3

    loc_dict    = location_dict(dlat,dlon,height)

    run_list    = []
    cDate       = sDate
    while cDate < eDate:
        tmp = {}
        tmp['date']         = cDate
        tmp['loc_dict']     = loc_dict
        tmp['output_dir']   = output_dir
        run_list.append(tmp)
        cDate   += dt

    ## Calculate Eclipse Data and Plot
    if multiproc:
        with multiprocessing.Pool(ncpus) as pool:
            pool.map(plot_eclipse_dict,run_list)
    else:
        # Single Processor
        for run_dict in run_list:
            fpath = plot_eclipse_dict(run_dict)


    timer.stop()
    import ipdb; ipdb.set_trace()


