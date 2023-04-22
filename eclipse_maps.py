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


def get_event_name(sDate,eDate,height,dlat,dlon):
    """
    Generate an event name that can be used to contain all data files.
    """

    sDate_str       = sDate.strftime('%Y%m%d.%H%M')
    eDate_str       = eDate.strftime('%Y%m%d.%H%M')

    nm = []
    nm.append(sDate_str)
    nm.append(eDate_str)
    nm.append('{!s}kmAlt'.format(int(height/1000.)))
    nm.append('{!s}dlat'.format(dlat))
    nm.append('{!s}dlon'.format(dlon))

    nm = '_'.join(nm)
    return nm


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

def calc_and_plot_eclipse(run_dict):
    """
    Top-level function to call both calc_obscuration_df() and plot_eclipse().

    This function takes in a single diction to allow it to work easily with
    multiprocessing.pool().
    """
    date        = run_dict['date']
    loc_dict    = run_dict['loc_dict']
    output_dir  = run_dict['output_dir']
    height      = loc_dict['height'][0]

    # Define output paths.
    date_str    = date.strftime('%Y%m%d.%H%M')
    fname       = '{!s}_{!s}km_eclipseObscuration'.format(date_str,height/1000.)
    fpath       = os.path.join(output_dir,fname)
    print('Processing {!s}...'.format(fpath))

    csv_path    = fpath+'.csv.bz2'
    df          = calc_obscuration_df(date,csv_path=csv_path,**loc_dict)

    fig_path    = fpath+'.png'
    png_path    = plot_eclipse(df,date,fig_path=fig_path,plot_min_sun_moon_sep=True)

    return png_path

def calc_obscuration_df(date,lat,lon,height,csv_path=None,**kw_args):
    # Set up data dictionary.
    dd              = {}
    dd['lat']       = lat
    dd['lon']       = lon
    dd['height']    = height

    # Eclipse Magnitude
    dates       = np.array(len(dd['lat'])*[date])
    result      =  eclipse_calc.calculate_obscuration(dates,dd['lat'],dd['lon'],height=dd['height'],return_dict=True)

    dd['obsc']              = result['obsc']
    dd['sun_moon_sep_deg']  = result['sun_moon_sep_deg']
    dd['solar_elev_deg']    = result['solar_elev_deg']

    # Store into dataframe.
    df          = pd.DataFrame(dd)

    # Save CSV Datafile.
    if csv_path:
        hdr = []
        hdr.append('# Solar Eclipse Obscuration file for {!s}'.format(date))
        hdr.append(','.join(df.columns))
        hdr = '\n'.join(hdr)
        hdr += '\n'

        with bz2.open(csv_path,'wt') as bzfl:
            bzfl.write(hdr)
            df.to_csv(bzfl,index=False,header=False)

    return df

def plot_eclipse(df,sDate,eDate=None,region='world',cmap=mpl.cm.gray_r,fig_path='output.png',
        min_obsc=0,max_obsc=1,nightshade=True,gridsquares=True,plot_min_sun_moon_sep=False,
        ecl_track_df=None):
    """
    df: Pandas DataFrame with Obscuration Data
    region: 'us' or 'world"
    height: [km]

    ecl_track_df: Pass a dataframe containing the track information to plot the eclipse track.
    """

    df = df.copy()

    if min_obsc > 0:
        tf = df['obsc'] < min_obsc
        df.loc[tf,'obsc'] = 0

    if max_obsc < 1:
        tf = df['obsc'] > max_obsc
        df.loc[tf,'obsc'] = 0

    if eDate is None:
        eDate = sDate

    height = df['height'].unique()
    n_heights = len(height)
    assert n_heights  == 1, f'One height expected, got: {n_heights}'
    height = height[0]

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

    # These if statements are to handle an error that can occur
    # when dlat or dlon are very small and you get the wrong number
    # of elements due to a small numerical error.
    if len(lats) > len(center_lats)+1:
        lats=lats[:len(center_lats)+1]

    if len(lons) > len(center_lons)+1:
        lons=lons[:len(center_lons)+1]

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

#    fig         = plt.figure(figsize=(12,10))
    fig         = plt.figure(figsize=(16,14))
    crs         = ccrs.PlateCarree()
    ax          = fig.add_subplot(111,projection=ccrs.PlateCarree())
    hmap        = eclipse_calc.maps.HamMap(sDate,eDate,ax,show_title=False,**map_prm)

    if gridsquares:
        hmap.overlay_gridsquares(label_precision=0,major_style={'color':'0.8','linestyle':'--'})

    if nightshade:
        hmap.plot_nightshade(zorder=500)

    cshape      = (len(center_lats),len(center_lons))
    obsc_arr    = df['obsc'].to_numpy().reshape(cshape)
    pcoll       = ax.pcolormesh(lons,lats,obsc_arr,vmin=vmin,vmax=vmax,cmap=cmap,zorder=5)

    cbar_shrink = 0.5
    cbar_label  = 'Obscuration'
    cbar        = fig.colorbar(pcoll,label=cbar_label,shrink=cbar_shrink)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    if plot_min_sun_moon_sep:
        ecl_ctr = find_eclipse_center(df)
        if ecl_ctr:
            ax.scatter(ecl_ctr['lon'],ecl_ctr['lat'],marker='o',s=50,color='gray',zorder=1000,ec='k')

    if ecl_track_df is not None:
        for inx,(rinx,row_0) in enumerate(ecl_track_df.iterrows()):
            lat_0 = row_0['lat']
            lon_0 = row_0['lon']

            tp = {}
            tp['bbox']          = dict(boxstyle='round', facecolor='white', alpha=0.75)
            tp['fontweight']    = 'bold'
            tp['fontsize']      = 8
            tp['zorder']        = 975
            tp['va']            = 'top'

            if inx == 0:
                ax.text(lon_0,lat_0,'{!s}'.format(rinx.strftime('%H%M UT')),**tp)
            if inx == len(ecl_track_df)-1:
                ax.text(lon_0,lat_0,'{!s}'.format(rinx.strftime('%H%M UT')),**tp)
                continue

            row_1 = ecl_track_df.iloc[inx+1]
            lat_1 = row_1['lat']
            lon_1 = row_1['lon']

            ax.annotate('', xy=(lon_1,lat_1), xytext=(lon_0,lat_0),zorder=950,
                    xycoords='data', size=10,
                    arrowprops=dict(facecolor='red', ec = 'none', arrowstyle="simple",
                connectionstyle="arc3,rad=-0.1"))

#            ax.annotate('', xy=(lon_1,lat_1), xytext=(lon_0,lat_0),zorder=950,
#                    xycoords='data', size=20,
#                    arrowprops=dict(facecolor='red', ec = 'none', arrowstyle="fancy",
#                connectionstyle="arc3,rad=-0.3"))

    if sDate == eDate:
        date_str    = sDate.strftime('%d %b %Y %H%M UT')
        title       = '{!s} Height: {!s} km'.format(date_str,height/1000.)
    else:
        date_str    = sDate.strftime('%d %b %Y %H%M UT') + eDate.strftime(' - %d %b %Y %H%M UT')
        title       = '{!s}\nHeight: {!s} km'.format(date_str,height/1000.)

    if min_obsc > 0 or max_obsc < 1:
        title += '\n Obscuration Range Plotted: ({!s}, {!s})'.format(min_obsc,max_obsc)

    fontdict    = {'size':'x-large','weight':'bold'}
    hmap.ax.text(0.5,1.075,title,fontdict=fontdict,transform=ax.transAxes,ha='center')
    fig.tight_layout()
    fig.savefig(fig_path,bbox_inches='tight')

    plt.close(fig)
    return fig_path

def find_eclipse_center(df,min_solar_elev_deg=1):
    """
    Find the the row in an eclipse dataframe that has the minimum moon-sun separation.
    
    # Set some critera to only look at cells that are eclipsed. It is not enough to only
    # look at minimum sun-moon separation distance, because every df will have a 
    # min(sep_moon_sep_deg), even if there is no eclipse happening. Solar eclipses will
    # have a maximum sun-moon separation of 33.4 arcminutes.
    # From https://astronomy.stackexchange.com/questions/28825/what-is-the-maximum-possible-separation-between-sun-and-moon-in-the-earth-sky-fo
    #   It can only be a solar eclipse if the Moon is touching the Sun.
    #
    #   In that case, their centers are at most (32.7 + 34.1) / 2 = 33.4 arcminutes apart.
    # Also, it is important to choose only cells that meet the eclipse separation AND are on the
    # dayside of the Earth. Therefore, we only look at rows with a minimum solar_elev_deg.
    # Setting min_solar_elev_deg = 1 seems to work well.
    """

    tf  = np.logical_and(df['solar_elev_deg'] >= (min_solar_elev_deg-1) ,df['sun_moon_sep_deg'] < (33.4/60.))
    if np.count_nonzero(tf) > 0:
        dft = df[tf]
        argmin = dft['sun_moon_sep_deg'].argmin()
        row = dft.iloc[argmin].to_dict()

        # Discard the edge cases
        if row['solar_elev_deg'] < min_solar_elev_deg:
            row = None
    else:
        row = None

    return row


def calc_max_obsc(in_csv_path,pattern='*.csv.bz2',out_csv_fname=None):
    """
    Read in a set of obscuration dataframe csv.bz2 files and create one file and
    dataframe that reports the maxium obscuration in each latitude-longitude cell.
    """
    fpaths = glob.glob(os.path.join(in_csv_path,pattern))
    fpaths.sort()

    # Load Eclipses
    df    = None
    dates = []
    files = []
    for fpath in tqdm.tqdm(fpaths,dynamic_ncols=True,desc='Computing Maximum Obscuration Map'):
        bname = os.path.basename(fpath)
        files.append(bname)

        date  = datetime.datetime.strptime(bname[:13],'%Y%m%d.%H%M')
        dates.append(date)
        cname = 'obsc-{!s}'.format(date.strftime('%H%M'))
        alt   = str(bname[14:17])

        dft   = pd.read_csv(fpath,comment='#')
        dft.rename(columns={'obsc':cname},inplace=True)
        if df is None:
            df = dft.copy()
        else:
            df[cname] = dft[cname]

    # Calculate max obscuration in each cell.
    tf = df.columns.str.contains('obsc-*')
    obsc_df = df.loc[:,tf]

    # Store max obscuration in new dataframe.
    max_obsc = df.loc[:,~tf].copy()
    max_obsc['obsc'] = obsc_df.max(1)

    sDate = min(dates)
    eDate = max(dates)

    # Save CSV Datafile.
    if out_csv_fname:
        hdr = []
        hdr.append('# Solar Eclipse Maximum Obscuration file for {!s} - {!s}'.format(sDate,eDate))
        hdr.append(','.join(max_obsc.columns))
        hdr = '\n'.join(hdr)
        hdr += '\n'

        with bz2.open(out_csv_fname,'wt') as bzfl:
            bzfl.write(hdr)
            max_obsc.to_csv(bzfl,index=False,header=False)

    return max_obsc

def compute_eclipse_track(in_csv_path,pattern='*.csv.bz2',out_csv_fname=None,track_geometry_path=None):
    """
    Read in a set of obscuration dataframe csv.bz2 files and create one file and
    dataframe that reports the center of the eclipse track for each time.

    track_geometry_path:    Path to save plots of Sun-Moon geometry used for making obscuration calculations.
                            In None, do not output these plots.

    """
    fpaths = glob.glob(os.path.join(in_csv_path,pattern))
    fpaths.sort()

    # Load Eclipses
    dates       = []
    files       = []
    ecl_track   = []
    for fpath in tqdm.tqdm(fpaths,dynamic_ncols=True,desc='Computing Eclipse Track'):
        bname = os.path.basename(fpath)
        files.append(bname)

        date  = datetime.datetime.strptime(bname[:13],'%Y%m%d.%H%M')
        alt   = str(bname[14:17])

        df          = pd.read_csv(fpath,comment='#')
        ecl_center  = find_eclipse_center(df)

        if (track_geometry_path is not None) and (ecl_center is not None):
            if not os.path.exists(track_geometry_path):
                os.makedirs(track_geometry_path)
            plot_obscuration_fname = '{!s}_geometry_view.png'.format(date.strftime('%Y%m%d.%H%M'))
            plot_obscuration_fpath = os.path.join(track_geometry_path,plot_obscuration_fname)
            result      =  eclipse_calc.calculate_obscuration(date,ecl_center['lat'],ecl_center['lon'],
                    height=ecl_center['height'],return_dict=True,plot_obscuration=plot_obscuration_fpath)

        if ecl_center:
            dates.append(date)
            ecl_track.append(ecl_center)

    ecl_track = pd.DataFrame(ecl_track,index=dates)

    # Compute the azimuth that the eclipse track is heading.
    azms = []
    for inx,(rinx,row_0) in enumerate(ecl_track.iterrows()):
        if inx == len(ecl_track)-1:
            azms.append(np.nan)
            continue

        row_1 = ecl_track.iloc[inx+1]
        
        lat_0 = row_0['lat']
        lon_0 = row_0['lon']
        lat_1 = row_1['lat']
        lon_1 = row_1['lon']

        azm   = eclipse_calc.geopack.greatCircleAzm(lat_0,lon_0,lat_1,lon_1)
        azms.append(azm)

    ecl_track['track_azm_deg'] = azms

    sDate = min(dates)
    eDate = max(dates)

    if out_csv_fname:
        hdr = []
        hdr.append('# Solar Eclipse center track for {!s} - {!s}'.format(sDate,eDate))
        hdr.append(','.join(['date_ut']+ecl_track.columns.to_list()))
        hdr = '\n'.join(hdr)
        hdr += '\n'

        with bz2.open(out_csv_fname,'wt') as bzfl:
            bzfl.write(hdr)
            ecl_track.to_csv(bzfl,header=False)

    return ecl_track 
    
if __name__ == '__main__':
    timer = ScriptTimer()

    recalc_eclipse  = True
    multiproc       = True
    ncpus           = multiprocessing.cpu_count()

    seDates = []

    # 21 August 2017 Total Solar Eclipse
#    dd = {}
#    dd['sDate'] = datetime.datetime(2017,8,21,14)
#    dd['eDate'] = datetime.datetime(2017,8,21,22)
#    seDates.append(dd)

    # 14 October 2023 Annular Solar Eclipse
    dd = {}
    dd['sDate'] = datetime.datetime(2023,10,14,14)
    dd['eDate'] = datetime.datetime(2023,10,14,21)
    seDates.append(dd)

    # 8 April 2024 Total Solar Eclipse
    dd = {}
    dd['sDate'] = datetime.datetime(2024,4,8,15)
    dd['eDate'] = datetime.datetime(2024,4,8,21)
    seDates.append(dd)

#    # SHORT TEST CASES
#    # 8 April 2024 Total Solar Eclipse - SHORT TEST CASE
#    dd = {}
#    dd['sDate'] = datetime.datetime(2024,4,8,18)
#    dd['eDate'] = datetime.datetime(2024,4,8,19)
#    seDates.append(dd)

#    # 14 October 2023 Annular Solar Eclipse
#    dd = {}
#    dd['sDate'] = datetime.datetime(2023,10,14,18)
#    dd['eDate'] = datetime.datetime(2023,10,14,19)
#    seDates.append(dd)

    heights     = np.arange(0,500,50)*1e3

    # Create Run Dictionaries for all iterations of dates, heights.
    run_dicts = []
    for seDate in seDates:
        for height in heights:
            rd = seDate.copy()
            rd['height'] = height
            run_dicts.append(rd)

    dt      = datetime.timedelta(minutes=5)

    # Latitude / Longitude Resolution
    dlat        = 0.2
    dlon        = 0.2

    for rd in run_dicts:
        sDate   = rd['sDate']
        eDate   = rd['eDate']
        height  = rd['height']

        ################################################################################ 
        event_name  = get_event_name(sDate,eDate,height,dlat,dlon)
        output_dir  = os.path.join('output',event_name)
        frames_dir  = os.path.join(output_dir,'frames')

        if recalc_eclipse:
            eclipse_calc.gen_lib.clear_dir(output_dir)
            eclipse_calc.gen_lib.make_dir(frames_dir)

        loc_dict    = location_dict(dlat,dlon,height)

        run_list    = []
        cDate       = sDate
        while cDate < eDate:
            tmp = {}
            tmp['date']         = cDate
            tmp['loc_dict']     = loc_dict
            tmp['output_dir']   = frames_dir
            run_list.append(tmp)
            cDate   += dt

        if recalc_eclipse:
            ## Calculate Eclipse Data and Plot
            if multiproc:
                with multiprocessing.Pool(ncpus) as pool:
                    pool.map(calc_and_plot_eclipse,run_list)
            else:
                # Single Processor
                for run_dict in run_list:
                    fpath = calc_and_plot_eclipse(run_dict)

        ## Calculate maximum obscuration in each lat-lon cell.
        max_obsc_bname  = os.path.join(output_dir,event_name+'_MAX_OBSCURATION')
        out_csv_fname   = max_obsc_bname+'.csv.bz2'
        max_obsc_df     = calc_max_obsc(frames_dir,out_csv_fname=out_csv_fname)

        ## Calculate eclipse track for event.
        ecl_track_bname = os.path.join(output_dir,event_name+'_ECLIPSE_TRACK')
        out_csv_fname   = ecl_track_bname+'.csv.bz2'
        track_geometry_path = os.path.join(output_dir,'track_geometry')
        ecl_track_df    = compute_eclipse_track(frames_dir,out_csv_fname=out_csv_fname,
                track_geometry_path=track_geometry_path)

        ## Plot full maximum obscuration map.
        out_png_fname   = max_obsc_bname+'.png'
        png_path        = plot_eclipse(max_obsc_df,sDate,eDate,nightshade=False,fig_path=out_png_fname)

        ## Plot max obscurations greater than 90%.
        min_obsc        = 0.9
        out_png_fname   = max_obsc_bname+'_{!s}minObsc.png'.format(min_obsc)
        png_path        = plot_eclipse(max_obsc_df,sDate,eDate,nightshade=False,fig_path=out_png_fname,
                                min_obsc=min_obsc)

        ## Plot full maximum obscuration map with track.
        out_png_fname   = ecl_track_bname+'.png'
        png_path        = plot_eclipse(max_obsc_df,sDate,eDate,nightshade=False,fig_path=out_png_fname,
                                ecl_track_df=ecl_track_df)

        ## Plot max obscurations greater than 90% with eclipse_track.
        min_obsc        = 0.9
        out_png_fname   = ecl_track_bname+'_{!s}minObsc.png'.format(min_obsc)
        png_path        = plot_eclipse(max_obsc_df,sDate,eDate,nightshade=False,fig_path=out_png_fname,
                                min_obsc=min_obsc,ecl_track_df=ecl_track_df)

    timer.stop()
