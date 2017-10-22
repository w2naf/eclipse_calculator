#!/usr/bin/env python
de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50
Re              = 6371  # Radius of the Earth

import os               # Provides utilities that help us do os-level operations like create directories
import datetime         # Really awesome module for working with dates and times.

import numpy as np      #Numerical python - provides array types and operations
import pandas as pd     #This is a nice utility for working with time-series type data.

# Some view options for debugging.
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib
#matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap

from . import locator

def cc255(color):
    cc = matplotlib.colors.ColorConverter().to_rgb
    trip = np.array(cc(color))*255
    trip = [int(x) for x in trip]
    return tuple(trip)

class BandData(object):
    def __init__(self,cmap='HFRadio',vmin=0.,vmax=30.):
        if cmap == 'HFRadio':
            self.cmap   = self.hf_cmap(vmin=vmin,vmax=vmax)
        else:
            self.cmap   = matplotlib.cm.get_cmap(cmap)

        self.norm   = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

        # Set up a dictionary which identifies which bands we want and some plotting attributes for each band
        bands   = []
        bands.append((28.0,  '10 m'))
        bands.append((21.0,  '15 m'))
        bands.append((14.0,  '20 m'))
        bands.append(( 7.0,  '40 m'))
        bands.append(( 3.5,  '80 m'))
        bands.append(( 1.8, '160 m'))

        self.__gen_band_dict__(bands)

    def __gen_band_dict__(self,bands):
        dct = {}
        for freq,name in bands:
            key = int(freq)
            tmp = {}
            tmp['name']         = name
            tmp['freq']         = freq
            tmp['freq_name']    = '{:g} MHz'.format(freq)
            tmp['color']        = self.get_rgba(freq)
            dct[key]            = tmp
        self.band_dict          = dct

    def get_rgba(self,freq):
        nrm     = self.norm(freq)
        rgba    = self.cmap(nrm)
        return rgba

    def get_hex(self,freq):

        freq    = np.array(freq)
        shape   = freq.shape
        if shape == ():
            freq.shape = (1,)

        freq    = freq.flatten()
        rgbas   = self.get_rgba(freq)

        hexes   = []
        for rgba in rgbas:
            hexes.append(matplotlib.colors.rgb2hex(rgba))

        hexes   = np.array(hexes)
        hexes.shape = shape
        return hexes

    def hf_cmap(self,name='HFRadio',vmin=0.,vmax=30.):
        fc = {}
        my_cdict = fc
        fc[ 0.0] = (  0,   0,   0)
        fc[ 1.8] = cc255('violet')
        fc[ 3.0] = cc255('blue')
        fc[ 8.0] = cc255('aqua')
        fc[10.0] = cc255('green')
        fc[13.0] = cc255('green')
        fc[17.0] = cc255('yellow')
        fc[21.0] = cc255('orange')
        fc[28.0] = cc255('red')
        fc[30.0] = cc255('red')
        cmap    = cdict_to_cmap(fc,name=name,vmin=vmin,vmax=vmax)
        return cmap

def cdict_to_cmap(cdict,name='CustomCMAP',vmin=0.,vmax=30.):
	norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
	
	red   = []
	green = []
	blue  = []
	
	keys = list(cdict.keys())
	keys.sort()
	
	for x in keys:
	    r,g,b, = cdict[x]
	    x = norm(x)
	    r = r/255.
	    g = g/255.
	    b = b/255.
	    red.append(   (x, r, r))
	    green.append( (x, g, g))
	    blue.append(  (x, b, b))
	cdict = {'red'   : tuple(red),
		 'green' : tuple(green),
		 'blue'  : tuple(blue)}
	cmap  = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	return cmap

def band_legend(fig=None,loc='lower center',markerscale=0.5,prop={'size':10},
        title=None,bbox_to_anchor=None,rbn_rx=True,ncdxf=False,ncol=None,band_data=None):

    if fig is None: fig = plt.gcf() 

    if band_data is None:
        band_data = BandData()

    handles = []
    labels  = []

    # Force freqs to go low to high regardless of plotting order.
    band_list   = list(band_data.band_dict.keys())
    band_list.sort()
    for band in band_list:
        color = band_data.band_dict[band]['color']
        label = band_data.band_dict[band]['freq_name']
        handles.append(mpatches.Patch(color=color,label=label))
        labels.append(label)

    fig_tmp = plt.figure()
    ax_tmp = fig_tmp.add_subplot(111)
    ax_tmp.set_visible(False)
    if rbn_rx:
        scat = ax_tmp.scatter(0,0,s=50,**de_prop)
        labels.append('RBN Receiver')
        handles.append(scat)
    if ncdxf:
        scat = ax_tmp.scatter(0,0,s=dxf_leg_size,**dxf_prop)
        labels.append('NCDXF Beacon')
        handles.append(scat)

    if ncol is None:
        ncol = len(labels)
    
    legend = fig.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
    return legend

class HamMap(object):
    """Plot Reverse Beacon Network data.

    **Args**:
        * **[sTime]**: datetime.datetime object for start of plotting.
        * **[eTime]**: datetime.datetime object for end of plotting.
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Nathaniel Frissell 2014 Sept 06
    """
    def __init__(self,sTime,eTime,ax=None,
            llcrnrlon=-180.,llcrnrlat=-90.,urcrnrlon=180.,urcrnrlat=90.,
            coastline_color='0.65',coastline_zorder=10,band_data=None,show_title=True,
            subtitle=None):

        llb = {}
        llb['llcrnrlon'] = llcrnrlon
        llb['llcrnrlat'] = llcrnrlat
        llb['urcrnrlon'] = urcrnrlon
        llb['urcrnrlat'] = urcrnrlat
        self.latlon_bnds    = llb

        self.metadata       = {}
        self.metadata['sTime'] = sTime
        self.metadata['eTime'] = eTime

        if band_data is None:
            band_data = BandData()
        self.band_data = band_data

        self.__setup_map__(ax=ax,
                coastline_color=coastline_color,coastline_zorder=coastline_zorder,
                subtitle=subtitle,show_title=show_title,**self.latlon_bnds)

#        self.plot_nightshade()
#        self.plot_band_legend(band_data=self.band_data)
#        self.overlay_gridsquares()

    def __setup_map__(self,ax=None,llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,
            coastline_color='0.65',coastline_zorder=10,show_title=True,subtitle=None):
        sTime       = self.metadata['sTime']
        eTime       = self.metadata['eTime']

        if ax is None:
            fig     = plt.figure(figsize=(10,6))
            ax      = fig.add_subplot(111)
        else:
            fig     = ax.get_figure()

        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection='cyl',ax=ax)

        if show_title:
            title = sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
            fontdict = {'size':matplotlib.rcParams['axes.titlesize'],'weight':matplotlib.rcParams['axes.titleweight']}
            ax.text(0.5,1.075,title,fontdict=fontdict,transform=ax.transAxes,ha='center')

        if subtitle is not None:
            fontdict    = {'weight':'normal'}
            ax.text(0.5,1.025,subtitle,fontdict=fontdict,transform=ax.transAxes,ha='center')

        # draw parallels and meridians.
        # This is now done in the locator. overlay section...
#        m.drawparallels(np.arange( -90., 91.,45.),color='k',labels=[False,True,True,False])
#        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True])
        m.drawcoastlines(color=coastline_color,zorder=coastline_zorder)
        m.drawmapboundary(fill_color='w')

        # Expose select object
        self.fig        = fig
        self.ax         = ax
        self.m          = m

    def plot_nightshade(self,color='0.60'):
        # Overlay nighttime terminator.
        sTime       = self.metadata['sTime']
        eTime       = self.metadata['eTime']
        half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
        center_time = sTime + half_time
        self.m.nightshade(center_time,color=color)
        
    def plot_band_legend(self,*args,**kw_args):
        band_legend(*args,**kw_args)

    def overlay_gridsquares(self,
            major_precision = 2,    major_style = {'color':'k',   'dashes':[1,1]}, 
            minor_precision = 0,    minor_style = {'color':'0.8', 'dashes':[1,1]},
            label_precision = 2,    label_fontdict=None, label_zorder = 100):
        """
        Overlays a grid square grid.

        Precsion options:
            None:       Gridded resolution of data
            0:          No plotting/labling
            Even int:   Plot or label to specified precision
        """
    
        # Get the dataset and map object.
        m           = self.m
        ax          = self.ax

        major_style['zorder'] = 200
        minor_style['zorder'] = 200

        # Determine the major and minor precision.
        maj_prec    = major_precision
        min_prec    = minor_precision
        label_prec  = label_precision

	# Draw Major Grid Squares
        if maj_prec > 0:
            lats,lons   = locator.grid_latlons(maj_prec,position='lower left')

            m.drawparallels(lats[0,:],labels=[False,True,True,False],**major_style)
            m.drawmeridians(lons[:,0],labels=[True,False,False,True],**major_style)

	# Draw minor Grid Squares
        if min_prec > 0:
            lats,lons   = locator.grid_latlons(min_prec,position='lower left')

            m.drawparallels(lats[0,:],labels=[False,False,False,False],**minor_style)
            m.drawmeridians(lons[:,0],labels=[False,False,False,False],**minor_style)

	# Label Grid Squares
        if label_prec > 0:
            lats,lons   = locator.grid_latlons(label_prec,position='center')
            grid_grid   = locator.gridsquare_grid(label_prec)
            xx,yy = m(lons,lats)
            for xxx,yyy,grd in zip(xx.ravel(),yy.ravel(),grid_grid.ravel()):
                ax.text(xxx,yyy,grd,ha='center',va='center',clip_on=True,
                        fontdict=label_fontdict, zorder=label_zorder)

    def overlay_gridsquare_data(self,gridsquares,data,
            cmap=None,vmin=None,vmax=None,zorder=99,
            label=None,plot_cbar=True):
        """
        Overlay gridsquare data on a map.
        """

        if cmap is None: cmap = matplotlib.cm.jet
        if vmin is None: vmin = np.min(data)
        if vmax is None: vmax = np.max(data)

        ll                  = locator.gridsquare2latlon
        lats_ll, lons_ll    = ll(gridsquares,'lower left')
        lats_lr, lons_lr    = ll(gridsquares,'lower right')
        lats_ur, lons_ur    = ll(gridsquares,'upper right')
        lats_ul, lons_ul    = ll(gridsquares,'upper left')

        coords  = zip(lats_ll,lons_ll,lats_lr,lons_lr,
                      lats_ur,lons_ur,lats_ul,lons_ul)

        verts   = []
        for lat_ll,lon_ll,lat_lr,lon_lr,lat_ur,lon_ur,lat_ul,lon_ul in coords:
            x1,y1 = self.m(lon_ll,lat_ll)
            x2,y2 = self.m(lon_lr,lat_lr)
            x3,y3 = self.m(lon_ur,lat_ur)
            x4,y4 = self.m(lon_ul,lat_ul)
            verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

        vals    = data

        bounds  = np.linspace(vmin,vmax,256)
        norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

        pcoll   = PolyCollection(np.array(verts),edgecolors='face',closed=False,cmap=cmap,norm=norm,zorder=zorder)
        pcoll.set_array(np.array(vals))
        self.ax.add_collection(pcoll,autolim=False)

        if plot_cbar:
            cbar    = self.fig.colorbar(pcoll,label=label,shrink=0.7)

        return pcoll
