#!/usr/bin/env python3

import datetime
import ephem
import numpy as np

def area_intersect(r_sun,r_moon,d):
    """
    Calculate the area of intersecting circles with radii R
    and r and separation d.
    
    Reference:
    Weisstein, Eric W. "Circle-Circle Intersection."
        From MathWorld--A Wolfram Web Resource.
        http://mathworld.wolfram.com/Circle-CircleIntersection.html
    """
    
    R = r_sun
    r = r_moon
    
    A = ( r**2 * np.arccos( (d**2 + r**2 - R**2)/(2*d*r))
        + R**2 * np.arccos( (d**2 + R**2 - r**2)/(2*d*R))
        - 0.5 * np.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R))
        )
    
    if np.isnan(A):
        if r_sun > r_moon:
            A = np.pi * r_moon**2
        elif r_sun <= r_moon:
            A = np.pi * r_sun**2

    return A

def eclipse_mag(lat,lon,date_time,debug=False):
    obs         = ephem.Observer()
    obs.lat     = '{!s}'.format(lat)
    obs.lon     = '{!s}'.format(lon)
    obs.date    = date_time.strftime('%Y/%m/%d %H:%M:%S')
    sun         = ephem.Sun()
    moon        = ephem.Moon()

    sun.compute(obs)
    moon.compute(obs)

    sep = ephem.separation((sun.az, sun.alt), (moon.az, moon.alt))

    sun_size_deg  = sun.size/3600.
    moon_size_deg = moon.size/3600.

    sun_size_rad  = sun_size_deg  * np.pi/180.
    moon_size_rad = moon_size_deg * np.pi/180.

    A            = area_intersect(sun_size_rad,moon_size_rad,sep)
    sun_area_rad = np.pi*sun_size_rad**2
    mag          = A/sun_area_rad

    if debug:
        print('Sun:  {:f} arcsec'.format(sun.size))
        print('Moon: {:f} arcsec'.format(moon.size))
        print('Sep:  {!s}'.format(sep))
        print('')

        print('Sun:  {:f} deg'.format(sun_size_deg))
        print('Moon: {:f} deg'.format(moon_size_deg))
        print('Sep:  {:f} deg'.format(np.degrees(sep)))
        print('')

        print('Sun:  {:f} radians'.format(sun_size_rad))
        print('Moon: {:f} radians'.format(moon_size_rad))
        print('Sep:  {:f} radians'.format(sep))
        print('')

        print('Intersection Area: {:f}'.format(A))
        print('Sun Area: {:f}'.format(sun_area_rad))
        print('Eclipse Magnitude: {:f}'.format(mag))
    
    return mag

if __name__ == '__main__':
    # UACNJ Jenny Jump - Hope, NJ
    # http://xjubier.free.fr/en/site_pages/solar_eclipses/TSE_2017_GoogleMapFull.html?Lat=40.90743&Lng=-74.92505&Elv=-1.0&Zoom=4&LC=1
    # Obscuration: 72.284%
    # Mag at Max: 0.77541
    # Moon/Sun size ratio: 1.02879
    lat =  40.90743
    lon = -74.92505
    partial_0   = datetime.datetime(2017,8,21,17,21,10)
    total_0     = None
    eclipse_max = datetime.datetime(2017,8,21,18,43,13)
    total_1     = None
    partial_1   = datetime.datetime(2017,8,21,19,59,29)

    
    mag         = eclipse_mag(lat,lon,eclipse_max,debug=True)
    print(mag)
