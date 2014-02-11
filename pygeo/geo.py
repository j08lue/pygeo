"""Geo-related routines"""
import numpy as np

def latlen(lat):
    """Return the lenght of one degree of latitude at the given latitude [deg] in metres"""
    lat = np.radians(lat)
    return 111132.954 - 559.822 * np.cos(2.*lat) + 1.175 * np.cos(4.*lat)


def lonlen(lat,
           a=6378137.0, # WGS84 ellipsoid
           b=6356752.3142):
    """Return the lenght of one degree of longitude at the given latitude [deg] in metres"""
    esq = (a**2 - b**2)/a**2
    lat = np.radians(lat)
    return np.pi * a * np.cos(lat) / ( 180. * np.sqrt(1 - esq*np.sin(lat)**2) )


def dist2lon(dist,lat):
    """Convert from distance [m] to decimal degree of longitude"""
    return dist / lonlen(lat)


def dist2lat(dist,lat):
    """Convert from distance [m] to decimal degree of latitude"""
    return dist / latlen(lat)


def remap_lon_180(lon):
    """Remap longitude coordinates `lon` to the -180:180 E range"""
    c = np.mod(lon,360)
    return np.where(c > 180,c-360,c)[()]


def make_lon_continuous(lon):
    """Make a sequence of longitudes `lon` (1D or 2D[y,x]) continuously increasing
    by subtracting 360 where there are cyclic breaks.
    """
    c = lon#np.mod(lon,360)
    breaks = np.where(np.diff(c) < 0)
    if len(breaks) == 2:
        for j,i in np.array(breaks).T:
            c[j,:i+1] -= 360
    elif len(breaks) == 1:
        for i in breaks[0]:
            c[:i+1] -= 360
    else:
        raise NotImplementedError('Works only for 1D or 2D arrays.')
    return c


def spherical2cartesian(lon,lat):
    """Convert spherical to cartesian coordinates"""
    rlon,rlat = np.map(np.radians,lon,lat)
    x = np.cos(rlat) * np.cos(rlon)
    y = np.cos(rlat) * np.sin(rlon)
    z = np.sin(rlat)
    return x,y,z


def arclength(lon1, lat1, lon2, lat2):
    """Returns arc length [radians] between two points [dec deg]

    Parameters
    ==========
    lon1,lat1,lon2,lat2 : float [dec deg]
        coordinates for two points

    Returns
    =======
    arclength : float [rad]
    
    Credits
    =======
    http://code.google.com/p/pyroms/
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # compute arc length
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    b = 2. * np.arcsin(np.sqrt(a))
    return b


def haversine(lon1, lat1, lon2, lat2):
    """Returns the great circle distance between two positions on a sphere.

    Parameters
    ==========
    lon1,lat1,lon2,lat2 : float [dec deg]
        coordinates for two points

    Returns
    =======
    distance [metres]
    
    """
    c = arclength(lon1, lat1, lon2, lat2)
    return 6378.1 * 1e3 * c


def bearing(lon1, lat1, lon2, lat2):
    """Returns the bearing between two points on a sphere

    Parameters
    ==========
    lon,lat : float [decimal degrees]

    Returns
    =======
    bearing : float [decimal degrees]
    
    Credits
    =======
    http://www.movable-type.co.uk/scripts/latlong.html
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # compute bearing
    return np.degrees(
        np.arctan2(
            (np.sin(lon2-lon1) * np.cos(lat2)),
            (np.cos(lat1) * np.sin(lat2) -
             np.sin(lat1) * np.cos(lat2) * np.cos(lon2-lon1))) )


def waypoints(lon1, lat1, lon2, lat2, f=0.5, n=None):
    """Returns intermediate waypoints on a great circle

    connecting the two given end points at a fraction f [%]
    of the distance from lon1,lat1.
    
    Parameters
    ==========
    lon1,lat1,lon2,lat2 : float
        start and end point coordinates [deg]
    f : float/array, optional
        fraction (can be a vector)
    n : integer, optinal
        alternatively to f, number of waypoints

    Note
    ====
    Either f or n must be provided. If both are, if is used.
    
    Credits
    =======
    http://williams.best.vwh.net/avform.htm#Intermediate
    """
    # if given, compute f from n
    if n != None:
        f = np.linspace(0,1,n)
    
    # compute arc length between points
    d = arclength(lon1, lat1, lon2, lat2)

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # compute waypoints
    a = np.sin((1.-f)*d) / np.sin(d)
    b = np.sin(f*d) / np.sin(d)
    x = a*np.cos(lat1)*np.cos(lon1) + b*np.cos(lat2)*np.cos(lon2)
    y = a*np.cos(lat1)*np.sin(lon1) + b*np.cos(lat2)*np.sin(lon2)
    z = a*np.sin(lat1)              + b*np.sin(lat2)
    lat = np.arctan2(z,np.sqrt(x**2+y**2.))
    lon = np.arctan2(y,x)
    return map(np.degrees,[lon,lat])


def waypoints_segments(lons, lats, f=None, n=10, returndist=False):
    """Refines a given lon-lat section by adding waypoints

    such that the resulting axis has n steps from one given point to the next.

    Parameters
    ==========
    lons,lats : array
        vectors defining a lon,lat track to be refined
    f : array, optional
        fraction at which to insert waypoints (favored over n)
    n : int, optional
        target number of steps from one point to the next
    returndist : bool, optional
        whether or not to return an axis with distance from first point

    Returns
    =======
    lons,lats : tuple of new axes
    lons,lats,dist : if returndist=True

    Dependencies
    ============
    waypoints, haversine
    """

    # generate fraction
    if f == None:
        f = np.linspace(0,1,n)
    else:
        n = len(f)

    # get lengths and initialize
    ntot = (len(lons)-1)*n
    waylons,waylats = np.zeros(ntot),np.zeros(ntot)

    # iterate through segments
    for p in xrange(len(lons)-1):
        ia = p*n
        io = (p+1)*n
        
        # get waypoints
        waylons[ia:io],waylats[ia:io] = waypoints(lons[p],lats[p],lons[p+1],lats[p+1],f)

    # compute distance
    if returndist:    
        dist = np.cumsum([0.]+[haversine(lon1, lat1, lon2, lat2) for lon1,lat1,lon2,lat2 in zip(waylons[1:],waylats[1:],waylons[:-1],waylats[:-1])])
        
        return waylons,waylats,dist
    else:
        return waylons,waylats


def lat2str(deg):
    degrees = np.sign(deg) * np.floor(np.abs(deg))
    minutes = np.sign(deg) * (np.abs(deg) - np.abs(degrees)) * 60
    direction = ['S','N'][deg<0]
    if minutes == 0:
        return u"{:.0f}\N{DEGREE SIGN}{}".format(np.abs(degrees),direction)
    else:
        return u"{:.0f}\N{DEGREE SIGN}{:.0f}'{}".format(np.abs(degrees),np.abs(minutes),direction)


def lon2str(deg,remap180=True):
    if remap180:
        deg = remap_lon_180(deg)
    degrees = np.sign(deg) * np.floor(np.abs(deg))
    minutes = np.sign(deg) * (np.abs(deg) - np.abs(degrees)) * 60
    direction = ['W','E'][deg>0]
    if minutes == 0:
        return u"{:.0f}\N{DEGREE SIGN}{}".format(np.abs(degrees),direction)
    else:
        return u"{:.0f}\N{DEGREE SIGN}{:.0f}\'{}".format(np.abs(degrees),np.abs(minutes),direction)

