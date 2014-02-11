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


def regular_grid(spacing,lon0=-180,centerlon=0,addcyclic=False):
    """Generate a regular grid with the given *spacing*,
    avoiding points at the poles and at the equator.

    Parameters
    ----------
    lon0 : float
        western-most longitude for the grid (default: 180)
        the longitude returned runs from lon0 to lon0+360
    centerlon : float, optional
        longitude where to center the grid
        if centerlon is given, lon0 is set to centerlon-180
    """
    if centerlon:
        lon0 = centerlon-180
    return (np.arange(lon0,lon0+360+(spacing*addcyclic),spacing),
            np.arange(-90-spacing/2.,90+spacing/2.+1e-8,spacing)[1:-1])


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


def binary_search_grid(gridlon,gridlat,px,py,returndist=False):
    """Find nearest grid points on a lon-lat grid

    Parameters
    ==========
    gridlon,gridlat : arrays (1D or 2D)
        longitude / latitude grid
    px,py : floats/lists/arrays
        one or more position to be found in the grid
    returndist : bool, optional
        return distance from target point to nearest grid point

    Returns
    =======
    ix,iy : indices for nearest points on grid (default)
    ix,iy,dist : same as above plus distance from given positions (returndist=True)

    Dependencies
    ============
    haversine
    
    Credits
    =======
    Mads Hvid Ribergaard mhri@dmi.dk (MATLAB version)
       
    """
    idm,jdm = np.shape(gridlon)

    def binary_search_grid_single(gridlon,gridlat,x,y):
        # initialize indices
        Lo = np.array([0,0])
        Hi = np.array([idm-1,jdm-1])

        def centerpoints(Lo,Hi):
            imid = np.array([Lo[0] + np.floor(.25*(Hi[0]-Lo[0])),
                             Hi[0] - np.floor(.25*(Hi[0]-Lo[0]))],dtype=int)
            jmid = np.array([Lo[1] + np.floor(.25*(Hi[1]-Lo[1])),
                             Hi[1] - np.floor(.25*(Hi[1]-Lo[1]))],dtype=int)
            return np.ix_(imid,jmid)
        
        imid,jmid = centerpoints(Lo,Hi)

        # Loop: Recrusive decrease square until 2*2 square is found
        while np.diff(imid) > 1 or np.diff(jmid) > 1:

            # calculate distance from point x,y to four corner grid points
            kmin = haversine(gridlon[imid,jmid],gridlat[imid,jmid],x,y).argmin()

            # determine which corner point is nearest
            if kmin == 0:
                Hi = Hi - np.floor(.5*(Hi-Lo))
            elif kmin == 1:
                Lo[0] = Lo[0] + np.floor(.5*(Hi[0]-Lo[0]))
                Hi[1] = Hi[1] - np.floor(.5*(Hi[1]-Lo[1]))
            elif kmin == 2:
                Lo[1] = Lo[1] + np.floor(.5*(Hi[1]-Lo[1]))
                Hi[0] = Hi[0] - np.floor(.5*(Hi[0]-Lo[0]))
            elif kmin == 3:
                Lo = Lo + np.floor(.5*(Hi-Lo))
          
            imid,jmid = centerpoints(Lo,Hi)

        # Find closest point from the remaining 2*2 square
        dist = haversine(gridlon[imid,jmid],gridlat[imid,jmid],x,y); print(dist,dist.argmin())
        kmin = dist.argmin()
        if kmin == 0:
            ele = Lo
        elif kmin == 1:
            ele = np.array([Hi[0],Lo[1]])
        elif kmin == 2:
            ele = np.array([Lo[0],Hi[1]])
        elif kmin == 3:
            ele = Hi

        # Did we find global minimum or just local (weired grid)?
        #check here

        ele = ele.astype(int)
        return ele[0],ele[1],dist.flat[kmin]

    # compute for all given points
    try:
        n=len(px)
        ix,iy = np.zeros((n),np.int),np.zeros((n),np.int)
        dist = np.zeros((n))
        for p in xrange(n):
            ix[p],iy[p],dist[p] = binary_search_grid_single(gridlon,gridlat,px[p],py[p])
    except TypeError:
        ix,iy,dist = binary_search_grid_single(gridlon,gridlat,px,py)

    if returndist:
        return ix,iy,dist
    else:
        return ix,iy


def nearest_gridpt(gridlon,gridlat,lon0,lat0,dlon=None,dlat=None,unravel=True):
    """Find nearest points in a lon-lat grid

    computing the distance for each grid point (slow for large grids!).

    Parameters
    ==========
    gridlon,gridlat : arrays (1D or 2D)
        longitude / latitude grid
    lon0,lat0 : float or array-like (1D)
        one or more position to be found in the grid
    dlon,dlat : float, optional
        surrounding where nearest gridpoints are expected to be found
    unravel : bool, optional
        whether to return multidimensional or linear index

    Returns
    =======
    ind : array
        linear index, if unravel=False
    ix,iy : tuple of arrays
        multiple indices, if unravel=True (default)

    Dependencies
    ============
    haversine

    """    
    gridlon = np.mod(gridlon,360)
    lon0 = np.mod(lon0,360)

    def _nearest(gridlon,gridlat,lon0,lat0,dlon,dlat):
        mask = np.zeros(np.shape(gridlon))
        if dlon is not None:
            mask &= (gridlon < lon0-dlon) | (gridlon > lon0+dlon)
        if dlat is not None:
            mask &= (gridlat < lat0+dlat) | (gridlat > lat0+dlat)
        gridlon = np.ma.masked_where(gridlon)
        gridlat = np.ma.masked_where(gridlat)
        return haversine(gridlon,gridlat,x,y).argmin()

    _vecnearest = np.vectorize(_nearest,doc=nearest_gridpt.__doc__,
                               excluded=['gridlon','gridlat','dlon','dlat'])

    ind = _vecnearest(gridlon,gridlat,lon0,lat0,dlon,dlat)

    if not unravel or np.ndim(gridlon) == 1:
        return ind
    else:
        return np.unravel_index(ind,gridlon.shape)
        

def lonlatbounds(lons,lats,mode=4):
    """Genereate grid boundary axes for pcolor plot
    
    from given lons,lats vectors.
    Works only for regularly spaced grids
    """
    delon = lons[1]-lons[0]
    delat = lats[1]-lats[0]
    if (mode == 1) : # extend existing grid points to the southeast
        lonbounds=np.zeros(len(lons)+1)
        latbounds=np.zeros(len(lats)+1)
        lonbounds[0:-1] = lons
        latbounds[0:-1] = lats
        lonbounds[-1] = lonbounds[-2]+delon
        latbounds[-1] = latbounds[-2]+delat
    elif (mode == 2) : # extend existing grid points to the northwest
        lonbounds=np.zeros(len(lons)+1)
        latbounds=np.zeros(len(lats)+1)
        lonbounds[1:] = lons
        latbounds[1:] = lats
        lonbounds[0] = lonbounds[1]-delon
        latbounds[0] = latbounds[1]-delat
    elif (mode == 3) : # extend existing grid points to the southwest
        lonbounds=np.zeros(len(lons)+1)
        lonbounds[1:] = lons
        lonbounds[0] = lonbounds[1]-delon
        latbounds=np.zeros(len(lats)+1)
        latbounds[0:-1] = lats
        latbounds[-1] = latbounds[-2]+delat
    elif (mode == 4) : # use existing grid points as midpoints of their respective grid boxes
        lonbounds=np.zeros(len(lons)+1)
        latbounds=np.zeros(len(lats)+1)
        lonbounds[0:-1] = lons - 0.5*delon
        latbounds[0:-1] = lats - 0.5*delat
        lonbounds[-1] = lonbounds[-2]+delon
        latbounds[-1] = latbounds[-2]+delat

    return (latbounds,lonbounds)


def lat2str(deg):
    degrees = np.sign(deg) * np.floor(np.abs(deg))
    minutes = np.sign(deg) * (np.abs(deg) - np.abs(degrees)) * 60
    direction = ['S','N'][deg<0]
    if minutes == 0:
        return u"{:.0f}\N{DEGREE SIGN}{}".format(np.abs(degrees),direction)
    else:
        return u"{:.0f}\N{DEGREE SIGN}{:.0f}'{}".format(np.abs(degrees),np.abs(minutes),direction)


def lon2str(deg,remap180=False):
    if remap180:
        deg = remap_lon_180(deg)
    degrees = np.sign(deg) * np.floor(np.abs(deg))
    minutes = np.sign(deg) * (np.abs(deg) - np.abs(degrees)) * 60
    direction = ['W','E'][deg>0]
    if minutes == 0:
        return u"{:.0f}\N{DEGREE SIGN}{}".format(np.abs(degrees),direction)
    else:
        return u"{:.0f}\N{DEGREE SIGN}{:.0f}\'{}".format(np.abs(degrees),np.abs(minutes),direction)

