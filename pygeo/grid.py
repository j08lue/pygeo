import numpy as np

from . import geo


def addcyclic(*arr, **axiskwarg):
    """
    ``arrout, lonsout = addcyclic(arrin,lonsin,axis=-1)``
    adds cyclic (wraparound) points in longitude to one or several arrays, 
    (e.g. ``arrin`` and ``lonsin``),
    where ``axis`` sets the dimension longitude is in (optional, default: right-most).
    """
    # get axis keyword argument (default: -1)
    axis = axiskwarg.get('axis', -1)
    # define function for a single grid array
    def _addcyclic(a):
        aT = np.swapaxes(a, 0, axis)
        idx = np.append(np.arange(aT.shape[0]), 0)
        return np.swapaxes(aT[idx], axis, 0)
    # process array(s)
    if len(arr) == 1:
        return _addcyclic(arr[0])
    else:
        return map(_addcyclic, arr)


def regular_grid(spacing,lon0=-180,centerlon=0,addcyclic=False):
    """Generate a regular grid with the given *spacing*,
    avoiding points at the poles and at the equator.

    Parameters
    ----------
    lon0 : float
        western-most longitude for the grid (default: -180)
        the longitude returned runs from lon0 to lon0+360
    centerlon : float, optional
        longitude where to center the grid
        if centerlon is given, lon0 is set to centerlon-180

    Returns
    -------
    lons, lats
    """
    if centerlon != 0:
        lon0 = centerlon-180
    return (np.arange(lon0,lon0+360+(spacing*addcyclic),spacing),
            np.arange(-90+spacing/2.,90,spacing))
    

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
            kmin = geo.haversine(gridlon[imid,jmid],gridlat[imid,jmid],x,y).argmin()

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
        dist = geo.haversine(gridlon[imid,jmid],gridlat[imid,jmid],x,y);
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
        return geo.haversine(gridlon,gridlat,lon0,lat0).argmin()

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


