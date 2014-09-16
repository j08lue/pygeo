"""Regrid data from one grid to another. Useful e.g. for plotting data that is on an irregular grid"""
import numpy as np
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import NearestNDInterpolator

def regrid_lonlat(lon,lat,data,targetlon,targetlat,mask=None,m=None,lon0=None,returnxy=False):
    """Regrid data from one lon/lat grid to another using delaunay triangulation
    
    Parameters
    ----------
    lon,lat : 2D arrays
        source grid
    data : 2D array
        data
    targetlon,targetlat : 2D arrays
        target grid
    mask : 2D boolean array, optional
        mask to select points from lon,lat,data to feed into interpolator
        by default, ~data.mask will be used
    m : Basemap instance, optional
        used to transform the coordinates.
        if not provided, a global Robinson projection will be used
    lon0 : int, optional
        longitude where the global Robinson projection should start
        by default, the minimum longitude (np.min(lon[mask])) will be used
    returnxy : bool
        whether to return the projected target coordinates x,y

    Returns
    -------
    [x,y,]targetdata
    """
    # merge and/or generate masks
    if mask is None:
        mask = np.ones(data.shape,bool)
    else:
        mask = np.asarray(mask,bool)
    try:
        mask &= ~data.mask
    except AttributeError:
        pass

    # check or make a basemap instance for transforming
    if m is None:
        if lon0 is None:
            lon0 = np.min(lon[mask])
        m = Basemap(projection='robin',lon_0=int(lon0))

    # do not crash if grids are identical
    if np.all(targetlon == lon) and np.all(targetlat == lat):
        print('Warning: Input and output grids are identical. Not interpolating.')
        if returnxy:
            x,y = m(lon,lat)
            return x,y,data
        else:
            return data

    # transform the coordinates
    x,y = m(lon[mask],lat[mask])
    xi,yi = m(targetlon,targetlat)
    
    # interpolate the data
    interp = NearestNDInterpolator(np.vstack((x,y)).T, data[mask])

    if returnxy:
        return xi,yi,np.ma.masked_invalid(interp(xi,yi))
    else:
        return np.ma.masked_invalid(interp(xi,yi))

