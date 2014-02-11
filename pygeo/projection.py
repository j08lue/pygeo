"""Tools for quickly finding a suitable projection for the region or coordinates specified"""
import numpy as np
import scipy.io
import urllib2
import xml.etree.ElementTree as ET

from . import geo

def select_projection(lonlim,latlim,round=None,**kwargs):
    """Select ideal projection based on `lonlim` and `latlim`
    
    Parameters
    ----------
    lonlim,latlim : tuples / arrays
        coordinates confining the region
    round : int, optional
        round coordinates to this number of decimals
    kwargs : keyword arguments, optional
        included in the returned dictionary

    Credits
    -------
    MATLAB/R2013a/toolbox/map/mapdisp/private/regionmap.m
    function selectProjection(latlim, lonlim)
    """
    # handle wrapped lonlim
    while lonlim[1] < lonlim[0]:
        lonlim[1] += 360

    meanlon = np.mean(lonlim)
    meanlat = np.mean(latlim)
    if round is not None:
        meanlon = np.round(meanlon)
        meanlat = np.round(meanlat)
        factor = 10.**round
        lonlim[0],latlim[0] = np.floor(lonlim[0]*factor)/factor,np.floor(latlim[0]*factor)/factor
        lonlim[1],latlim[1] = np.ceil(lonlim[1]*factor)/factor,np.ceil(latlim[1]*factor)/factor

    if np.all(latlim == [-90,90]) and np.diff(lonlim)[0] > 360:
        # entire globe
        basemapkwargs = dict(
                projection = 'robin', # Robinson
                lon_0 = meanlon)
        
    elif np.max(np.abs(latlim)) < 20:
        # straddles equator, but doesn't extend into extreme latitudes
        basemapkwargs = dict(
                projection = 'merc',
                llcrnrlon=lonlim[0],urcrnrlon=lonlim[1],
                llcrnrlat=latlim[0],urcrnrlat=latlim[1],
                lat_ts = meanlat)
        
    elif (np.abs(np.diff(latlim)[0]) <= 90 
            and np.abs(np.sum(latlim)) > 20 
            and np.max(np.abs(latlim)) < 90):
        # doesn't extend to the pole, not straddling equator
        basemapkwargs = dict(
                projection = 'eqdc', # equidistant conic
                lat_1=latlim[0],
                lat_2=latlim[1],
                lon_0=meanlon,
                lat_0=meanlat)

        basemapkwargs['height'] = geo.haversine(meanlon,latlim[0],meanlon,latlim[1])
        if latlim[0] > 0:
            # on northern hemisphere, use width at southern boundary
            basemapkwargs['width'] = geo.haversine(lonlim[0],latlim[0],lonlim[1],latlim[0])
        else:
            basemapkwargs['width'] = geo.haversine(lonlim[0],latlim[1],lonlim[1],latlim[1])
        
    elif np.abs(np.diff(latlim)[0]) < 85 and np.max(np.abs(latlim)) < 90:
        # doesn't extend to the pole, not straddling equator
        basemapkwargs = dict(
                projection = 'sinu', # Sinusoidal
                lon_0 = meanlon)
        
    elif np.max(latlim) == 90 and np.min(latlim) >= 84:
        basemapkwargs = dict(
                projection = 'npstere', # North-Polar Stereographic
                boundinglat = np.min(latlim),
                lon_0 = meanlon)
        
    elif np.min(latlim) == -90 and np.max(latlim) <= -80:
        basemapkwargs = dict(
                projection = 'spstere', # South-Polar Stereographic
                boundinglat = np.max(latlim),
                lon_0 = meanlon)
        
    elif np.max(np.abs(latlim)) == 90 and np.abs(np.diff(lonlim)[0]) < 180:
        basemapkwargs = dict(
                projection = 'poly', # Polyconic
                llcrnrlon=lonlim[0],urcrnrlon=lonlim[1],
                llcrnrlat=latlim[0],urcrnrlat=latlim[1],
                lon_0=meanlon,
                lat_0=meanlat)
        
    elif np.max(np.abs(latlim)) == 90 and np.abs(np.diff(latlim)[0]) < 90:
        basemapkwargs = dict(
                projection = 'aeqd', # Azimuthal Equidistant
                llcrnrlon=lonlim[0],urcrnrlon=lonlim[1],
                llcrnrlat=latlim[0],urcrnrlat=latlim[1],
                lon_0=meanlon,
                lat_0=meanlat)

    else:
        basemapkwargs = dict(
                projection = 'mill', # Miller
                llcrnrlon=lonlim[0],urcrnrlon=lonlim[1],
                llcrnrlat=latlim[0],urcrnrlat=latlim[1])

    return dict(basemapkwargs,**kwargs)


def regions_from_matlab_file(regionsfile):
    """Read regions coordinate data from MATLAB file
    e.g. MATLAB/R2013a/toolbox/map/mapdisp/private/regions.mat
    
    Parameters
    ----------
    regionsfile : str, optional
        path to regions file
    """
    regions = {}

    regionsmat = scipy.io.loadmat(regionsfile,struct_as_record=True,squeeze_me=True)
    try:
        regionsmat = regionsmat['worldRegions']
    except:
        pass

    regions = {}
    for i in xrange(len(regionsmat)):
        regions[regionsmat['name'][i]] = dict(lonlim=regionsmat['lonlim'][i],latlim=regionsmat['latlim'][i])

    return regions


def regionmap(query,regionsfile,download=False,username='demo',**basemapkwargs):
    """Get map projection parameters for a given region/country in the World
    based on information from <http://geonames.org>
    
    Parameters
    ----------
    query : str or dict
        country name or code
        or dictionary e.g. dict(capital='Berlin')
        values are not case sensitive
    regionsfile : str, optional
        file from which to get the country information
    download : bool
        whether to download the information from <http://api.geonames.org/countryInfo>
        in that case, please consider using your own
    username : str, optional
        username for geonames.org 
        can be obtained from <http://www.geonames.org/login>
    basemapkwargs : dict
        keyword arguments included with the returned dictionary

    Example
    -------
    basic functionality

        bmkwargs = regionmap_geonamesdotorg('Germany')

    to include your keyword arguments with the returned dictionary, use, e.g.,

        bmkwargs = regionmap_geonamesdotorg('DE',resolution='i')
    """

    if download:
        url = 'http://api.geonames.org/countryInfo?{}'.format(username)
        response = urllib2.urlopen(url)
        tree = ET.parse(response)
    else:
        tree = ET.parse(regionsfile)

    root = tree.getroot()

    if isinstance(query,basestring):
        query = dict(countryName=query,countryCode=query)

    found = False
    for country in root.iterfind('country'):
        for key,value in query.iteritems():
            try:
                if country.findtext(key).lower() == value.lower():
                    found = True
                    break
            except AttributeError:
                continue
        if found:
            print('Found country \'{}\''.format(country.findtext('countryName')))
            lonlim = np.array((country.findtext('west'),country.findtext('east')),'f8')
            latlim = np.array((country.findtext('south'),country.findtext('north')),'f8')
            break
    if not found:
        raise ValueError('Nothing found for your query \'{}\'. Check your keys and values.'.format(query))

    return select_projection(lonlim=lonlim,latlim=latlim,**basemapkwargs)


def regionmap_matlab(query,regionsfile,**basemapkwargs):
    """Get map projection parameters for a given country in the World
    based on information from Matlab's country information in, e.g.
    /usr/common/MATLAB/R2013a/toolbox/map/mapdisp/private/regions.mat    
    
    Parameters
    ----------
    region : str
        region name
    regionsfile : str
        file from which to get the region definitions
    basemapkwargs : dict
        keyword arguments included with the returned dictionary

    Note
    ----
    If scipy.io.loadmat has problems loading the data because of 
    some strange binary format, simply re-save the file (like, `save(load(filename))`)
    """
    # get regions information from matlab file
    regions = regions_from_matlab_file(regionsfile)

    # run through regions to find case-insensitive match
    found = False
    for name,lim in regions.iteritems():
        if name.lower() == query.lower():
            lonlim = lim['lonlim']
            latlim = lim['latlim']
            found = True
            break
    if not found:
        raise KeyError('No region limits found for \'{}\'.'.format(query))
        
    return select_projection(lonlim=lonlim,latlim=latlim,**basemapkwargs)
