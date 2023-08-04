from typing import Dict, Tuple

import numpy as np
import rasterio as rst
from pyproj import CRS


def ascii_to_tif(
    asc : str, 
    epsg : int, 
    header : int = 6, 
    how : str = 'center'
) -> Tuple[np.ndarray, Dict]: 
    """
    Converts ESRI ASCII rasters to geotifs
    
    Parameters:
    ----------
        asc: full name of the ascii file (file path)
        epsg: EPSG number for projection
        header (optional): number of meta lines included in the file
        how (optional): how the xy-offset is defined in the header.
            for center-origin, use 'center', and None (or anything else) for 'corner'
        
    Returns:
    ----------
        arr: ascii data in an array
        meta: metadata of the image
    
    """
    with open(asc) as src:
        data = src.read().splitlines()

    asc_meta = {}
    for line in data[:header]:
        e = line.split(' ')
        asc_meta[e[0]] = float(e[1])
        
    if how=='center':
        asc_meta['yllcorner'] = asc_meta['yllcenter'] + asc_meta['nrows']*asc_meta['cellsize']
        asc_meta['xllcorner'] = asc_meta['xllcenter']

    meta = ({
        'driver' : 'GTiff',
        'dtype' : 'float32',
        'nodata' : asc_meta['nodata_value'],
        'width' : int(asc_meta['ncols']),
        'height' : int(asc_meta['nrows']),
        'count' : 1,
        'crs' : CRS.from_epsg(epsg),
        'transform' : rst.Affine(asc_meta['cellsize'], 0.0, asc_meta['xllcorner'], 
                                 0.0, -asc_meta['cellsize'], asc_meta['yllcorner'])

    })

    arr = np.zeros([int(meta['height']), int(meta['width'])])
    for n, line in enumerate(data[header:]):
        line = line.split(' ')
        line = [float(l) for l in line if l]
        arr[n] = line
        
    return arr, meta