from typing import Tuple, Dict, List
from pathlib import Path
from operator import itemgetter
import warnings

import rasterio as rst
import numpy as np
import xarray as xr
from rasterio.enums import Resampling

__all__ = ['s2stack']

BANDMAP = {
    '10' : {
        2 : 'B2', 
        3 : 'B3', 
        4 : 'B4', 
        8 : 'B8'
    },
    '20' : {
        5 : 'B5', 
        6 : 'B6', 
        7 : 'B7',
        10 : 'B8A',
        11 : 'B11', 
        12 : 'B12'
    },
    '60' : {
        1 : 'B1', 
        9 : 'B9'
    }
}

INDEXMAP = {'10' : 0, '20' : 1, '60' : 2}

RSALG = {
    'nearest' : Resampling.nearest,
    'bilinear' : Resampling.bilinear,
    'cubic' : Resampling.cubic,
    'average' : Resampling.average
}

def s2stack(
    safe_path : str,
    out_path : str | Path = None,
    resolution : List | str | int = 10, 
    resampling_method : str = 'nearest', 
    band_first : bool = True,
    normalize : bool = False,
    **kwargs
) -> xr.DataArray:
    """
    Read-and-write function for Sentinel-2 .SAFE.zip files to stack individual bands into a single
    image with an option to normalize.

    Parameters:
        safe_path (str): 
            Path to the Sentinel-2 .SAFE.zip file.
        out_path (str | Path, optional): 
            Full output path name for saving the image. If None, the image is not saved. Defaults to None.
        resolution (List | str, optional): 
            Spatial resolution(s) (m) of bands to use. Will return bands matching the lowest (i.e. 60>10) specified
            resolution but will defer to higher resolution products where available. E.g. a resolution value of 
            [10, 20] will return the native 10m bands (2, 3, 4, 8) but will resample and return the 20m products
            for the remaining bands. Defaults to 10.
        resampling_method (str, optional): 
            Method used to resample bands of lower resolution if multiple are specified. Defaults to 'nearest'.
        band_first (bool, optional): 
            Bands are returned as the first dimension in the array. Defaults to True.
        normalize (bool, optional): 
            Normalize values between (0, 1). Additional keywords can be specified to determine which values to
            use as the min and max. Defaults to False.

    Returns:
        xr.DataArray 
            xarray with image data and metadata as attributes.
    """    

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=rst.errors.NotGeoreferencedWarning)
        with rst.open(safe_path) as src:
            subdatasets = src.subdatasets
    
    if isinstance(resolution, str) or isinstance(resolution, int):
        resolution = [resolution]
    resolution = [str(r) for r in resolution]
    
    bands = {}
    for n, res_index in enumerate(resolution):
        with rst.open(subdatasets[INDEXMAP[res_index]]) as src:
            indexes = [[d.split(',')[0] for d in src.descriptions].index(v)+1 for v in BANDMAP[res_index].values()]
            target_res = min([int(r) for r in resolution])
            
            if src.res[0] > target_res:
                scale_factor = target_res / src.res[0]
                img = src.read(
                    indexes=indexes,
                    out_shape=(
                        len(indexes), 
                        int(src.height / scale_factor), 
                        int(src.width / scale_factor)
                    ), resampling=RSALG[resampling_method]
                )
            else:
                img = src.read(indexes)
            
            if n==0:
                meta = src.meta.copy()

            for key, band in zip(BANDMAP[res_index].keys(), img):
                bands[key] = band
    
    axis = 0 if band_first else -1
    img = np.stack(list(dict(sorted(bands.items())).values()), axis=axis)
    meta.update({'driver' : 'GTiff', 'count' : img.shape[axis]})

    desc = {}
    for d in [BANDMAP[r] for r in resolution]:
        desc.update(d)
    desc = list(desc.values())

    if normalize:
        img = band_scaling(img, **kwargs)
        meta.update({'dtype' : 'float32'})

    if out_path is not None:
        with rst.open(out_path, 'w', **meta) as dst:
            for i, d in enumerate(desc, start=1):
                dst.set_band_description(i, d) 
            dst.write(img)
    name = Path(out_path).stem if out_path is not None else '_'.join(itemgetter(0, 5, 2)(Path(safe_path).stem.split('_')))
    array = xr.DataArray(
        img, 
        coords={'bands' : desc, 'y' : range(img.shape[1]), 'x' : range(img.shape[2])}, 
        dims=('bands', 'y', 'x'),
        attrs=meta,
        name=name
    )
    return array


def band_scaling(
    img : np.ndarray,
    limits : Tuple | Dict = (1, 99),
    heuristic : bool = False,
    n_samples : int = 1e5,
) -> np.ndarray:
    """
    Band scaling using either percentile cutoffs (e.g. 2nd and 98th percentile) common to visualization
    or manually defined bounds for each band.

    Parameters
    ----------
        img (np.ndarray): 
            The array/image to be rescaled. Can be either 2D or 3D.
        limits (Dict, optional): 
            Either a dictionary of manually defined bounds for each band given in the format {band : (lower, upper)}
            or a tuple of the percentile cutoffs. Defaults to (2, 98).
        heuristic (bool, optional): 
            Reduce the memory overhead by calculating statistics from a random subset of pixels. Defaults to False.
        n_samples (int, optional): 
            Number of sample pixels to use when applying heuristics to find the limits. Defaults to 1e5.

    Returns
    ----------
        np.ndarray: Rescaled image as float in range (0, 1)
    """    

    def heuristic_bounds(arr : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y, x = arr.shape
        coords = np.unique(np.array([(np.random.randint(y), np.random.randint(x)) for _ in range(int(n_samples))]), axis=0)
        vals = np.array([arr[coords[n][0], coords[n][1]] for n in range(len(coords))]).reshape(-1)
        top = np.percentile(vals, upper)
        bottom = np.percentile(vals, lower)
        return bottom, top

    if isinstance(limits, tuple):
        lower, upper = limits
    
    if img.ndim==2:
        img = img[np.newaxis, ...]
    
    # Normalization based on the lower and upper nth percentile
    img = np.nan_to_num(img, nan=0.0)
    out_img = np.zeros(img.shape, dtype='float32')
    for n, band in enumerate(img):
        if isinstance(limits, dict):
            bottom, top = limits[n+1]
        elif heuristic:
            bottom, top = heuristic_bounds(band)
        else:
            bottom = np.percentile(band[band>0], lower)
            top = np.percentile(band[band>0], upper)
        
        out_img[n] = band.astype('float32') - bottom
        out_img[n] = out_img[n] / (top-bottom)

    out_img[out_img > 1] = 1
    out_img[out_img < 0] = 1e-5
    out_img[img==0] = 0
    
    return out_img