from typing import Tuple, Dict, List
from pathlib import Path
import warnings

import rasterio as rst
import numpy as np
from rasterio.enums import Resampling


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

def stack(
    safe_path : str,
    out_path : str | Path,
    resolution : List | str = '10', 
    resampling_method : str = 'nearest', 
    band_first : bool = True
) -> Tuple[np.ndarray, Dict]:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=rst.errors.NotGeoreferencedWarning)
        with rst.open(safe_path) as src:
            subdatasets = src.subdatasets
    
    if isinstance(resolution, str):
        resolution = [resolution]
    
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
    if out_path is not None:
        with rst.open(out_path, 'w', **meta) as dst:
            dst.write(img)
    return img, meta