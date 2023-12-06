from typing import Tuple, List
import numpy as np
import pandas as pd
from pyogrio import read_dataframe

from .io import Raster

__all__ = ['point_query']


def point_query(
    raster,
    shapefile : str,
    transform = None, 
    nodata : float = None, 
    crs : str = None, 
    count : int = 1, 
    band : int | List = None,
    columns : List = None,
    limits : Tuple = None,
    mask : np.ndarray = None,
    **kwargs
) -> Tuple[np.ndarray]:
    
    with Raster(raster, transform, nodata, crs, count, band) as src:
        image = src.read(**kwargs)
        x, y, m = read_points(shapefile, src.transform, src.shape, limits)
        data = image.array
    
    if mask is not None:
        data = np.where(mask, data, np.nan)
    
    df = pd.DataFrame(
        np.hstack([m.reshape(-1, 1), np.stack([data[:, yi, xi] for yi, xi in zip(y, x)])]), 
        columns=columns
    )
    return df


def read_points(shapefile, transform, data_shape, limits=None):
    features = read_dataframe(shapefile)
    xy = np.zeros((2, len(features)))
    measured = np.zeros((len(features)))
    for i, feature in features.iterrows():
        xy[:, i] = feature.geometry.x, feature.geometry.y
        measured[i] = feature['bathymetry']

    xi, yi = ~transform * xy
    xi = np.round(xi).astype(int)
    yi = np.round(yi).astype(int)

    inside = xi < data_shape[1]
    inside &= yi < data_shape[0]
    inside &= xi > 0
    inside &= yi > 0

    valid = np.isfinite(measured)
    if limits is not None:
        with np.errstate(invalid="ignore"):
            valid &= measured >= limits[0]
            valid &= measured <= limits[1]

    good = inside & valid
    measured_valid = measured[good]
    xi_valid = xi[good]
    yi_valid = yi[good]

    df = pd.DataFrame(dict(depth=measured_valid, ii=xi_valid, jj=yi_valid))
    df = df.groupby(["ii", "jj"])["depth"].median().reset_index()
    x, y, m = [arr.flatten() for arr in np.split(df.values, indices_or_sections=[1, 2], axis=1)]
    x = x.astype('int')
    y = y.astype('int')
    return x, y, m