from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, List, Dict

import geopandas as gpd
import rasterio as rst
import xarray as xr
import numpy as np
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.merge import merge as merge_rasters
from shapely import box, Polygon
from pyogrio import read_dataframe

from .io import Raster


def vectorize(
    raster : Path | str | xr.DataArray | np.ndarray, 
    transform : rst.Affine = None, 
    crs : str | rst.CRS = None,
    nodata : float = None, 
    band : int = 1,
    name : str = 'raster_val',
    **kwargs
) -> gpd.GeoDataFrame:
    """
    Polygonizes a raster based on contiguous areas of equal value. Outputs a geodataframe of
    polygons with values corresponding to the raster value.

    Parameters:
        raster (Path | str | xr.DataArray | np.ndarray): 
            Input raster to be polygonized.
        transform (rst.Affine, optional): 
            Affine transformation of the array. If raster is a path this may be used to overwrite the
            default transformation, otherwise it is only required for DataArrays or ndarrays. Defaults to None.
        crs (str | rst.CRS, optional): 
            Coordinate system of the array. If raster is a path this may be used to overwrite the
            default coordinate system, otherwise it is only required for DataArrays or ndarrays. Defaults to None.
        nodata (float, optional): 
            Values to ignore when converting to polygon. Defaults to None.
        band (int, optional): 
            The band/index to vectorize. Indexes are 1-based so the first band will be 1. Defaults to 1.
        name (str, optional): 
            Column name of the vectorized raster values. Defaults to 'raster_val'.

    Returns:
        gpd.GeoDataFrame: 
            Singleparts geodataframe with all polygons.
    """    

    with Raster(raster, transform=transform, crs=crs, nodata=nodata, bands=band) as src:
        image = src.read(**kwargs)
        results = ({
            'properties': {
                'raster_val': v
            }, 
            'geometry': s
        } for _, (s, v) in enumerate(
            shapes(image.data, mask=None, transform=image.transform)))

        gdf = gpd.GeoDataFrame.from_features(list(results)).set_crs(image.crs)
        gdf[name] = gdf[name].astype(image.data.dtype)
        gdf = gdf[gdf[name]!=image.nodata]
            
    return gdf


def resample(
    raster : Path | str,
    res : float, 
    resampling : str = 'nearest',
    **kwargs
    ) -> xr.DataArray:
    """
    Resamples the input raster to the target spatial resolution using the specified resampling method.

    Parameters:
        raster (Path | str): 
            Input raster to be resampled.
        res (float): 
            Target spatial output resolution in georeferenced units.
        how (str, optional): 
            Resampling method to use when up- or downsampling the image. Defaults to 'nearest'.

    Returns:
        xr.DataArray: 
            DataArray with the resampling input image, and metadata as attributes.
    """    
    rs_method = {
        'nearest' : Resampling(0),
        'bilinear' : Resampling(1),
        'cubic' : Resampling(2),
        'cubic_spline' : Resampling(3),
        'lanczos' : Resampling(4),
        'average' : Resampling(5),
        'mode' : Resampling(6),
        'gauss' : Resampling(7),
        'max' : Resampling(8),
        'min' : Resampling(9),
        'med' : Resampling(10),
        'q1' : Resampling(11),
        'q3' : Resampling(12),
        'sum' : Resampling(13),
        'rms' : Resampling(14)
    }
    
    with rst.open(raster) as src:
        profile = src.meta.copy()
        scale_factor = res / src.res[0]
        img = src.read(
            out_shape=(
                src.count, 
                int(src.height / scale_factor), 
                int(src.width / scale_factor)
            ), 
            resampling=rs_method[resampling],
            **kwargs
        )
        transform = src.transform * src.transform.scale(
            (src.width / img.shape[-1]),
            (src.height / img.shape[-2])
        )
    profile.update({
        'transform' : transform,
        'height' : img.shape[-2],
        'width' : img.shape[-1],
        'count' : img.shape[0] if np.ndim(img)==3 else 1
    })
    
    dims = ('band', 'y', 'x') if np.ndim(img)==3 else ('y', 'x')
    data = xr.DataArray(
        img, dims=dims, attrs=profile, name=Path(raster).stem, 
        coords=[range(d) for d in img.shape]
    )

    return data


def merge(filelist : List[Path | str]) -> xr.DataArray:   
    """
    Merges a list of rasters into a mosaic in painter's order.

    Parameters:
        filelist (List[Path | str]): 
            List of raster paths to mosaic.

    Returns:
        xr.DataArray: 
            DataArray with the mosaicked images, and metadata as attributes
    """     
    src_files_to_mosaic = []
    for n, img in enumerate(filelist):
        src = rst.open(img)
        if n==0:
            profile = src.profile.copy()
        src_files_to_mosaic.append(src)

    mosaic, transform = merge_rasters(src_files_to_mosaic)
    for src in src_files_to_mosaic:
        src.close()

    profile.update({
        'driver' : 'GTiff',
        'height' : mosaic.shape[1],
        'width' : mosaic.shape[2],
        'transform' : transform,
    })

    dims = ('band', 'y', 'x') if np.ndim(mosaic)==3 else ('y', 'x')
    data = xr.DataArray(
        mosaic, dims=dims, attrs=profile, name=Path(filelist[0]).stem, 
        coords=[range(d) for d in mosaic.shape]
    )
    return data


def intersect(filepaths : List[Path | str]) -> Tuple:
    """
    _summary_

    Parameters:
        filepaths (List[Path  |  str]): 
            _description_

    Returns:
        Tuple: 
            _description_
    """    
    def get_bounds(path : str) -> Tuple[Polygon, str]:   
        def _get_raster_bounds(raster_path : str) -> Polygon: 
            with rst.open(raster_path) as src:
                bbox = box(*src.bounds)
                crs = src.crs
            return crs, bbox

        def _get_shp_bounds(shp_path : str) -> Polygon:
            gdf = read_dataframe(shp_path)
            bbox = box(*gdf.total_bounds)
            return gdf.crs, bbox

        match Path(path).suffix.lower():
            case '.tif' | '.tiff' | '.geotif' | '.geotiff':
                return *_get_raster_bounds(path), 'raster'
            case '.shp' | '.geojson':
                return *_get_shp_bounds(path), 'shapefile'
            
    crs, bbox, filetypes = zip(*[get_bounds(fp) for fp in filepaths])
    intersect = gpd.GeoSeries(bbox[0], crs=crs[0])
    for b in bbox[1:]:
        intersect = intersect.intersection(b)

    out_files = []
    for fp, filetype in zip(filepaths, filetypes):
        match filetype:
            case 'raster':
                with rst.open(fp) as src:
                    img, transform = mask(src, intersect, crop=True)
                    profile = src.profile.copy()

                profile.update({
                    "driver": "GTiff",
                    "height": img.shape[1],
                    "width": img.shape[2],
                    "transform": transform
                })
                dims = ('band', 'y', 'x') if np.ndim(img)==3 else ('y', 'x')
                data = xr.DataArray(
                    img, dims=dims, attrs=profile, name=Path(fp).stem, 
                    coords=[range(d) for d in img.shape]
                )
                out_files.append(data)

            case 'shapefile':
                gdf = read_dataframe(fp)
                out_file = gdf.clip(intersect)
                out_files.append(out_file)
    return tuple(out_files)
  

def warp(
    src_img : Path | str | xr.DataArray | np.ndarray, 
    ref_img : Path | str | xr.DataArray | np.ndarray,
    src_transform : rst.Affine = None,
    src_nodata : float = None,
    src_crs : str = None,
    src_bands : int | List = None,
    ref_transform : rst.Affine = None,
    ref_crs : str = None,
) -> xr.DataArray:
    """
    Aligns a raster using a reference raster by projecting it and warping if a projection is not sufficient.

    Parameters:
        src_img (Path | str | xr.DataArray | np.ndarray): 
            Input raster to be aligned.
        ref_img (Path | str | xr.DataArray | np.ndarray): 
            Reference raster.
        src_transform (rst.Affine, optional): 
            Transform of the source raster. If raster is a path this may be used to overwrite the
            default transformation, otherwise it is only required for DataArrays or ndarrays. 
            Defaults to None.
        src_nodata (float, optional): 
            Nodata value of the source raster. If raster is a path this may be used to overwrite the
            default nodata value. Defaults to None.
        src_crs (str, optional): 
            Coordinate system of source raster. If raster is a path this may be used to overwrite the
            default coordinate system, otherwise it is only required for DataArrays or ndarrays. 
            Defaults to None.
        src_bands (int | List, optional): 
            The bands/indexes to use. Indexes are 1-based so the first band will be 1. If None, uses all bands.
            Defaults to None.
        ref_transform (rst.Affine, optional): 
            Transform of the reference raster. If raster is a path this may be used to overwrite the
            default transformation, otherwise it is only required for DataArrays or ndarrays. 
            Defaults to None.
        ref_crs (str, optional): 
            Coordinate system of reference raster. If raster is a path this may be used to overwrite the
            default coordinate system, otherwise it is only required for DataArrays or ndarrays. 
            Defaults to None.

    Returns:
        xr.DataArray: 
            DataArray with aligned image, and metadata as attributes.
    """    
    with Raster(
        ref_img, 
        transform=ref_transform, 
        crs=ref_crs, 
    ) as ref:
        ref_img = ref.read()

    with Raster(
        src_img, 
        transform=src_transform, 
        nodata=src_nodata, 
        crs=src_crs, 
        bands=src_bands
    ) as src:
        src_img = src.read()

    out_shape = (src_img.shape[0], *ref_img.shape) if np.ndim(src_img.data)==3 else ref_img.shape
    img, transform = reproject(
        source=src_img.data,
        destination=np.zeros(out_shape),
        src_transform=src_img.transform,
        src_crs=src_img.crs,
        src_nodata=src_img.nodata,
        dst_transform=ref_img.transform,
        dst_crs=ref_img.crs
    )
    height, width = img.shape
    left, bottom, right, top = ref_img.bounds

    # if a reprojection does not align the two arrays, the source will be warped instead.
    # this is likely caused by integer rounding in the transformation.
    if not img.shape[-2:] == ref_img.shape:
        transform, width, height = calculate_default_transform(
            src_crs=src_img.crs,
            dst_crs=ref_img.crs,
            width=src_img.width,
            height=src_img.height,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            dst_width=ref_img.width,
            dst_height=ref_img.height
        )

    img = img[np.newaxis, ...] if np.ndim(img)==2 else img
    meta = {
        'driver' : 'GTiff',
        'count' : img.shape[0],
        'dtype' : src_img.data.dtype,
        'transform' : transform,
        'height' : height,
        'width' : width,
        'nodata' : src_img.nodata,
        'crs' : ref_img.nodata
    }

    data = xr.DataArray(
        img, dims=('band', 'y', 'x'), attrs=meta, name=Path(src_img).stem, 
        coords=[range(d) for d in img.shape]
    )
            
    return data