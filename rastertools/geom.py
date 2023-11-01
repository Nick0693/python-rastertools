from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, List, Dict

import geopandas as gpd
import rasterio as rst
import xarray as xr
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from shapely.geometry import box
from pyogrio import read_dataframe

from .io import Raster

if TYPE_CHECKING:
    from shapely.geometry.polygon import Polygon 
    from numpy import ndarray


def vectorize(
        raster, 
        transform=None, 
        nodata=None, 
        crs=None, 
        count=1, 
        band=1, 
        **kwargs
    ) -> gpd.GeoDataFrame:
    """
    Polygonizes a raster based on contiguous areas of equal value. Output is a list of shapes
    in GeoJSON format. Only works for integer encoded arrays.
    
    Parameters
    ----------
        in_raster: the file path of the input raster to be polygonized
        
    Returns
    ----------
        gdf: singleparts geodataframe with all polygons
    """

    with Raster(raster, transform, nodata, crs, count, band) as src:
        image = src.read(**kwargs)
        results = ({
            'properties': {
                'raster_val': v
            }, 
            'geometry': s
        } for _, (s, v) in enumerate(
            shapes(image.array, mask=None, transform=image.transform)))

        gdf = gpd.GeoDataFrame.from_features(list(results))
        gdf = gdf.set_crs(image.crs)
        gdf['raster_val'] = gdf['raster_val'].astype('uint8')
        gdf = gdf[gdf['raster_val']!=image.nodata]
            
    return gdf


def merge(file_list : List[str], out_path : str) -> None:
    """_summary_

    Parameters
    ----------
        file_list (List[str]): _description_
        out_path (str): _description_
    """    
    src_files_to_mosaic = []
    for img in file_list:
        src = rst.open(img)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    meta = src.meta.copy()
    meta.update({
        'driver' : 'GTiff',
        'height' : mosaic.shape[1],
        'width' : mosaic.shape[2],
        'transform' : out_trans,
    })

    with rst.open(out_path, 'w', **meta) as dst:
        dst.write(mosaic.astype(rst.float32))


def intersect(path1 : str, path2 : str) -> Tuple:
    """"_summary_"

    Parameters
    ----------
        path1 (str): _description_
        path2 (str): _description_

    Returns
    ----------
        Tuple: _description_
    """    
    bbox1, filetype1 = get_bounds(path1)
    bbox2, filetype2 = get_bounds(path2)
    intersect = bbox1.intersection(bbox2)

    out_files = []
    for fp, filetype in zip((path1, path2), (filetype1, filetype2)):
        match filetype:
            case 'raster':
                with rst.open(fp) as src:
                    img, transform = mask(src, [intersect], crop=True)
                    meta = src.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": img.shape[1],
                    "width": img.shape[2],
                    "transform": transform
                })
                out_file = xr.DataArray(
                    img, dims=('x', 'y', 'band'), attrs=meta, name=Path(fp).stem, 
                    coords=[range(img.shape[0]), range(img.shape[1]), range(img.shape[2])]
                )
                out_files.append(out_file)

            case 'shapefile':
                gdf = read_dataframe(fp)
                out_file = gdf.clip(intersect)
                out_files.append(out_file)
    return tuple(out_files)


def _get_raster_bounds(raster_path : str) -> Polygon:
    """_summary_

    Parameters
    ----------
        raster_path (str): _description_

    Returns
    ----------
        shapely.geometry.polygon.Polygon: _description_
    """    
    with rst.open(raster_path) as src:
        bbox = box(*src.bounds)
    return bbox


def _get_shp_bounds(shp_path : str) -> Polygon:
    """_summary_

    Parameters
    ----------
        shp_path (str): _description_

    Returns
    ----------
        shapely.geometry.polygon.Polygon: _description_
    """    
    gdf = read_dataframe(shp_path)
    bbox = box(*gdf.total_bounds)
    return bbox


def get_bounds(path : str) -> Tuple[Polygon, str]:
    """_summary_

    Parameters
    ----------
        path (str): _description_

    Returns
    ----------
        Tuple[shapely.geometry.polygon.Polygon, str]: _description_
    """    
    match Path(path).suffix.lower():
        case '.tif' | '.tiff' | '.geotif' | '.geotiff':
            return _get_raster_bounds(path), 'raster'
        case '.shp' | '.geojson':
            return _get_shp_bounds(path), 'shapefile'
        

def resample(
    img_path : str, 
    res : float, 
    how : str = 'nearest'
    ) -> Tuple[ndarray, Dict]:
    """
    Resamples a raster image to the specified resolution

    Parameters
    ----------
        img_path: path to raster image
        res: desired output resolution
        how (optional): the resampling method 
    Returns
    ----------
        img: resampled raster image
        meta: metadata of resampled raster image
    """
    rs_alg = {
        'nearest' : Resampling.nearest,
        'bilinear' : Resampling.bilinear,
        'cubic' : Resampling.cubic,
        'average' : Resampling.average
    }
    
    with rst.open(img_path) as src:
        meta = src.meta.copy()
        scale_factor = res / src.res[0]
        img = src.read(
            out_shape=(
                src.count, 
                int(src.height / scale_factor), 
                int(src.width / scale_factor)
            ), resampling=rs_alg[how]
        )
        transform = src.transform * src.transform.scale(
            (src.width / img.shape[-1]),
            (src.height / img.shape[-2])
        )
        meta.update({
            'transform' : transform,
            'height' : img.shape[-2],
            'width' : img.shape[-1],
        })
    return img, meta


def warp(
    src_path : str, 
    dst_path : str, 
    ) -> Tuple[ndarray, Dict]:
    """
    Aligns a raster using a reference raster by projecting it and warping if a
    projection is not sufficient.

    Parameters
    ----------
        src_path: path to raster image to be aligned
        dst_path: path to the reference raster image
    Returns
    ----------
        img: aligned raster img 
        src_meta: metadata of aligned raster img
    """
    with rst.open(dst_path) as src:
        dst_img = src.read(1)
        dst_meta = src.meta.copy()
        bounds = src.bounds

    with rst.open(src_path) as src:
        src_img = src.read(1)
        src_meta = src.meta.copy()
        img, transform = reproject(
            source=src_img,
            destination=dst_img,
            src_transform=src_meta['transform'],
            src_crs=src_meta['crs'],
            src_nodata=src_meta['nodata'],
            dst_transform=dst_meta['transform'],
            dst_crs=dst_meta['crs']
        )
        height, width = img.shape

    # if a reprojection does not align the two arrays, the source will be warped instead
    if not img.shape == dst_img.shape:
        transform, width, height = calculate_default_transform(
            src_crs=src_meta['crs'],
            dst_crs=dst_meta['crs'],
            width=src_meta['width'],
            height=src_meta['height'],
            left=bounds[0],
            bottom=bounds[1],
            right=bounds[2],
            top=bounds[3],
            dst_width=dst_meta['width'],
            dst_height=dst_meta['height']
        )

    src_meta.update({
        'transform' : transform,
        'height' : height,
        'width' : width,
    })
            
    return img, src_meta