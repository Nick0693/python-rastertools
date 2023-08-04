from pathlib import Path
from typing import Tuple

import geopandas as gpd
import rasterio as rst
import shapely
import xarray as xr
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import box
from pyogrio import read_dataframe


def vectorize(in_raster : str) -> gpd.GeoDataFrame:
    """
    Polygonizes a raster based on contiguous areas of equal value. Output is a list of shapes
    in GeoJSON format.
    
    Parameters
    ----------
        in_raster: the file path of the input raster to be polygonized
        
    Returns
    ----------
        gdf: singleparts geodataframe with all polygons
    """
    
    with rst.Env():
        with rst.open(in_raster) as src:
            image = src.read(1)
            results = ({
                'properties': {
                    'raster_val': v
                }, 
                'geometry': s
            } for _, (s, v) in enumerate(
                shapes(image, mask=None, transform=src.transform)))

            gdf = gpd.GeoDataFrame.from_features(list(results))
            gdf = gdf.set_crs(src.crs)
            gdf['raster_val'] = gdf['raster_val'].astype('uint8')
            
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


def _get_raster_bounds(raster_path : str) -> shapely.geometry.polygon.Polygon:
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


def _get_shp_bounds(shp_path : str) -> shapely.geometry.polygon.Polygon:
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


def get_bounds(path : str) -> Tuple[shapely.geometry.polygon.Polygon, str]:
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