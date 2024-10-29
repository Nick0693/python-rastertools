# rastertools
`rastertools` is a package used to pipeline preprocessing steps for rasters. The package is built around the Raster class, which helps keep track of metadata when reading georeferenced raster files. The package contains additional earth observation-specific functions to unpack and process optical data.

## Installation
rastertools can be installed with pip:

`pip install https://github.com/Nick0693/python-rastertools.git`

## Documentation

### EO functionality

`s2stack` is used for stacking (and resampling) Sentinel-2 data from a .SAFE.zip file. Either a single or multiple spatial resolutions may be specified. The data will be resampled to the lowest resolution specified.

`s2img = s2stack(MMM_MSIXXX_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_.SAFE.zip, resolution=[10, 20])`

For a full description of input arguments I refer to the docstring.

### The Raster class

Most functions leverage the `Raster` class, which keeps track of various attributes pertaining to geographical data like shape, affine transformation, bounds, coordinate system, etc. The class may also be called directly using either a path to a geotiff, an ndarray, or an xarray

```
# from a path
with rastertools.Raster(r'path/to/file.tif') as src:
    image = src.read()

# from an array, with associated transform and crs
with rastertools.Raster(array, transform=transform, crs=crs) as src:
    image = src.read()

# from an xarray, with metadata as attribute
with rastertools.Raster(dataarray) as src:
    image = src.read()
```

The geographic information about the object can then be retrieved by using its attributes, e.g. `image.data` or `image.transform`

The list of attributes stored in the object includes:
* data (the array values)
* shape (shape of the array)
* height (the height of the array)
* width (the width i)
* transform (the affine transformation)
* crs (the coordinate system)
* nodata (the nodata value)
* res (the spatial resolution in georeferenced units)
* bounds (the spatial extent of the raster)
* dtype (data type)
* meta (driver, count, dtype, height, width, transform, crs, nodata)

### Functions

The package contains several functions for handling common geometric operations on raster data.

`vectorize` is used to convert discrete rasters into polygons. Takes a path to a geotiff, an ndarray, or an xarray as input.

`resample` is used to up- or downsample data to a desired spatial resolution. Takes a path to a geotiff, an ndarray, or an xarray as input.

`merge` performs a union of a list of rasters. Takes a list of filepaths as input.

`intersect` extracts the intersecting area between all input files in their original format. Takes a list of filepaths as input.