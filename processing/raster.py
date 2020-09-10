import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.plot import show
import matplotlib.pyplot as plt
import fiona
import os
import geopandas as gpd
from shapely.geometry import box, Polygon
from contextlib import contextmanager


def get_features(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


parent_path = "D:/Dissertation/DSM"

# years = [str(i) for i in range(2012, 2019)]

for tile in [i for i in os.listdir(parent_path) if '.' not in i]:

    img_paths = [os.path.join(parent_path, tile, i) for i in os.listdir(os.path.join(parent_path, tile))]

    src_files_to_mosaic = [rasterio.open(fp) for fp in img_paths]

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                     }
                    )

    with rasterio.open(os.path.join(parent_path, tile, tile + "_full.tif"), "w", **out_meta) as dest:
        dest.write(mosaic)
        print(tile)

# out_image, out_transform = mask(mosaic, [shape1], crop=True)

# with rasterio.open(path + images[0]) as src:
#     show(src)
#     # gdf.plot()
#     out_image, out_transform = mask(src, [shape1], crop=True)
#     out_meta = src.meta

# with rasterio.open(parent_path + "/clip_out.tif", "w", **out_meta) as dest:
#     dest.write(out_image)

# with fiona.open(parent_path + "/MM/2012/TQ35/mastermap-topo_3551698_0.gpkg", "r") as shape:
#     shapes = [feature["geometry"] for feature in shape if feature['properties']]

# gdf = gpd.read_file(parent_path + "/MM/2012/TQ35/mastermap-topo_3551698_0.gpkg", layer='TopographicArea')
# gdf = gdf[gdf['DescriptiveGroup'] == 'Building']
#
# shape = get_features(gdf)
# shape1 = gdf.iloc[10].geometry
# buffer = 2
# bbox = box(shape1.bounds[0] - buffer, shape1.bounds[1] - buffer, shape1.bounds[2] + buffer, shape1.bounds[3] + buffer)
