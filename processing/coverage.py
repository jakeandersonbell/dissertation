# Produces extent shapfiles for each tile for each year

import rasterio
import os
from shapely.geometry import mapping, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import fiona

parent_path = "D:/Dissertation/imagery/"
years = [str(i) for i in list(range(2012, 2019))]

for year in years:
    tiles = [i for i in os.listdir(parent_path + year) if ".txt" not in i]
    for tile in tiles:
        img_paths = [parent_path + year + "/" + tile + "/" + tile[0:2].lower() + "/" + i
                     for i in os.listdir(parent_path + year + "/" + tile + "/" + tile[0:2].lower()) if ".jpg" in i]
        rasters = [rasterio.open(i) for i in img_paths]
        polys = [Polygon([[i.bounds[0], i.bounds[1]], [i.bounds[0], i.bounds[3]],
                                   [i.bounds[2], i.bounds[3]], [i.bounds[2], i.bounds[1]]]) for i in rasters]
        poly = unary_union(polys)
        # plt.plot(poly.exterior.xy)

        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
        }

        # Write a new Shapefile
        with fiona.open(parent_path + year + "/" + tile + '/' + tile + '.shp', 'w', 'ESRI Shapefile', schema) as c:
            c.write({
                'geometry': mapping(poly),
                'properties': {'id': 1},
            })

