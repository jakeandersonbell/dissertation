"""This script generates the negative image features"""

import pickle
import os
import fiona
import pandas as pd
import geopandas as gpd
import rasterio
import json
import matplotlib.pyplot as plt
from rasterio.mask import mask
from processing.mastermap import get_mm
from processing.clipping import get_features
from tqdm import tqdm

# How many labelled examples do we have already?
lab_path = 'D:/Dissertation/labelled/building_mask'
mm_path = 'D:/Dissertation/MM'
parent_path = 'D:/Dissertation'
img_path = 'D:/Dissertation/imagery'

for year in os.listdir(os.path.join(lab_path, 'positive')):
    count = 0
    for tile in os.listdir(os.path.join(lab_path, 'positive', year)):
        count += len(os.listdir(os.path.join(lab_path, 'positive', year, tile)))
    print(year + ': ' + str(count))

for year in os.listdir(os.path.join(lab_path, 'negative')):
    count = 0
    for tile in os.listdir(os.path.join(lab_path, 'negative', year)):
        count += len(os.listdir(os.path.join(lab_path, 'negative', year, tile)))
    print(year + ': ' + str(count))

# 2013: 804
# 2014: 734
# 2015: 840
# 2016: 365
# 2017: 46
# 2018: 228


years = [str(i) for i in range(2013, 2019)]
# Get images for no fires
for year in years:
    print(year)

    # Check if a folder exists for that tile
    if not os.path.exists(os.path.join(lab_path, "negative", year)):
        os.mkdir(os.path.join(lab_path, "negative", year))

    # Open the match dictionary
    with open('building_point_match_' + year + '.pickle', 'rb') as src:
        toid = pickle.load(src)

    # Count number of positive features for that year
    count = 0
    for tile in os.listdir(os.path.join(lab_path, 'positive', year)):
        count += len(os.listdir(os.path.join(lab_path, 'positive', year, tile)))

    # Tile loop

    # Get the imagery extent
    year_shape = gpd.read_file(os.path.join(img_path, year, year + "_overview.shp"))

    for tile in os.listdir(os.path.join(lab_path, 'positive', year)):
        print(tile)

        # Clip the imagery if folder does not already exist
        if not os.path.exists(os.path.join(lab_path, "negative", year, tile)):
            os.mkdir(os.path.join(lab_path, "negative", year, tile))

            count = len(os.listdir(os.path.join(lab_path, 'positive', year, tile)))

            buildings = get_mm(tile, year)

            # open the MM tile to subset the filtered toids
            print("Filtering building from TOID match...")
            buildings = buildings[~buildings['TOID'].isin(toid[year].values())]

            buildings = buildings.sample(frac=1)

            shapes = buildings['geometry'].head(count * 10)

            # Clipping
            with rasterio.open(parent_path + '/imagery/' + year + '/' + tile + '/' + tile + '_full.tif') as src:
                out_meta = src.meta.copy()
                print(src.bounds)

                tick = 0  # To be used for the iterations
                index = 0  # To be used to index the geometries, will allow us to skip geometries that raise exceptions

                # Use while so that when an exception is raised we do not produce less examples
                while tick < count * 10:
                    if True in list(year_shape.contains(buildings['geometry'].iloc[index])):
                        try:
                            out_image, out_transform = mask(src, [buildings['geometry'].iloc[index]], crop=True)

                            out_meta.update({"driver": "GTiff",
                                             "height": out_image.shape[1],
                                             "width": out_image.shape[2],
                                             "transform": out_transform}
                                            )

                            with rasterio.open(parent_path + "/labelled/building_mask/negative/" + year + "/" + tile + "/" +
                                               buildings.iloc[index]['TOID'] + ".tif", "w+", **out_meta) as dest:
                                dest.write(out_image)
                            tick += 1
                            index += 1
                        except ValueError:
                            index += 1
                            pass
                    else:
                        index += 1
