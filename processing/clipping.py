"""This script cuts the positive buildings out of the imagery and DSM"""

import pickle
import os
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.mask import mask
import rasterio.crs
from shapely.geometry import Point
from rasterio.plot import show
import numpy as np
from shutil import copy, move
from datetime import datetime
from dateutil.relativedelta import relativedelta
from processing.mastermap import get_mm
from processing.matching import get_img_dates, get_filtered_fires


def get_features(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][b]['geometry'] for b in range(len(gdf))]


def clip_shapes(gdf, out_folder, year_col='img_year', buff=False, fil_no_data=True, years=True, imagery=True):
    # Clips aerial imagery using input gdf shapes
    # Filters by year if this is relevant - i.e. positive samples
    # Buffers shapes proportionate to area/minimum-rectangle-area ratio
    # Will not write no data (black) images fil_no_data can be used to remove no data images from
    # DataFrame
    if imagery:
        parent_path = "D:/Dissertation/imagery"
    else:
        parent_path = "D:/Dissertation/DSM"

    # If imagery organised by year
    if years:
        years = [str(i) for i in range(2012, 2019)]
    else:
        years = ['tiles']

    if not year_col:
        frame = gdf

    for year in years:
        removed = 0
        # filter frame by image year - only required for the positives
        if year_col:
            frame = gdf[gdf[year_col] == year]
        if imagery:
            img_dates = get_img_dates(year)  # Date image was taken
        else:
            img_dates = os.listdir(os.path.join(parent_path, year))
        for tile in img_dates:
            # For each tile clip out the buildings

            with rasterio.open(os.path.join(parent_path, year, tile, tile + '_full.tif'), 'r+') as src:
                src.crs = rasterio.crs.CRS({"init": "epsg:27700"})
                out_meta = src.meta.copy()
                for i, row in frame.iterrows():
                    try:
                        shape = row['geometry']
                        if buff:
                            # Buffer by (short edge * 0.25) * actual area / minimum rectangle area
                            # Ensures that narrow/branching buildings are not dominated by buffer
                            x, y = shape.minimum_rotated_rectangle.exterior.coords.xy
                            edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])),
                                           Point(x[1], y[1]).distance(Point(x[2], y[2])))
                            buff_amount = (min(edge_length) * 0.25) * shape.area / shape.minimum_rotated_rectangle.area
                            shape = shape.buffer(buff_amount)
                        out_image, out_transform = mask(src, [shape], crop=True)

                        if not imagery:
                            # Difference relative to ground
                            out_image -= np.min(out_image)

                        if np.max(out_image) == 0:
                            # If image is black remove row from data frame and skip iteration
                            removed += 1
                            if fil_no_data:
                                gdf = gdf[gdf['index'] != row['index']]
                        elif imagery and np.mean(out_image) < 20:  # Do not check mean on DSM
                            # If image is black remove row from data frame and skip iteration
                            removed += 1
                            if fil_no_data:
                                gdf = gdf[gdf['index'] != row['index']]
                        else:
                            out_meta.update({"driver": "GTiff",
                                             "height": out_image.shape[1],
                                             "width": out_image.shape[2],
                                             "transform": out_transform}
                                            )

                            img_path = os.path.join(out_folder, str(gdf.iloc[i]['index']) + ".tif")

                            with rasterio.open(img_path, "w+", **out_meta) as dest:
                                dest.write(out_image)
                    except Exception as e:
                        print(e)
                        continue
        print(str(removed), 'no data images removed for', str(year))
    return gdf


def annual_overviews():
    parent_path = "D:/Dissertation/imagery/"

    years = [str(i) for i in range(2012, 2019)]
    # get the imagery coverage shapefile for each year
    for year in years:
        tiles = [i for i in os.listdir(parent_path + str(year)) if '.txt' not in i]

        gdfs = [gpd.read_file(parent_path + year + '/' + i + '/' + i + '.shp') for i in tiles]

        gdf = pd.concat(gdfs)

        # gdf.plot()

        gdf.to_file(driver='ESRI Shapefile', filename=parent_path + year + '/' + year + "_overview.shp")


def move_data(path):
    years = [str(i) for i in range(2012, 2019)]

    for i in ['positive', 'negative']:
        for j in years:
            for fol in os.listdir(os.path.join(path, i, j)):
                for fil in os.listdir(os.path.join(path, i, j, fol)):
                    copy(os.path.join(path, i, j, fol, fil), os.path.join(path, i, 'all', fil))


def remove_no_data(path, gdf, col='index'):
    # Removes gdf entries not represented by imagery in the path
    for file in os.listdir(os.path.join(path)):
        with rasterio.open(os.path.join(path, file)) as src:
            maxp = np.max(src.read(1))
        if maxp == 0:
            os.remove(os.path.join(path, file))
            gdf = gdf = gdf[gdf['index'] != file.split('.')[0]]


def remove_missing_img_data(path, gdf, col='index'):
    # Removes gdf entries not represented by imagery in the path
    existing = [i.split('.')[0] for i in os.listdir(path)]
    return gdf[gdf[col].isin(existing)]


if __name__ == "__main__":
    positive = gpd.read_file("data/feature_tables/20_07_2/positive.csv", GEOM_POSSIBLE_NAMES="geometry",
                             KEEP_GEOM_COLUMNS="NO")

    negative = gpd.read_file("data/feature_tables/20_07_2/negative.csv", GEOM_POSSIBLE_NAMES="geometry",
                             KEEP_GEOM_COLUMNS="NO")

    negative['CalYear'] = negative['CalYear'].astype(str)

    clip_shapes(negative, 'year', 'labelled/final/negative')
    clip_shapes(positive, 'year', 'labelled/final/positive')

    move_data('labelled/final')

    remove_no_data('labelled/final')

    path = 'D:/Dissertation/labelled/test'

    stats = []

    for file in os.listdir(os.path.join(path)):
        with rasterio.open(os.path.join(path, file)) as src:
            res = [np.max(src.read(1)), np.min(src.read(1)),
                   np.mean(src.read(1)), np.median(src.read(1))]
            stats.append(res)

    path = 'D:/Dissertation/DSM'

    for i in [i for i in os.listdir(path) if '.' not in i]:
        for j in os.listdir(os.path.join(path, i, 'tq')):
            copy(os.path.join(path, i, 'tq', j), os.path.join(path, j))

    tiles = ['TQ' + str(i) for i in range(7, 60)]

    for i in [i for i in os.listdir(path) if '.asc' in i]:
        tile = 'TQ' + str(i[2]) + str(i[4])
        if not os.path.exists(os.path.join(path, tile)):
            os.mkdir(os.path.join(path, tile))
        move(os.path.join(path, i), os.path.join(path, tile, i))





