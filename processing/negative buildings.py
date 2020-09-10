"""Get building shape for negative instances"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from processing.mastermap import get_mm

parent_path = 'D:\Dissertation'

years = [str(i) for i in range(2012, 2019)]

frames = []
for year in years:
    path = os.path.join(parent_path, 'imagery', year, year + '_overview.shp')
    frame = gpd.read_file(path)
    frame['year'] = year
    frames.append(frame)

# Join imagery year and tile
# Extents of imagery years
overviews = gpd.GeoDataFrame(pd.concat(frames))
overviews.set_index(overviews)

points = gpd.read_file('negative_points.csv', GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")

points['year'] = np.empty((len(points), 0)).tolist()

point_year = gpd.sjoin(points, overviews, op='within')
point_year = point_year.drop('index_right', 1)

tiles = gpd.read_file(r'C:\Users\Jake\python_code\dissertation\data\grids\10k_grid\10k_grid.shp')
tiles = tiles.drop(['25K', 'ENGLAND', 'SCOTLAND', 'WALES'], 1)

point_year = gpd.sjoin(point_year, tiles, op='within')
point_year = point_year.drop('index_right', 1)
point_year = point_year.drop_duplicates()
point_year['year'] = point_year['year'].astype(str)
point_year['TILE_NAME'] = point_year['TILE_NAME'].astype(str)
point_year.crs = {'init': 'epsg:27700'}

matches = []

# Find the tile and year for imagery for each negative example
for year in years:
    for tile in [i for i in os.listdir(os.path.join(parent_path, 'MM', year)) if '.' not in i]:
        print(tile)
        buildings = get_mm(tile, year)

        points = point_year[point_year['year'] == year]
        points = points[points['TILE_NAME'] == tile]

        if not points.empty:
            building_match = gpd.sjoin(buildings, points, op='contains')
            matches.append(building_match)

matched = gpd.GeoDataFrame(pd.concat(matches))

matched.to_file(driver='ESRI Shapefile', filename='negative_buildings_non_res.shp')