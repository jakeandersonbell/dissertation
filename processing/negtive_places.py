"""Functionality to get Google Places data for negative incident class"""

import pandas as pd
import geopandas as gpd
import os
import pyproj
import requests
import time
import ast
from shapely.ops import transform
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
from processing.places_api import places
from processing.mastermap import get_mm


def random_points(polygon, num_points):
    # Return randomly generated points within a polygon
    min_x, min_y, max_x, max_y = polygon.bounds

    points = []

    for i in range(num_points):
        random_point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if random_point.within(poly):
            points.append(random_point)

    return points


def get_negative_places():
    place_types = ['establishment', 'point_of_interest']

    frames = []

    # Keep a record of tiles completed
    completed_tiles = []

    for i, tile in counts_5k.iterrows():

        if i < 4:
            continue

        # The number of requests to be made - as a function of positive examples in the region
        if int(tile['COUNT']) < 3:
            n_req = 1
        else:
            n_req = int(int(tile['COUNT']) / 3)

        poly = tiles_5k.loc[tiles_5k['TILE_NAME'] == tile['TILE']].geometry.iloc[0]

        points = random_points(poly, n_req)  # Get a random point for each request to be made

        for p in points:
            time.sleep(0.3)

            p = transform(project.transform, p)  # transform to latlng

            p_string = str(p.xy[1][0]) + ',' + str(p.xy[0][0])  # construct the string to go in the query

            place_type = place_types[np.random.randint(0, 2)]  # pick place type at random

            for j in places(p_string, place_type, api_key):
                point = Point(j['geometry']['location']['lng'], j['geometry']['location']['lat'])

                # is the location within the polygon
                data = {'query_type': place_type,
                        'geometry': j['geometry'],
                        'name': j['name'],
                        'address': j['formatted_address'],
                        'types': j['types']}
                frames.append(pd.DataFrame.from_dict(data, orient='index').transpose())

        completed_tiles.append(tile)

    frame = pd.concat(frames)
    return frame, completed_tiles


def convert_geom(s):
    location = ast.literal_eval(s)['location']
    location = Point(location['lng'], location['lat'])
    return transform(project.transform, location)


if __name__ == "__main__":
    api_key = open('misc/key.txt', 'r').read()

    tile_counts = pd.read_csv('data/tile_counts.csv', sep=';')
    tiles_5k = gpd.read_file('data/grids/5k_grid/OSGB_Grid_5km.shp')
    buildings = gpd.read_file('data/positive_buildings_non_res.shp')
    build_df = pd.DataFrame(buildings)
    # build_df.groupby('type').count()

    # Get counts of incidents within 5k tiles
    poly = tiles_5k.iloc[4]['geometry']
    counts_5k = gpd.sjoin(buildings, tiles_5k)
    counts_5k = counts_5k.groupby('TILE_NAME')['TOID'].count().reset_index()
    counts_5k.columns = ['TILE', 'COUNT']

    project = pyproj.Transformer.from_proj(
            pyproj.Proj(init='epsg:27700'),  # source coordinate system
            pyproj.Proj(init='epsg:4326'))  # destination coordinate system

    data = get_negative_places()[0]

    project = pyproj.Transformer.from_proj(
            pyproj.Proj(init='epsg:4326'),  # source coordinate system
            pyproj.Proj(init='epsg:27700'))  # dest coordinate system

    data['geometry'] = data['geometry'].apply(lambda x: convert_geom(x))

    gdf = gpd.GeoDataFrame(data, crs='EPSG:27700', geometry=data.geometry)

    gdf.to_csv('negative_points.csv')



