"""Matches fire incidents to building geometry"""

import os
import pickle
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from processing.mastermap import get_mm
import pandas as pd
import geopandas as gpd
from tqdm import tqdm


def get_img_dates(year):
    # Returns dictionary of image dates from imagery year folder
    # Folder names in dir
    tiles = [i for i in os.listdir("D:/Dissertation/imagery/" + str(year)) if "." not in i]

    dates = {}

    for tile in tiles:
        f = open("D:/Dissertation/imagery/" + year + "/" + tile + ".txt", "r")
        content = f.read()

        result = content.find("Date Flown:")

        dates[tile] = datetime.strptime(content[result + 18:result + 28], "%Y-%m-%d")
    return dates


def get_filtered_fires(tile, data, year):
    # Filter fires data for incidents within a year of imagery flown date
    # Filters for total tile, not just data available for tile - there will be some extra fires retuurned that are not
    # represented by imagery
    img_dates = get_img_dates(year)

    filtered = data[data['DateOfCall'].between(img_dates[tile], img_dates[tile] + relativedelta(years=1))]
    min_x, max_x = (50 + int(tile[2])) * 10000, ((51 + int(tile[2])) * 10000) - 0.1
    min_y, max_y = (10 + int(tile[3])) * 10000, ((11 + int(tile[3])) * 10000) - 0.1

    filtered = filtered[filtered['Easting_m'].between(min_x, max_x)]
    filtered = filtered[filtered['Northing_m'].between(min_y, max_y)]

    gdf = gpd.GeoDataFrame(filtered, geometry=gpd.points_from_xy(filtered['Easting_m'], filtered['Northing_m']))
    gdf.crs = {'init': 'epsg:27700'}
    return gdf


def incident_toid_match():
    # Returns a dataframe of IncidentNumber - TOID matches
    parent_path = "D:/Dissertation"
    years = [str(i) for i in list(range(2012, 2019))]

    # read the fire data and convert date column to datetime
    data = pd.read_csv(parent_path + "/fire/filtered_fire.csv")
    data['DateOfCall'] = pd.to_datetime(data['DateOfCall'])

    frames = []

    # This loop links the fire incident with the building for each tile in each year for the available data
    for year in tqdm(years):

        img_dates = get_img_dates(year)  # Date image taken
        print("Year:", year)
        # matches[year] = {}

        for tile in img_dates.keys():
            print("Tile:", tile)

            points = get_filtered_fires(tile, data, year)

            points = points[points.within(gpd.read_file(
                parent_path + '/imagery/' + year + '/' + tile + '/' + tile + '.shp').iloc[0].geometry)]

            buildings = get_mm(tile, year)

            if buildings.empty or points.empty:
                continue

            # Filter buildings for those that contain points
            frame = gpd.sjoin(buildings, points)[['TOID', 'CalculatedAreaValue', 'IncidentNumber',
                                                  'DateOfCall', 'CalYear', 'geometry']]
            frame['img_year'] = year
            # for p in points.iloc:
            #     # Let's build spatial index for intersection points
            #     for b in buildings.iloc:
            #         if p.geometry.intersects(b.geometry):
            #             frames.append(pd.DataFrame({'IncidentNumber': p['IncidentNumber'],
            #                                         'TOID': b['TOID'],
            #                                         'year': year,
            #                                         'geometry': b.geometry}))
            #             break
            frames.append(frame)

    return pd.concat(frames)


def places_toid_match(positive_toid):
    place = pd.read_csv('data/places/positive_place_toid.csv')
    # remove duplicates and aggregate types
    place = place.groupby('TOID')['types'].apply(lambda x: list(set(x))).reset_index()
    positive_toid['TOID'] = positive_toid['TOID'].astype('int64')

    return positive_toid.merge(place, on='TOID')


def neg_toid_match():
    # Takes in points and returns point - toid matches
    years = [str(i) for i in list(range(2012, 2019))]

    points = gpd.read_file("data/places/negative_points.csv", GEOM_POSSIBLE_NAMES="geometry",
                           KEEP_GEOM_COLUMNS="NO")

    points.crs = {'init': 'epsg: 27700'}

    frames = []

    # This loop links the fire incident with the building for each tile in each year for the available data
    for year in tqdm(years):
        img_dates = get_img_dates(year)  # Date image taken
        print("Year:", year)
        for tile in img_dates.keys():
            print("Tile:", tile)

            # # Filter points by tile
            # points = points[points.within(gpd.read_file(
            #     parent_path + '/imagery/' + year + '/' + tile + '/' + tile + '.shp').iloc[0].geometry)]

            buildings = get_mm(tile, year)

            if points.empty:
                continue

            # Filter buildings for those that contain points
            # for p in points.iloc:
            #     # Let's build spatial index for intersection points
            #     for b in buildings.iloc:
            #         if p.geometry.intersects(b.geometry):
            #             frames.append(pd.DataFrame({}))
            #             matches[year][p['field_1']] = [b['TOID'], b['geometry']]
            #             break
            frame = gpd.sjoin(buildings, points)
            frame['img_year'] = year

            frames.append(frame)

    return pd.concat(frames)

