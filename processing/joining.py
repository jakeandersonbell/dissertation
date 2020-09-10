"""Joining fire and master map data"""

import pandas as pd
import geopandas as gpd
import os
import csv
import numpy as np


def fire_building_join():
    # Read in fires data
    fire_dir = 'D:/Dissertation/fire'
    fires = pd.concat([pd.read_csv(os.path.join(fire_dir, i)) for i in os.listdir(fire_dir) if 'LFB' in i])
    fires = fires[['IncidentNumber', 'DateOfCall', 'CalYear', 'TimeOfCall', 'HourOfCall',
                  'IncidentGroup', 'StopCodeDescription', 'SpecialServiceType',
                  'PropertyCategory', 'PropertyType', 'AddressQualifier', 'Easting_m', 'Northing_m']]

    # Filter building match for non residential
    all_match = pd.read_csv('all_building_match.csv')
    all_match = all_match.drop(columns=['Unnamed: 0'])
    all_match.columns = ['IncidentNumber', 'TOID']

    #
    non_res = pd.read_csv('non_res_addresses2.csv')
    # 1811 unique toids
    non_res = non_res.groupby('TOID')['types'].apply(lambda x: list(set(x))).reset_index(name='types')

    buildings = gpd.read_file('data/old/positive_buildings.shp')

    buildings = pd.DataFrame(buildings)

    buildings['TOID'] = buildings['TOID'].astype(np.int64)

    # 1811 unique toids
    buildings = buildings[buildings['TOID'].isin(non_res['TOID'].unique())]

    result = pd.merge(buildings, non_res, on='TOID')

    result2 = result.merge(all_match, on='TOID', how='inner')

    # Convert the types as they are mixed
    fires['IncidentNumber'] = fires['IncidentNumber'].astype(str)
    result2['IncidentNumber'] = result2['IncidentNumber'].astype(str)

    result3 = result2.merge(fires, on='IncidentNumber', how='inner')

    buildings = gpd.GeoDataFrame(result3)

    buildings['types'] = str(buildings['types'])

    buildings.to_csv("data/positive_buildings_non_res2.csv")

    buildings = gpd.read_file("data/positive_buildings_non_res.shp")


pos_buildings = gpd.read_file("data/feature_tables/17_07/positive.csv", GEOM_POSSIBLE_NAMES="geometry",
                              KEEP_GEOM_COLUMNS="NO")

neg_buildings = gpd.read_file("data/feature_tables/17_07/negative.csv", GEOM_POSSIBLE_NAMES="geometry",
                              KEEP_GEOM_COLUMNS="NO")

demographic = gpd.read_file("data/joined_demographic.shp")

positive = gpd.sjoin(pos_buildings, demographic, op='within')

negative = gpd.sjoin(neg_buildings, demographic, op='within')

positive.to_csv("data/feature_tables/20_07/positive.csv")
negative.to_csv("data/feature_tables/20_07/negative.csv")
