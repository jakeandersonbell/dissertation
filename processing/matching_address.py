"""This script matches address UPRN to buildings"""

import os
import pandas as pd
import csv
import pickle
import geopandas as gpd
from shapely.geometry import Point
from processing.mastermap import get_mm
from processing.clipping import get_features

df = pd.DataFrame(pd.read_pickle('building_point_match_2013.pickle')['2013'].items())

for i in [str(j) for j in range(2014, 2019)]:
    df = df.append(pd.DataFrame(pd.read_pickle('building_point_match_' + i + '.pickle')[i].items()))

df = df.rename(columns={0: 'IncidentNumber',
                        1: 'TOID'})
df = df.astype(str)


lab_path = 'D:/Dissertation/labelled/building_mask/'  # Path of labelled imagery

toid = []

# Make a list of all TOIDs in the directory - obtained from file names
for year in os.listdir(lab_path):
    for tile in os.listdir(lab_path + year):
        for id in os.listdir(lab_path + year + '/' + tile):
            toid.append(id[:-4])  # slice out the file extension

# Filter incdnt-TOID match by labelled features
df = df[df['TOID'].isin(toid)]


# Reading in AddressBase
# Some missing TOIDs
add_path = 'D:/Dissertation/addressbase/'  # Path of addressbase

add_files = [i for i in os.listdir(add_path) if '.csv' in i]

add = pd.read_csv(add_path + add_files[0])

with open('addressbase-plus-header.csv', newline='') as f:
    header = list(csv.reader(f))[0]
header[0] = 'UPRN'
add.columns = header

toid_os = ['osgb' + str(i) for i in toid]

add = add[add['OS_TOPO_TOID'].isin(toid_os)]

# Reading in MM buildings
mm_path = 'D:/Dissertation/MM/'

# Go through all buildings and find addressbase points that intersect them
for year in os.listdir(mm_path):
    print(year)
    add_match_dict = {}
    for tile in [n for n in os.listdir(mm_path + year) if '.txt' not in n]:
        print(tile)
        buildings = get_mm(tile, year)
        buildings = buildings[buildings['TOID'].isin(toid)]

        # 5km addressbase tiles associated with 10km mm tiles
        tile_add = [i for i in os.listdir(add_path) if i[2] == tile[2] and i[4] == tile[3]]

        add_points = pd.read_csv(add_path + tile_add[0])
        geometry = [Point(xy) for xy in zip(add_points[add_points.columns[7]],
                                            add_points[add_points.columns[8]])]

        add_points = gpd.GeoDataFrame(add_points, geometry=geometry)

        for b in buildings.iloc:
            for p in add_points.iloc:
                if p.geometry.intersects(b.geometry):
                    print("m")
                    add_match_dict[p[0]] = b['TOID']

    with open('uprn_toid_' + year, 'wb') as dest:
        pickle.dump(add_match_dict, dest)



