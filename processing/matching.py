import fiona
import os
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
from processing.mastermap import get_mm
from rtree import index
import pandas as pd
import geopandas as gpd
import numpy as np


def get_img_dates(path, year):
    # Returns dictionary of image dates from imagery year folder
    # Folder names in dir
    tiles = [i for i in os.listdir(path + "/imagery/" + str(year)) if ".txt" not in i]

    dates = {}

    for tile in tiles:
        f = open(parent_path + "/imagery/" + year + "/" + tile + ".txt", "r")
        content = f.read()

        result = content.find("Date Flown:")

        dates[tile] = datetime.strptime(content[result + 18:result + 28], "%Y-%m-%d")
    return dates


def get_filtered_fires(tile, data):
    # Filter fires data for incidents within a year of imagery flown date
    filtered = data[data['DateOfCall'].between(img_dates[tile], img_dates[tile] + relativedelta(years=1))]
    min_x, max_x = (50 + int(tile[2])) * 10000, ((51 + int(tile[2])) * 10000) - 0.1
    min_y, max_y = (10 + int(tile[3])) * 10000, ((11 + int(tile[3])) * 10000) - 0.1

    filtered = filtered[filtered['Easting_m'].between(min_x, max_x)]
    filtered = filtered[filtered['Northing_m'].between(min_y, max_y)]

    gdf = gpd.GeoDataFrame(filtered, geometry=gpd.points_from_xy(filtered['Easting_m'], filtered['Northing_m']))
    return gdf


if __name__ == "__main__":

    parent_path = "D:/Dissertation"
    years = [str(i) for i in list(range(2012, 2019))]

    # read the fire data and convert date column to datetime
    data = pd.read_csv(parent_path + "/fire/filtered_fire.csv")
    data['DateOfCall'] = pd.to_datetime(data['DateOfCall'])

    matches = {}

    for year in years:
        img_dates = get_img_dates(parent_path, year)
        print("Year:", year)
        matches[year] = {}

        for tile in img_dates.keys():
            print("Tile:", tile)

            points = get_filtered_fires(tile, data)

            buildings = get_mm(tile, year)

            # Filter buildings for those that contain points
            for p in points.iloc:
                for b in buildings.iloc:
                    if p.geometry.intersects(b.geometry):
                        matches[year][p['IncidentNumber']] = b['TOID']
                        break



# for date in img_dates.values():



# find all fire events within a year of imagery



# get the mastermap


# get the MM shapes that the points lie in


# cut out the buildings from imagery





"""
# No need to pass "layer='etc'" if there's only one layer
with fiona.open('test.gpkg', layer='layer_of_interest') as layer:
    for feature in layer:
        print(feature['geometry'])
"""