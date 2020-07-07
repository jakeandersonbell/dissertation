import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from processing.mastermap import get_mm
import pandas as pd
import geopandas as gpd


def get_img_dates(path, year):
    # Returns dictionary of image dates from imagery year folder
    # Folder names in dir
    tiles = [i for i in os.listdir(path + "/imagery/" + str(year)) if ".txt" not in i]

    dates = {}

    for tile in tiles:
        f = open(path + "/imagery/" + year + "/" + tile + ".txt", "r")
        content = f.read()

        result = content.find("Date Flown:")

        dates[tile] = datetime.strptime(content[result + 18:result + 28], "%Y-%m-%d")
    return dates


def get_filtered_fires(tile, data):
    # Filter fires data for incidents within a year of imagery flown date
    # Filters for total tile, not just data available for tile - there will be some extra fires retuurned that are not
    # represented by imagery
    filtered = data[data['DateOfCall'].between(img_dates[tile], img_dates[tile] + relativedelta(years=1))]
    min_x, max_x = (50 + int(tile[2])) * 10000, ((51 + int(tile[2])) * 10000) - 0.1
    min_y, max_y = (10 + int(tile[3])) * 10000, ((11 + int(tile[3])) * 10000) - 0.1

    filtered = filtered[filtered['Easting_m'].between(min_x, max_x)]
    filtered = filtered[filtered['Northing_m'].between(min_y, max_y)]

    gdf = gpd.GeoDataFrame(filtered, geometry=gpd.points_from_xy(filtered['Easting_m'], filtered['Northing_m']))
    return gdf


if __name__ == "__main__":

    import pickle

    parent_path = "D:/Dissertation"
    years = [str(i) for i in list(range(2017, 2019))]

    # read the fire data and convert date column to datetime
    data = pd.read_csv(parent_path + "/fire/filtered_fire.csv")
    data['DateOfCall'] = pd.to_datetime(data['DateOfCall'])

    # This loop links the fire incident with the building for each tile in each year for the available data
    for year in years:

        matches = {}

        img_dates = get_img_dates(parent_path, year)  # Date image taken
        print("Year:", year)
        matches[year] = {}

        # Completed 6th iteration

        for tile in img_dates.keys():
            print("Tile:", tile)

            points = get_filtered_fires(tile, data)

            points = points[points.within(gpd.read_file(
                parent_path + '/imagery/' + year + '/' + tile + '/' + tile + '.shp').iloc[0].geometry)]

            buildings = get_mm(tile, year)

            # Filter buildings for those that contain points
            for p in points.iloc:
                # Let's build spatial index for intersection points
                for b in buildings.iloc:
                    if p.geometry.intersects(b.geometry):
                        matches[year][p['IncidentNumber']] = b['TOID']
                        break

            print(len(matches[year].keys()))

        # Save the dict
        with open('building_point_match_' + year + '.pickle', 'wb') as dest:
            pickle.dump(matches, dest)

