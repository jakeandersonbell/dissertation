"""This script cuts the positive fire buildings out of the imagery"""

import pickle
import os
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.mask import mask
from rasterio.plot import show
from datetime import datetime
from dateutil.relativedelta import relativedelta
from processing.mastermap import get_mm
from processing.matching import get_img_dates, get_filtered_fires


def get_features(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][b]['geometry'] for b in range(len(buildings))]


if __name__ == "__main__":
    parent_path = "D:/Dissertation"

    # Open dict of fire-building shape matches
    # with open('building_point_match_2013.pickle', 'rb') as file:
    #     match_2013 = pickle.load(file)
    #
    data = pd.read_csv(parent_path + "/fire/filtered_fire.csv")
    data['DateOfCall'] = pd.to_datetime(data['DateOfCall'])
    #
    years = [str(i) for i in range(2012, 2019)]
    # # img_dates = get_img_dates(parent_path, year)
    #
    # match_df = pd.DataFrame(match_2013['2013'].items(), columns=['IncidentNumber', 'TOID'])
    #
    # tiles = ['TQ29']

    for year in years:
        # Open the point match dict file
        with open('building_point_match_' + year + '.pickle', 'rb') as file:
            match = pickle.load(file)

        match_df = pd.DataFrame(match[year].items(), columns=['IncidentNumber', 'TOID'])  # Make it into a df

        img_dates = get_img_dates(parent_path, year)  # Date image was taken
        for tile in img_dates:
            # Make a year folder
            if not os.path.exists(parent_path + "/labelled/" + year + "/" + tile):
                os.mkdir(parent_path + "/labelled/" + year + "/" + tile)

            points = get_filtered_fires(tile, data)  # Get the associated fire points
            match_df_1 = match_df[match_df['IncidentNumber'].isin(points['IncidentNumber'])]

            buildings = get_mm(tile, year)
            buildings = buildings[buildings['TOID'].isin(match_df_1['TOID'])]

            shapes = get_features(buildings)

            # For each tile clip out the buildings
            with rasterio.open(parent_path + '/imagery/' + year + '/' + tile + '/' + tile + '_full.tif') as src:
                out_meta = src.meta.copy()
                print(src.bounds)
                for i in range(len(shapes)):
                    out_image, out_transform = mask(src, [shapes[i]], crop=True)

                    # show(out_image)

                    out_meta.update({"driver": "GTiff",
                                     "height": out_image.shape[1],
                                     "width": out_image.shape[2],
                                     "transform": out_transform}
                                    )

                    with rasterio.open(parent_path + "/labelled/" + year + "/" + tile + "/" +
                                       buildings.iloc[i]['TOID'] + ".tif", "w+", **out_meta) as dest:
                        dest.write(out_image)

    parent_path = "D:/Dissertation/imagery/"
    # get the imagery coverage shapefile for each year
    for year in years:
        tiles = [i for i in os.listdir(parent_path + str(year)) if '.txt' not in i]

        gdfs = [gpd.read_file(parent_path + year + '/' + i + '/' + i + '.shp') for i in tiles]

        gdf = pd.concat(gdfs)

        # gdf.plot()

        gdf.to_file(driver='ESRI Shapefile', filename=parent_path + year + '/' + year + "_overview.shp")