"""Use Googles Places API to get information about non-residential building use"""

import requests
import json
import os
import rasterio
import pyproj
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely import geometry
from shapely.ops import transform
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

api_key = open('processing/key.txt', 'r').read()


def bng_to_latlng(bounds):
    # The transformer to project BNG to latlong
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:27700'),  # source coordinate system
        pyproj.Proj(init='epsg:4326'))  # destination coordinate system

    p1, p2, p3, p4 = geometry.Point(bounds[0], bounds[1]), geometry.Point(bounds[0], bounds[3]), \
                     geometry.Point(bounds[2], bounds[3]), geometry.Point(bounds[2], bounds[1])

    polygon = geometry.Polygon([[p.x, p.y] for p in [p1, p2, p3, p4]])

    return transform(project.transform, polygon)


def places(latlng, place_type):
    # url variable store url
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json?"

    query = url + 'type=' + place_type + '&location=' + latlng + '&radius=100' + '&key=' + api_key
    r = requests.get(query)

    # json method of response object convert
    # json format data into python format data
    x = r.json()

    return x['results']


if __name__ == "__main__":
    positive_path = 'D:/Dissertation/labelled/building_mask/positive/all'

    types_dict = {}  # To get count of types

    building_dict = {}

    place_types = ['establishment', 'point_of_interest']

    frames = []

    # Loop through all positive labels
    for img in tqdm(os.listdir(positive_path)):
        with rasterio.open(os.path.join(positive_path, img)) as ds:
            poly = bng_to_latlng(ds.bounds)

            # plt.plot(poly.exterior.xy)

            # Get the centroid
            centroid = str(poly.centroid.xy[1][0]) + ',' + str(poly.centroid.xy[0][0])

            time.sleep(0.3)

            # Use establishment and point_of_interest
            for place_type in place_types:
                # google places for centroid
                for i in places(centroid, place_type):
                    point = geometry.Point(i['geometry']['location']['lng'], i['geometry']['location']['lat'])

                    # is the location within the polygon
                    if point.within(poly):
                        data = {'TOID': img.replace('.tif', ""),
                                'query_type': place_type,
                                'name': i['name'],
                                'address': i['formatted_address'],
                                'types': i['types']}
                        frames.append(pd.DataFrame(data))

                        for t in i['types']:
                            if t not in types_dict.keys():
                                types_dict[t] = 1
                            else:
                                types_dict[t] += 1

                        if img.replace('.tif', "") not in building_dict.keys():
                            building_dict[img.replace('.tif', "")] = i['types']
                        else:
                            building_dict[img.replace('.tif', "")] += i['types']

            # count for the types returned
