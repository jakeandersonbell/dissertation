import fiona
import os
from datetime import datetime

parent_path = "D:/Dissertation"

os.listdir(parent_path + "/imagery/2012")


def get_img_dates(path):
    #
    # Folder names in dir
    tiles = [i for i in os.listdir(parent_path + "/imagery/2012") if ".txt" not in i]

    dates = {}

    for tile in tiles:
        f = open(parent_path + "/imagery/2012/" + tile + ".txt", "r")
        content = f.read()

        result = content.find("Date Flown:")
        dates[tile] = datetime.strptime(content[result + 18:result + 28], "%Y-%m-%d").date()
    return dates

# get tile date

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