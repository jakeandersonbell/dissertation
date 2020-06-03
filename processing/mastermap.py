import fiona
import os
import geopandas as gpd


def get_mm(tile, year):
    #  Function to find and open the mastermap geopackage
    parent_path = "D:/Dissertation/"
    path = parent_path + 'MM/' + str(year) + '/' + tile + '/'

    # There are multiple files with the tilename so filter by ones that end in gpkg
    filename = [i for i in os.listdir(path) if ".gpkg" in i[-5:]][0]

    # Open geopackage and only keep building polys
    gdf = gpd.read_file(path + filename, layer='TopographicArea')
    return gdf[gdf['DescriptiveGroup'] == 'Building']
