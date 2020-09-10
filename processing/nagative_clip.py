import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
import os


def clip_shapes(gdf, year_col, out_folder):
    parent_path = "D:/Dissertation"

    years = [str(i) for i in range(2012, 2019)]

    match = gpd.read_file('data/positive_buildings_non_res.shp')
    for year in years:
        # Open the point match dict file
        match = gdf[gdf[year_col] == int(year)]
        # convert to df
        # match_df = pd.DataFrame(match[year].items(), columns=['IncidentNumber', 'TOID'])  # Make it into a df

        img_dates = get_img_dates(year)  # Date image was taken
        for tile in img_dates:
            # Make a year folder
            if not os.path.exists(parent_path + "/labelled/" + year + "/" + tile):
                os.mkdir(parent_path + "/labelled/" + year + "/" + tile)

            # For each tile clip out the buildings
            with rasterio.open(parent_path + '/imagery/' + year + '/' + tile + '/' + tile + '_full.tif') as src:
                out_meta = src.meta.copy()
                print(src.bounds)
                for i in range(len(gdf)):
                    try:
                        out_image, out_transform = mask(src, [gdf['geometry'].iloc[i]], crop=True)

                        if np.max(out_image) == 0.0:
                            print("no data")

                        out_meta.update({"driver": "GTiff",
                                         "height": out_image.shape[1],
                                         "width": out_image.shape[2],
                                         "transform": out_transform}
                                        )

                        with rasterio.open(parent_path + "/" + out_folder + "/" + year + "/" + tile + "/" +
                                           str(gdf.iloc[i]['TOID']) + ".tif", "w+", **out_meta) as dest:
                            dest.write(out_image)
                    except:
                        continue