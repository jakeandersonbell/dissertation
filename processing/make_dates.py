import pandas as pd
import geopandas as gpd
import numpy as np
import os


pos_buildings = gpd.read_file("data/positive_buildings_non_res.shp")
pos_buildings['DateOfCall'] = pd.to_datetime(pos_buildings['DateOfCall'])

neg_buildings = gpd.read_file("data/negative_buildings_non_res.shp")
neg_buildings['DateOfCall'] = ''

# Assign a random date from the positives - should have a similar distribution
for i, row in neg_buildings.iterrows():
    neg_buildings.at[i, 'DateOfCall'] = pos_buildings.loc[np.random.randint(len(pos_buildings))]['DateOfCall']

# Get the month int
neg_buildings['month_int'] = neg_buildings['DateOfCall'].apply(lambda x: x.month)
pos_buildings['month_int'] = pos_buildings['DateOfCall'].apply(lambda x: x.month)

neg_buildings['CalYear'] = neg_buildings['DateOfCall'].apply(lambda x: x.year)

# Convert back to string so they can be saved as shp
neg_buildings['DateOfCall'] = neg_buildings['DateOfCall'].apply(lambda x: x.strftime('%Y-%m-%d'))
pos_buildings['DateOfCall'] = pos_buildings['DateOfCall'].apply(lambda x: x.strftime('%Y-%m-%d'))

# Filter out some redundant columns
pos_buildings = pos_buildings[['TOID', 'Calculated', 'types', 'DateOfCall', 'CalYear', 'month_int', 'geometry']]
neg_buildings = neg_buildings[['TOID', 'Calculated', 'types', 'DateOfCall', 'CalYear', 'month_int', 'geometry']]

out_path = 'data/feature_tables/17_07'
pos_buildings.to_csv(os.path.join(out_path, 'positive.csv'))
neg_buildings.to_csv(os.path.join(out_path, 'negative.csv'))
