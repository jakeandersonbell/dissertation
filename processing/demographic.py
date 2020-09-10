# Joining relevant demographic data
# Data was downloaded from DigiMap and is attributed to ONS output areas

import pandas as pd
import geopandas as gpd
import os

census_path = 'data/census'

keys = [i.split('-')[1].split('_')[0] for i in os.listdir(census_path) if '.' not in i]

frames = []
for i in os.listdir(census_path):
    if i.split('.')[-1] not in ['txt', 'zip', 'csv', 'cpg', 'dbf', 'prj', 'shp', 'shx']:
        try:
            frames.append(gpd.read_file(os.path.join(census_path, i, i + '.gpkg'), layer='Output Area'))
        # Expecting some packages to not ave layers and only output area
        except ValueError:
            frames.append(gpd.read_file(os.path.join(census_path, i, i + '.gpkg')))

# https://www.nomisweb.co.uk/census/2011/lc2101ew
# ethnicity by age
ethnicity = pd.read_csv('data/census/bulk.csv')
# Just keep ethnicity totals
keep = [i for i in ethnicity.columns if 'Sex: All persons; Age: All categories: Age' in i]
keep = [i for i in keep if 'Total' in i]
# Along with area codes
keep = ['geography code'] + keep
# Filter columns to keep
ethnicity = ethnicity[keep]
# Simplify the column names
ethnicity.columns = [i.replace('Sex: All persons; Age: All categories: Age; Ethnic Group: ', '') for i in
                     ethnicity.columns]
ethnicity.columns = [i.replace('; measures: Value', '') for i in ethnicity.columns]

# Get percentages
ethnicity = ethnicity.set_index('geography code')
ethnicity = ethnicity.div(ethnicity.sum(axis=1), axis=0).reset_index()

ethnicity = ethnicity.rename(columns={'geography code': 'area_code'})

frames = dict(zip(keys, frames))

# Do not require accommodation, car ownership as only concerned with non residential
del frames['accommodation']
del frames['car']
del frames['long']
del frames['children']
del frames['country']
del frames['occupancy']
del frames['oac']

# Simple age range proportions will be enough for age
# Age will hold the geometry and area_code will be used to join
frames['age'] = frames['age'][['area_code', 'age_0_to_15_perc', 'age_16_to_34_perc',
                               'age_35_to_64_perc', 'age_65_plus_perc', 'geometry']]

frames['economic'] = frames['economic'][['area_code', 'unemployed_perc']]

frames['hh'] = frames['hh'][['area_code', 'one_pers_all_perc']]

frames['pop'] = frames['pop'][['area_code', 'density']]

frames['qualifications'] = frames['qualifications'][['area_code', 'no_qualifications_perc', 'level1_perc',
                                                     'level2_perc', 'level3_perc', 'level4_perc']]

frames['social'] = frames['social'][['area_code', 'social_grade_ab_perc', 'social_grade_c1_perc',
                                     'social_grade_c2_perc', 'social_grade_de_perc']]

frames['tenure'] = frames['tenure'][['area_code', 'owned_perc', 'rent_social_perc',
                                     'private_rent_perc']]

result = frames.pop('age')

frames['ethnicity'] = ethnicity

for i in frames.keys():
    result = result.merge(frames[i], on='area_code')

result_df = pd.DataFrame(result)

result.to_file(driver='ESRI Shapefile', filename='data/census/joined_demographic.shp')

"""Crime"""

crime = pd.read_csv('Data/Crime/MPS_LSOA_Level_Crime_Historic.csv')
crime_new = pd.read_csv('Data/Crime/MPS_LSOA_Level_Crime_recent.csv')

crime = pd.merge(crime, crime_new, on=['LSOA Code', 'Major Category'])
del crime_new

years = [str(i) for i in range(2008, 2021)]

for year in years:
    cols = [i for i in crime.columns if i[:4] == year]
    crime[year] = crime[cols].sum(axis=1)
    crime = crime.drop(cols, axis=1)

crime = crime.drop([i for i in crime.columns if 'Minor' in i or 'Borough' in i], axis=1)
crime.iloc[:, 2:] = crime.iloc[:, 2:].astype(int)

crime = crime.groupby(['LSOA Code', 'Major Category']).sum().reset_index()

lsoa = gpd.read_file('data/crime/LSOA_2011_London_gen_MHW.shp')
lsoa = lsoa.rename(columns={'LSOA11CD': 'LSOA Code'})

lsoa = lsoa[['LSOA Code', 'geometry']]

pop = pd.read_csv('data/census/population/SAPE20DT1-mid-2016-lsoa-syoa-estimates-formatted.csv',
                  skiprows=4)[['Area Codes', 'All Ages']]

pop = pop.rename(columns={'Area Codes': 'LSOA Code', 'All Ages': 'pop'})

crime = pd.merge(crime, pop, on='LSOA Code')

crime = pd.merge(crime, lsoa, on='LSOA Code')

crime = crime[['LSOA Code', 'Major Category', '2016', 'geometry', 'pop']]

crime = crime.pivot(index='LSOA Code', columns='Major Category').reset_index()

crime.columns = crime.columns.droplevel(0)

crime.columns = ['LSOA', 'Burglary', 'Robbery', 'Violent', 'geometry', 'geom2', 'geom3', 'pop', 'rmv', 'rmv2']

crime = crime.drop(columns=['LSOA', 'Robbery', 'geom2', 'geom3', 'rmv', 'rmv2'], axis=1)

crime['Burglary'] = crime['Burglary'].divide(crime['pop'])
crime['Violent'] = crime['Violent'].divide(crime['pop'])

crime = gpd.GeoDataFrame(crime, geometry=crime.geometry)

crime.to_file(driver='ESRI Shapefile', filename='data/crime/crime_2016.shp')
