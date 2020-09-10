from processing.matching import *
from processing.make_dates import make_dates
from processing.clipping import clip_shapes, remove_missing_img_data
from processing.types import one_hot_types
from sklearn import preprocessing


"""Stage 1"""
# Joining TOID
positive_toid = incident_toid_match()  # Spatial join of incidents to MM buildings

positive_toid.to_csv('data/process_flow/positive/1_positive_toid_match.csv')

"""Stage 2"""
# Joining Places
positive_toid = places_toid_match(positive_toid)  # Join of TOID to places

positive_toid.to_csv('data/process_flow/positive/2_positive_toid_place_match.csv')

negative_toid = neg_toid_match()  # Spatial join of random Google Places results to MM buildings

# Drop duplicates for the same building from multiple years at random - i.e. do not just keep first
negative_toid = negative_toid.sample(frac=1).drop_duplicates('TOID').reset_index()

negative_toid.to_csv('data/process_flow/negative/2_negative_toid_place_match.csv')

positive = gpd.read_file('data/process_flow/positive/2_positive_toid_place_match.csv',
                         GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")

negative = gpd.read_file('data/process_flow/negative/2_negative_toid_place_match.csv',
                         GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")

"""Stage 3"""
# Make random dates for negatives, a month int feature and remove irrelevant features
positive, negative = make_dates(positive, negative)

del positive_toid, negative_toid

# Remove irrelevant columns so far
positive = positive[['TOID', 'CalculatedAreaValue', 'types', 'DateOfCall',
                     'CalYear', 'month_int', 'img_year', 'geometry']]
negative = negative[['TOID', 'CalculatedAreaValue', 'types', 'DateOfCall',
                     'CalYear', 'month_int', 'img_year', 'geometry']]

positive.to_csv('data/process_flow/positive/3_positive_dated.csv')
negative.to_csv('data/process_flow/negative/3_negative_dated.csv')

positive = gpd.read_file('data/process_flow/positive/3_positive_dated.csv', GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")

negative = gpd.read_file('data/process_flow/negative/3_negative_dated.csv', GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")

"""Stage 4"""
# Produce image data set
# Make index to use for image names
positive = positive.reset_index()
negative = negative.reset_index()

# Clip shapes
positive = clip_shapes(positive, 'img_year', 'labelled/positive', buff=True)
# Do not filter fabricated events by year as the imagery can be from any time - minimise feature loss from this
# Do not filter no data entries from DataFrame as imagery may be found from a subsequent year
negative = clip_shapes(negative, 'labelled/negative', year_col=False, buff=True, fil_no_data=False)

# Attempt to cut out DSM
positive = clip_shapes(positive, 'D:/Dissertation/labelled/DSM/positive', year_col=False, buff=True,
                       fil_no_data=False, years=False, imagery=False)

negative = clip_shapes(negative, 'D:/Dissertation/labelled/DSM/negative', year_col=False, buff=True,
                       fil_no_data=False, years=False, imagery=False)

# Remove rows for buildings whose imagery could not be found
positive = remove_missing_img_data('D:/Dissertation/labelled/positive', positive)
negative = remove_missing_img_data('D:/Dissertation/labelled/negative', negative)

positive.to_csv('data/process_flow/positive/4_positive_indexed.csv')
negative.to_csv('data/process_flow/negative/4_negative_indexed.csv')


positive = gpd.read_file('data/process_flow/positive/4_positive_indexed.csv', GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")

negative = gpd.read_file('data/process_flow/negative/4_negative_indexed.csv', GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")


"""Stage 5"""
# Categorise the type variables so one hot vector are less sparse
positive = one_hot_types(positive)
negative = one_hot_types(negative)


positive = positive.drop(columns=['field_1', 'types', 'point_of_interest', 'establishment', 'park', 'neighborhood', 'funeral_home',
                                  'premise', 'atm', 'sublocality', 'sublocality_level_1', 'campground'])
negative = negative.drop(columns=['field_1', 'types', 'point_of_interest', 'establishment', 'sublocality', 'sublocality_level_1',
                                  'park', 'neighborhood', 'funeral_home', 'premise', 'cemetery', 'atm', 'route',
                                  'locality'])


positive.to_csv('data/process_flow/positive/6_positive_one_hot.csv')
negative.to_csv('data/process_flow/negative/6_negative_one_hot.csv')


positive = gpd.read_file('data/process_flow/positive/6_positive_one_hot.csv', GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")

negative = gpd.read_file('data/process_flow/negative/6_negative_one_hot.csv', GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")


"""Stage 6"""
# Joining demographic and crime data
demographic = gpd.read_file("data/census/joined_demographic.shp")
crime = gpd.read_file("data/crime/crime_2016.shp")

# Spatial join to demographic data, drop duplicated values and remove new right index columns
positive = gpd.sjoin(positive, demographic)
positive = positive.drop_duplicates('index')
positive = positive.drop(['field_1', 'index_right'], 1)

negative = gpd.sjoin(negative, demographic)
negative = negative.drop_duplicates('index')
negative = negative.drop(['field_1', 'index_right'], 1)

del demographic

positive = gpd.sjoin(positive, crime)
positive = positive.drop(['index_right'], 1)

negative = gpd.sjoin(negative, crime)
negative = negative.drop(['index_right'], 1)

positive.to_csv('data/process_flow/positive/7_positive_complete.csv')
negative.to_csv('data/process_flow/negative/7_negative_complete.csv')


"""Stage 7"""
positive = gpd.read_file('data/process_flow/positive/7_positive_complete.csv', GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")

negative = gpd.read_file('data/process_flow/negative/7_negative_complete.csv', GEOM_POSSIBLE_NAMES="geometry",
                         KEEP_GEOM_COLUMNS="NO")

positive_x = positive['geometry'].centroid.x
positive_y = positive['geometry'].centroid.y
positive_ind = positive['index']

negative_x = negative['geometry'].centroid.x
negative_y = negative['geometry'].centroid.y
negative_ind = negative['index']

# Drop all non-feature columns
to_drop = ['TOID', 'DateOfCall', 'CalYear', 'img_year', 'geometry', 'area_code']
positive = positive.drop(to_drop, 1)
negative = negative.drop(to_drop, 1)

# Final normalize
min_max_scaler = preprocessing.MinMaxScaler()
pos_scaled = min_max_scaler.fit_transform(positive.values)
positive = pd.DataFrame(pos_scaled)

neg_scaled = min_max_scaler.fit_transform(negative.values)
negative = pd.DataFrame(neg_scaled)

positive.to_csv('data/process_flow/positive/8_positive_final.csv')
negative.to_csv('data/process_flow/negative/8_negative_final.csv')

positive['x'] = positive_x
positive['y'] = positive_y
positive[1] = positive_ind

negative['x'] = negative_x
negative['y'] = negative_y
negative[1] = negative_ind

positive.to_csv('data/process_flow/positive/9_positive_with_point.csv')
negative.to_csv('data/process_flow/negative/9_negative_with_point.csv')
