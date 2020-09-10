import pandas as pd
import geopandas as gpd
import os
import ast

# A dictionary to simplify the types
type_categories = {'amusement': ['casino', 'movie_theater', 'bowling_alley', 'aquarium', 'amusement_park'],
                   'car': ['car_wash', 'car_repair', 'gas_station', 'parking'],
                   'drink': ['bar', 'night_club'],
                   'emergency': ['fire_station', 'police'],
                   'food': ['bakery', 'cafe', 'food', 'meal_delivery', 'meal_takeaway', 'restaurant'],
                   'contractor': ['electrician', 'general_contractor', 'locksmith', 'moving_company', 'painter',
                                  'plumber',
                                  'roofing_contractor'],
                   'leisure': ['gym', 'spa'],
                   'medical': ['dentist', 'doctor', 'health', 'hospital', 'pharmacy', 'physiotherapist',
                               'veterinary_care'],
                   'office': ['lawyer', 'embassy', 'local_government_office', 'political', 'accounting', 'finance',
                              'insurance_agency'],
                   'public_building': ['church', 'city_hall', 'courthouse', 'library', 'place_of_worship',
                                       'primary_school', 'school', 'secondary_school', 'stadium', 'synagogue',
                                       'tourist_attraction', 'university', 'art_gallery', 'museum', 'hindu_temple',
                                       'mosque'],
                   'retail': ['bicycle_store', 'book_store', 'car_dealer', 'car_rental', 'clothing_store',
                              'convenience_store', 'department_store', 'drugstore', 'electronics_store', 'bank',
                              'furniture_store', 'grocery_or_supermarket', 'hardware_store', 'home_goods_store',
                              'jewelry_store', 'liquor_store', 'post_office', 'real_estate_agency', 'shoe_store',
                              'shopping_mall', 'store', 'supermarket', 'travel_agency', 'florist', 'movie_rental',
                              'pet_store'],
                   'service': ['beauty_salon', 'hair_care', 'laundry'],
                   'transport': ['bus_station', 'light_rail_station', 'subway_station', 'taxi_stand', 'train_station',
                                 'transit_station', 'airport'],
                   'storage': ['storage']}


def simplify_types(x):
    # Function to simplify types list
    for i, item in enumerate(x):
        for j in type_categories.keys():
            if item in type_categories[j]:
                x[i] = j

    return list(set(x))


def one_hot_types(df):
    # Apply to categorise types
    df['types'] = df['types'].apply(lambda x: simplify_types(ast.literal_eval(x)))

    # One hot encoding
    # Get one hot of lists within column
    df_hot = pd.get_dummies(df['types'].apply(pd.Series).stack()).sum(level=0)

    return pd.concat([df, df_hot], axis=1)