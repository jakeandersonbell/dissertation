"""Fabricate dates for negative classes with similar distribution"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os


def make_dates(positive, negative):
    positive['DateOfCall'] = pd.to_datetime(positive['DateOfCall'])

    negative['DateOfCall'] = ''

    # Assign a random date from the positives - should result in a similar distribution
    for i, row in negative.iterrows():
        negative.at[i, 'DateOfCall'] = positive.loc[np.random.randint(len(positive))]['DateOfCall']

    # Get the month int so it can be used as a feature
    negative['month_int'] = negative['DateOfCall'].apply(lambda x: x.month)
    positive['month_int'] = positive['DateOfCall'].apply(lambda x: x.month)

    negative['CalYear'] = negative['DateOfCall'].apply(lambda x: x.year)

    return positive, negative
