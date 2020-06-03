import pandas as pd

parent_path = "D:/Dissertation"

# incident records - varying date format
data1 = pd.read_csv(parent_path + "/fire/LFB_2009_2012.csv")  # dd-Mmm-yy
data2 = pd.read_csv(parent_path + "/fire/LFB_2013_2016.csv")  # dd-Mmm-yy
data3 = pd.read_csv(parent_path + "/fire/LFB_2017.csv")  # dd/mm/yyyy

# All on one df
data = data1.append(data2)
data = data.append(data3)

# Filters
data = data.loc[data['Easting_m'].notnull()]
data = data.loc[data['IncidentGroup'] == 'Fire']
data = data.loc[data['StopCodeDescription'] == 'Primary Fire']
data = data.loc[(data['AddressQualifier'] == 'Correct incident location') | (data['AddressQualifier'] == 'Within same building')]
data = data.loc[data['PropertyCategory'] != 'Road Vehicle']

print(len(data))

# Convert to uniform date
data['DateOfCall'] = pd.to_datetime(data['DateOfCall'])

data.to_csv(parent_path + "/fire/filtered_fire.csv")
