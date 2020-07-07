"""This script gets building energy efficiency data"""

import requests
import os
import csv
import pickle
from tqdm import tqdm
import re
from io import StringIO
import pandas as pd

auth_key = 'dWNlc2pyMEB1Y2wuYWMudWs6YjhmNWQzODIxZDYwOTFlYjNlZGRiNDgyNjNlNTg3NjQ0ZmE2NzA5ZQ=='

headers = {
    'Accept': 'text/csv',
    'Authorization': 'Basic %s' % auth_key
}


# test = "1 rectory cottages"
# pstcd = "CA10 3AD"


def search_address(address=False, postcode=False):
    # Construct the query string
    search_str = "?"
    search_str += "address="
    if address:
        search_str += "".join([i + "+" for i in address.split()])
    search_str += "&postcode="
    if postcode:
        search_str += postcode.replace(" ", "")
    return search_str + "&local-authority=&constituency=&from-month=1&from-year=2008&to-month=12&to-year=2020"


def fetch_epc(address="", postcode=""):
    # make API request and return dataframe
    r = requests.get('https://epc.opendatacommunities.org/api/v1/domestic/search' + search_address(address, postcode),
                     headers=headers)

    epc_csv = StringIO(str(r.content, 'utf-8'))  # bytes to csv
    try:
        return pd.read_csv(epc_csv)
    except pd.errors.EmptyDataError:
        pass


# df = fetch_epc(address=test, postcode=pstcd)

# uprn_path = 'C:/Users/Jake/python_code/dissertation/'
# parent_path = 'D:/Dissertation/'
# add_path = 'D:/Dissertation/addressbase/'
#
# with open('addressbase-plus-header.csv', newline='') as f:
#     header = list(csv.reader(f))[0]
# header[0] = 'UPRN'
#
# years = [str(i) for i in range(2013, 2019)]
#
# uprn_files = [i for i in os.listdir(uprn_path) if 'uprn' in i]
#
# for year in years:
#     dfs = []  # A list to hold all return data frames - to be concatenated later
#
#     with open(uprn_path + [i for i in uprn_files if year in i][0], 'rb') as src:
#         uprn = pickle.load(src)
#
#     tiles = [i for i in os.listdir(parent_path + 'imagery/' + year) if '.' not in i]
#
#     # Filter addressbase files for those covered by the imagery for that year
#     adds = [i for i in os.listdir(add_path) if
#             i[2] in [j[2] for j in tiles] and
#             i[4] in [n[3] for n in tiles]]
#
#     for i in adds:
#         add = pd.read_csv(add_path + i)
#
#         add.columns = header
#
#         # Filter for UPRNs that have a match
#         add = add[add['UPRN'].isin(uprn.keys())]
#
#         for ind, val in add.iterrows():
#             # relevant address info
#             res = [val['SUB_BUILDING_NAME'], val['BUILDING_NAME'], val['BUILDING_NUMBER'],
#                    val['THOROUGHFARE']]
#
#             # get rid of nans
#             res = [str(i) for i in res if str(i) != 'nan']
#
#             res.append(val['POSTCODE'])
#
#             # skip the iteration if there is no address info
#             if len(res) == 1 and str(res[0]) == 'nan':
#                 continue
#
#             res = [str(int(i)) if type(i) == float else i for i in res]
#
#             if len(res) > 1:
#                 print("ye")
#                 row = fetch_epc(" ".join(res[:-1]), res[-1])
#                 if row is not None:
#                     print('s')
#                     row['UPRN'] = val['UPRN']
#                     row['TOID'] = uprn[val['UPRN']]
#                     dfs.append(row)
#
#     df_res = pd.concat([i for i in dfs if i is not None])
#
#     with open('epc_data_' + year + '.pickle', 'wb') as dest:
#         pickle.dump(df_res, dest)



def get_first(s):
    # Try to get the postcode from the address column
    try:
        return re.search(r'\b[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][ABD-HJLNP-UW-Z]{2}\b', s).group()
    except AttributeError:
        return None


def cut_pc(s):
    try:
        pc = re.search(r'\b[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][ABD-HJLNP-UW-Z]{2}\b', s).group()
        s = s.replace(pc, '')
        return s[:-5]  # Trim off ' , UK'
    except AttributeError:
        pass


df = pd.read_csv('non_res_addresses.csv')

df = df.groupby(['TOID', 'types', 'name', 'address'])['query_type'].apply(list).reset_index()

df = df.groupby(['TOID', 'name', 'address'])['types'].apply(list).reset_index()

df['postcode'] = df['address'].apply(lambda x: get_first(x))  # Apply the function across address series to new column

df['address'] = df['address'].apply(lambda x: cut_pc(x))  # Trim ', UK' from the end

frames = []

for i in tqdm(range(len(df))):
    frames.append(fetch_epc(df.iloc[i]['address'], df.iloc[i]['postcode']))
