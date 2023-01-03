### Import necessary packages
import pandas as pd
import requests

# Import data from API
response_API = requests.get('https://data.cityofnewyork.us/resource/uvpi-gqnh.json?$limit=1000000')
response_API_2 = requests.get('https://data.cityofnewyork.us/resource/93vf-i5bz.json')
print(response_API.status_code)

# gather data from json format
data = response_API.json()
neigh = response_API_2.json()

df = pd.json_normalize(data)
df_neigh = pd.json_normalize(neigh)

# Save data as pickle
df.to_pickle("data/dummy.pkl")
df.to_pickle("data/neigh.pkl")

# Read data from pickle
df = pd.read_pickle("data/dummy.pkl")
neigh = pd.read_pickle("data/neigh.pkl")

# Explore dataset
df.head()

# Remove unncessary columns
df = df.drop(['B', 'C'], axis=1)

df["curb_loc"].unique()
'OffsetFromCurb' in df["curb_loc"].unique()

def hola(df):
    """
    Function to convert cdf columns to categorical depending on their row number compared with the
    total length of the dataframe
    :param df, tolerance: Tolerance in x.xx format
    :return: Dataframe with converted columns
    """


def unique_values(df):
    """
    Get distinct values and number of them in a dataframe
    :param df: Dataframe to check
    :return: Df with list and count of unique values
    """
    values = df.apply(lambda col: col.unique())
    counts = df.apply(lambda col: col.nunique())
    resumen = pd.concat([values, counts], axis=1)
    return (resumen)


import geopandas as gpd
neighborhoods = gpd.read_file('data/neigh.shp')
neighborhoods
neighborhoods.plot()

# We restrict to South America.
ax = neighborhoods[neighborhoods.continent == 'South America'].plot(
    color='white', edgecolor='black')

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax, color='red')

plt.show()


import pandas as pd

df = pd.read_csv(r"C:\Users\Diego\Desktop\Competitions_datacamp\data\credit_data\bureau.csv", nrows=100000)

def remove_rows_with_na(df, proportion):
    """
    Function to remove rows with more than a proportion value of Nans
    :param df:
    :param proportion:
    :return: df without those rows and df of rows removed
    """
    df_copy = df.copy()
    rows_to_rm = (df.isnull().sum(axis=1) / len(df.columns.tolist())) > proportion
    result = df_copy[~rows_to_rm]
    removed = df_copy[rows_to_rm]
    return result, removed

cosa,cosa2 = remove_rows_with_na(df, 0.9)

import numpy as np
import pandas as pd

def cap_outliers_percentiles(df, vars_to_skip, perc1=10, perc2=90):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_result = df.copy()

    for c in list(df_result.columns):
        if (df_result[c].dtype in numerics) and (c not in vars_to_skip):
            # Computing 10th, 90th percentiles and replacing the outliers
            floor_percentile = np.percentile(df_result[c], perc1)
            cap_percentile = np.percentile(df_result[c], perc2)
            # print(tenth_percentile, ninetieth_percentile)b = np.where(sample<tenth_percentile, tenth_percentile, sample)
            df_result[c] = np.where(df_result[c] > cap_percentile, cap_percentile, df_result[c])
            df_result[c] = np.where(df_result[c] < floor_percentile, floor_percentile, df_result[c])
            print(df_result[c].describe())

    return df_result