### Import necessary packages
import pandas as pd
import requests

# Import data from API
response_API = requests.get('https://data.cityofnewyork.us/resource/uvpi-gqnh.json?$limit=1000000')
print(response_API.status_code)

# gather data from json format
data = response_API.json()
df = pd.json_normalize(data)

# Save data as pickle
df.to_pickle("data/dummy.pkl")

df = pd.read_pickle("data/dummy.pkl")