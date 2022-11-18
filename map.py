import folium
import pandas as pd

df = pd.read_csv('worldcitiespop.csv')

m = folium.Map(location=[43.2220, 76.8512])
m

m = folium.Map(location=[42.483333, 1.466667], zoom_start=12, tiles="Stamen Terrain")

tooltip = "Clickable Marker"

for index, row in df.iterrows():
    folium.Marker(
        [row['Latitude'], row['Longitude']], popup=row['AccentCity'], tooltip=tooltip
    ).add_to(m)

m
