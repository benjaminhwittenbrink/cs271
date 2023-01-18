
import pandas as pd 
from shapely.geometry import Point
import geopandas as gpd

import matplotlib.pyplot as plt 

plt.rcParams.update({'font.size': 20})

geometry = [Point(xy) for xy in zip(tmp['LONGNUM'], tmp['LATNUM'])]
gdf = gpd.GeoDataFrame(tmp, geometry=geometry)   

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
africa = world.query('continent == "Africa"')
gdf.plot(ax=africa.plot(figsize=(20, 12), edgecolor='gray', color='#FFFDD0'), column = "Under5_Mortality_Rate", marker='o', markersize=3, cmap = "inferno_r", legend=True, 
         legend_kwds={"label":"Under 5 Mortality Rate"})

plt.axis('off')
plt.show()

geometry = [Point(xy) for xy in zip(df['LONGNUM'], df['LATNUM'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)  
gdf["Mean_BMI"] = np.where(gdf["Mean_BMI"]> 35, 35, gdf["Mean_BMI"])

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
africa = world.query('continent == "Africa"')
gdf.plot(ax=africa.plot(figsize=(20, 12), edgecolor='gray', color='#FFFDD0'), column = "Mean_BMI", marker='o', markersize=3, cmap = "inferno_r", legend=True, 
         legend_kwds={"label":"Mean Body Mass Index"})

plt.axis('off')
plt.show()