#script for reading in the .nc file
#
#print all the names of all the covariates
#print the properties of all the covariates
#plot the mean daily rainfall for each point in space
#for London, plot the time series rainfall 

import matplotlib.pyplot as plt
import numpy as np

from netCDF4 import Dataset, num2date
from pandas.plotting import register_matplotlib_converters

#required so that python datetime can be converted and can be plotted on a graph
register_matplotlib_converters()

print("==========")

#load the .nc file using netCDF4
file_name = "Data/Rain_Data/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_djf_uk.nc"
print("Look into the data in " + file_name + "\n")
dataset = Dataset(file_name, "r", format="NETCDF4")
data = dataset.variables

#data is an ordered dict
#loop through the keys and print it
print("The name of the variables are:")
for key in data.keys():
    print(key)
print()

#print each item in dat
print("For each of the variables, here are the properties:\n")
for key, value in data.items():
    print("Variable: " + key)
    print(value)
    print()

#get the mean rainfall (mean over time) for each point in space
mean_rainfall = np.mean(data['rr'][:],0)
#get the longitude and latitude_error
longitude_array = data['longitude']
latitude_array = data['latitude']
#for plotting, get a grid of x's and y's
x_grid, y_grid = np.meshgrid(longitude_array, latitude_array)

#heatmap plot the mean rainfall
plt.figure()
plt.pcolor(x_grid, y_grid, mean_rainfall)
plt.title("Mean "+data['rr'].long_name+" ("+data['rr'].units+")")
plt.xlabel("longitude (degree)")
plt.ylabel("latitude (degree)")
cbar = plt.colorbar()
plt.show()

#get rain data for London (select a specific longitude and latitude)
#London: 51.5074 deg N, 0.1278 deg W
min_longitude_error = float("inf")
for i, longitude in enumerate(longitude_array):
    longitude_error = abs(longitude - 0.1278)
    if min_longitude_error > longitude_error:
        longitude_index = i
        min_longitude_error = longitude_error

min_latitude_error = float("inf")
for i, latitude in enumerate(latitude_array):
    latitude_error = abs(latitude - 51.5074)
    if min_latitude_error > latitude_error:
        latitude_index = i
        min_latitude_error = latitude_error

time = data['time']
time = num2date(time[:], time.units)
rainfall_series = data['rr'][:,latitude_index,longitude_index]
plt.figure()
plt.plot(time, rainfall_series)
plt.title("London")
plt.xlabel("Time (calendar year)")
plt.ylabel(data['rr'].long_name+" ("+data['rr'].units+")")
plt.show()
