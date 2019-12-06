#check the latitude and longitude resolution of the data
#check the time resolution and if it is consistent

import numpy as np
from netCDF4 import Dataset, num2date

#list of .nc files to look at
file_array = [
    "Data/Rain_Data_Nov19/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc",
    "Data/Rain_Data_Nov19/ana_input_1.nc"
]

for file_name in file_array:
    
    #load the data
    data = Dataset(file_name, "r", format="NETCDF4")
    data = data.variables
    #get the latitude, longitude and time
    latitude_array = np.sort(np.asarray(data["latitude"]))
    longitude_array = np.sort(np.asarray(data["longitude"]))
    time = data['time']
    time_array = num2date(time[:], time.units)
    
    #print the file name, latitude, longitude and time
    print("Data in "+file_name)
    print("Number of latitude points:", latitude_array.size)
    print("Number of longitude points:", longitude_array.size)
    print("List of latitudes:")
    print(latitude_array)
    print("List of longitudes")
    print(longitude_array)
    print("List of time")
    print(time_array)
    
    #get and print the resolution of the time
    time_diff = time_array[1] - time_array[0]
    print("Time resolution:", time_diff)
    #check if the resolution of the time is consistent throughout the data
    for i in range(2, len(time_array)):
        time1 = time_array[i-1]
        time2 = time_array[i]
        if (time2 - time1).seconds != time_diff.seconds:
            print("Time difference is not a day between", time2, "and", time1)
    
    print()
