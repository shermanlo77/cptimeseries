"""Base class for ERA5 with subclasses for Era5Wales and Era5Cardiff
"""

import datetime
from os import path
import pathlib

import gdal
import netCDF4
import numpy as np
import pandas as pd
import pupygrib

from dataset import ana
from dataset import data
from dataset import isle_of_man
from dataset import location
from dataset import wales

# ERA5 will need to be cropped to fit with the rest of the data
TRIM_INDEX = (slice(2, -2), slice(2, -2))

PATH_TO_ERA5 = path.join(pathlib.Path(__file__).parent.absolute(),
                         "..", "Data", "era5")
ERA5_1_FILE = path.join(PATH_TO_ERA5, "tp_sum_1.grib")
ERA5_2_FILE = path.join(PATH_TO_ERA5, "tp_sum_2.grib")
ERA5_3_FILE = path.join(PATH_TO_ERA5, "tp_sum_3.grib")
ERA5_4_FILE = path.join(PATH_TO_ERA5, "tp_sum_4.grib")


class Era5(data.DataDualGrid):
    """Base class for only loading precipitation from a .grib file
    """
    # only designed to load rain
    # topography_normalise not used

    def __init__(self, path_to_storage=None):
        super().__init__(path_to_storage=None)

    def load_rain(self, file_name):
        # override to read from grib file
        file = open(file_name, "rb")
        message_array = pupygrib.read(file)
        raster_array = gdal.Open(file_name)
        longitude_grid, latitude_grid = np.meshgrid(
            data.LONGITUDE_ARRAY, data.LATITUDE_ARRAY)

        # store precipitation in rain_array_file for now then append to
        # self.rain
        # self.rain may be None, or already instantiated as a numpy array
        rain_array_file = []

        for i, message in enumerate(message_array):

            # check coordinates are as expected
            coordinates = message.get_coordinates()
            assert(
                np.all(
                    np.isclose(coordinates[0][TRIM_INDEX], longitude_grid)))
            assert(
                np.all(
                    np.isclose(coordinates[1][TRIM_INDEX], latitude_grid)))

            raster = raster_array.GetRasterBand(i+1)

            # get date
            time = raster.GetMetadata()["GRIB_VALID_TIME"].lstrip().rsplit()[0]
            time = datetime.datetime.fromtimestamp(int(time))
            # only use as data point if before or on MAX_DATE
            if time.date() <= data.MAX_DATE:
                # append date and precipitation
                self.time_array.append(time.date())
                rain = raster.ReadAsArray() * 1000
                rain = rain[TRIM_INDEX]
                rain_array_file.append(rain)

        file.close()

        # append precipitation, numpy format
        rain_array_file = np.asarray(rain_array_file)
        if self.rain is None:
            self.rain = rain_array_file
        else:
            self.rain = np.concatenate((self.rain, rain_array_file))

    def get_model_field(self, latitude_index, longitude_index):
        # override to return a cloneable empty object
        return pd.DataFrame({})

    def load_mask(self, file_name):
        """Load mask from a .nc file
        """
        rain_data = netCDF4.Dataset(file_name, "r", format="NETCDF4")
        rain = rain_data.variables["rr"]
        self.mask = np.flip(rain[0].mask, 0)

    def crop(self, lat, long):
        # override, to only crop the rain, mask and topography
        self.rain = self.rain[:, lat[0]:lat[1], long[0]:long[1]]
        self.mask = self.mask[lat[0]:lat[1], long[0]:long[1]]
        for key in self.topography:
            self.topography[key] = self.topography[key][
                lat[0]:lat[1], long[0]:long[1]]


class Era5Ana(Era5):

    def __init__(self, path_to_storage=None):
        super().__init__(path_to_storage=None)

    def load_data(self):
        self.load_rain(ERA5_3_FILE)
        self.load_rain(ERA5_4_FILE)
        self.load_mask(ana.RAIN_FILE)


class Era5Wales(Era5Ana):

    def __init__(self, path_to_storage=None):
        super().__init__(path_to_storage=None)

    def load_data(self):
        super().load_data()
        self.crop(wales.LAT, wales.LONG)


class Era5IsleOfMan(Era5):

    def __init__(self, path_to_storage=None):
        super().__init__(path_to_storage=None)

    def load_data(self):
        self.load_rain(ERA5_3_FILE)
        self.load_mask(ana.RAIN_FILE)
        self.crop(isle_of_man.LAT, isle_of_man.LONG)


class Era5Cardiff(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(Era5Ana(), "Cardiff")
