"""Contains the class DataDualGrid, a wrapper class for model fields,
     precipitation adn topography data.
"""

import datetime
import math
from os import path
import pathlib

import gdal
import joblib
import netCDF4
import numpy as np
from numpy import ma
import pandas as pd
import pupygrib
from scipy import interpolate

CITY_LOCATION = {
    "London": [51.5074, -0.1278],
    #+0.1 to avoid masking a coastal city
    "Cardiff": [51.4816 + 0.1, -3.1791],
    "Edinburgh": [55.9533, -3.1883],
    "Belfast": [54.5973, -5.9301],
    "Dublin": [53.3498, -6.2603],
}
LATITUDE_ARRAY = np.linspace(58.95, 49.05, 100)
LONGITUDE_ARRAY = np.linspace(-10.95, 2.95, 140)
LATITUDE_TOPO_ARRAY = np.linspace(59.15, 48.85, 104)
LONGITUDE_TOPO_ARRAY = np.linspace(-11.15, 3.15, 144)
LATITUDE_COARSE_ARRAY = np.linspace(59.445, 48.888, 20)
LONGITUDE_COARSE_ARRAY = np.linspace(-12.5, 4.167, 21)
RADIUS_OF_EARTH = 6371E3
ANGLE_RESOLUTION = 0.1
RESOLUTION = math.radians(ANGLE_RESOLUTION)*RADIUS_OF_EARTH
GRAVITATIONAL_FIELD_STRENGTH = 9.81
DERIVED_MODEL_FIELD = {
    "wind_speed": "m s-1",
    "specific_humidity_rate": "m deg-1 s-1",
    "total_column_water_rate": "kg m-1 deg-1 s-1",
}
MAX_DATE = datetime.date(2019, 7, 31) #end point of data inclusive

class DataDualGrid(object):
    """Wrapper class for model fields, precipitation and topography

    Notes: load model fields first, then precipitation as the model fields
        determine what time points to extract from the precipitation data

    Attributes:
        model_field: dictionary, keys are name of model fields, values are 3d
            array of matrices, 0th dimension for each time point, 1st and 2nd
            corresponding to model fields on the fine grid
        model_field_coarse: dictionary, keys are name of model fields, values
            are array of matrices corresponding to model fields on the coarse
            grid
        model_field_units: dictionary, keys are name of model fields, values are
            strings with represent the units the model fields are measured in
        rain: 3d masked array, contains precipitation data, 0th dimension for
            each time point, 1st and 2nd corresponding to fine grid
        mask: boolean matrix, shape corresponding to fine grid, True if a point
            is on water, ie no precipitation measure on water
        rain_units: units precipitation is measured in
        time_array: array of date objects
        topography: dictionary, keys are names of topography, values are a
            matrix with topography values, corresponding on the fine grid
        topography_normalise: dictionary, keys are names of topography, values
            are a matrix with normalised topography values, corresponding on the
            fine grid. Normalised to have mean 0, std 1
    """

    def __init__(self, path_to_storage=None):
        self.model_field = None
        self.model_field_coarse = None
        self.model_field_units = {}
        self.rain = None
        self.mask = None
        self.rain_units = None
        self.time_array = []
        self.topography = {}
        self.topography_coarse = {}
        self.topography_normalise = {}

        if path_to_storage is None:
            path_to_storage = pathlib.Path(__file__).parent.absolute()
        storage_file = path.join(path_to_storage, type(self).__name__+".gz")
        #attempt to load the data from .gz file
        if path.isfile(storage_file):
            print("Loading", storage_file)
            self.copy_from(joblib.load(storage_file))
        #otherwise read and dump the data into a .gz file
        else:
            longitude_grid, latitude_grid = np.meshgrid(
                LONGITUDE_ARRAY, LATITUDE_ARRAY)
            self.topography["longitude"] = longitude_grid
            self.topography["latitude"] = latitude_grid

            longitude_grid, latitude_grid = np.meshgrid(
                LONGITUDE_COARSE_ARRAY, LATITUDE_COARSE_ARRAY)
            self.topography_coarse["longitude"] = longitude_grid
            self.topography_coarse["latitude"] = latitude_grid

            self.normalise_topography()
            self.load_data()
            print("Saving", storage_file)
            joblib.dump(self, storage_file)

    def copy_from(self, other):
        """Shallow copy another Data object
        """
        self.model_field = other.model_field
        self.model_field_coarse = other.model_field_coarse
        self.model_field_units = other.model_field_units
        self.rain = other.rain
        self.mask = other.mask
        self.rain_units = other.rain_units
        self.time_array = other.time_array
        self.topography = other.topography
        self.topography_normalise = other.topography_normalise

    def load_data(self):
        """Call all the appropriate methods to load the data into memory
        """
        raise NotImplementedError()

    def load_model_field(self, file_name):
        """Load model fields from grib file

        Load model fields onto memory. Supports coarse grid only.

        Only supports dates up to the constant MAX_DATE

        Args:
            file_name: name of the .grib file to read from
        """
        file = open(file_name, "rb")
        message_array = pupygrib.read(file)
        raster_array = gdal.Open(file_name)
        longitude_coarse_grid, latitude_coarse_grid = np.meshgrid(
            LONGITUDE_COARSE_ARRAY, LATITUDE_COARSE_ARRAY)

        #grib comment is model field name
        time_array_all = {} #dictionary of datetime, keys: model field names
        grib_comment_array = [] #array of model field names
        #dictionary, array of pointers for GetRasterBand, keys: grib comments
        raster_pointer = {}
        #go through data, get model field names and times
        for i, message in enumerate(message_array):

            #check coordinates are as expected
            coordinates = message.get_coordinates()
            assert(
                np.all(np.isclose(coordinates[0], longitude_coarse_grid)))
            assert(
                np.all(np.isclose(coordinates[1], latitude_coarse_grid)))

            #get model field name
            raster = raster_array.GetRasterBand(i+1)
            grib_comment = raster.GetMetadata()["GRIB_COMMENT"]
            if not grib_comment in grib_comment_array:
                grib_comment_array.append(grib_comment)
                raster_pointer[grib_comment] = []
                time_array_all[grib_comment] = []
            raster_pointer[grib_comment].append(i+1)

            #get time, GRIB_VALID_TIME and GRIB_REF_TIME contain strings
            time = raster.GetMetadata()["GRIB_VALID_TIME"].lstrip().rsplit()[0]
            time2 = raster.GetMetadata()["GRIB_REF_TIME"].lstrip().rsplit()[0]
            assert(time.isnumeric())
            assert(time2.isnumeric())
            time = int(time)
            time2 = int(time2)
            assert(time == time2)

            #only use as data point if before or on MAX_DATE
            time = datetime.datetime.fromtimestamp(time)
            if time.date() <= MAX_DATE:
                time_array_all[grib_comment].append(time)

        #check all times from all model fields are the same
        time_array = time_array_all[grib_comment_array[0]]
        for time_array_copy in time_array_all.values():
            for i, time_i in enumerate(time_array_copy):
                assert(time_array[i] == time_i)

        #do not need pubgrib any more, only used to get coordinates
        file.close()

        #rename model field names and add units
        model_field_name_array = []
        for grib_comment in grib_comment_array:
            model_field_name = grib_comment.split(" [")
            units = model_field_name[1]
            model_field_name = model_field_name[0].lower().replace(" ", "_")
            units = units.split("]")[0]
            if model_field_name == "precipitable_water_content":
                model_field_name = "total_column_water"
            elif model_field_name == "u-velocity":
                model_field_name = "x_wind"
            elif model_field_name == "v-velocity":
                model_field_name = "y_wind"
            elif model_field_name == ("geopotential_"
                "(at_the_surface_=_orography)"):
                model_field_name = "geopotential"
            model_field_name_array.append(model_field_name)
            if not model_field_name in self.model_field_units:
                self.model_field_units[model_field_name] = units

        #smooth daily readings and get derived model fields, eg using gradients
        raster_generator = GeneratorGrib(
            raster_array, raster_pointer, 4, time_array)
        model_field = self.get_derive_smooth_model_field(raster_generator,
                                                         model_field_name_array,
                                                         LATITUDE_COARSE_ARRAY,
                                                         LONGITUDE_COARSE_ARRAY)

        for key, model_field_i in model_field.items():
            model_field[key] = np.asarray(model_field_i)
        #assign or append model field to the member variable
        if self.model_field_coarse is None:
            self.model_field_coarse = model_field
        else:
            for key, model_field_array in self.model_field_coarse.items():
                self.model_field_coarse[key] = np.concatenate(
                    (model_field_array, model_field[key]))

    def get_derive_smooth_model_field(self,
                                      model_field_generator,
                                      new_keys,
                                      latitude_array,
                                      longitude_array):
        """Format model fields

        Smooth the 4 daily readings to a daily reading using the mean. Add
            additional model fields such as wind speed and rates. This is then
            returned. Also add time points to self.time_array

        Notes: due to daylight saving in UK, did not implement assert the
            difference between time points in daily readings as they can be off
            by an hour

        Args:
            model_field_generator: see generator class wrappers such as
                GeneratorNc and GeneratorGrib, they iterate model fields 4 at a
                time
            latitude_array: vector of latitude points, used for gradients
            longitude_array: vector of longitude points, used for gradients

        Return:
            dictionary of model fields
        """
        n_read_per_day = model_field_generator.n_read_per_day
        time_array = model_field_generator.time_array

        #to store the smoothed model fields
        model_field_smooth = {}
        for key in new_keys:
            model_field_smooth[key] = []
        #add derived model fields and their units
        for key, units in DERIVED_MODEL_FIELD.items():
            if units not in self.model_field_units:
                self.model_field_units[key] = units
            model_field_smooth[key] = []

        dataset_iterator = iter(model_field_generator)

        for i in range(0, len(time_array), n_read_per_day):
            model_field_day = {}
            for model_field_name in model_field_smooth:
                #get the 4 readings in the day
                if not model_field_name in DERIVED_MODEL_FIELD:
                    model_field_day[model_field_name] = next(dataset_iterator)

            #work out derived model fields
            model_field_day["wind_speed"] = np.sqrt(
                np.square(model_field_day["x_wind"])
                + np.square(model_field_day["y_wind"]))
            model_field_day["specific_humidity_rate"] =  self.get_rate(
                model_field_day, "specific_humidity", latitude_array,
                longitude_array)
            model_field_day["total_column_water_rate"] = self.get_rate(
                model_field_day, "total_column_water", latitude_array,
                longitude_array)

            #smooth model fields
            for model_field_name in model_field_smooth:
                model_field_smooth[model_field_name].append(
                    np.mean(model_field_day[model_field_name], 0))

            #append date
            date_array = time_array[i:i+4]
            for i in range(n_read_per_day-1):
                assert(date_array[i].date() == date_array[i+1].date())

            date = date_array[0].date()
            #assert time difference is one day
            if self.time_array:
                assert((date - self.time_array[-1]) == datetime.timedelta(1))
            self.time_array.append(date)

        return model_field_smooth

    def get_rate(self, model_field, key, latitude_array, longitude_array):
        """Get rate of a specific model field

        Gradient of model field x wind speed, abs
        """
        grad = np.gradient(
            model_field[key], latitude_array, longitude_array, axis=(1,2))
        return np.sqrt(
            np.square(grad[0] * model_field["y_wind"])
            + np.square(- grad[1] * model_field["x_wind"]))

    def interpolate_model_field(self):
        """Interpolate the model fields in self.model_field_coarse to the fine
            grid, store it in self.model_field
        """
        self.model_field = {}
        for key, model_field in self.model_field_coarse.items():
            self.model_field[key] = []
            for i in range(len(self.time_array)):
                interpolator = interpolate.RectBivariateSpline(
                    np.flip(LATITUDE_COARSE_ARRAY), LONGITUDE_COARSE_ARRAY,
                    np.flip(model_field[i,:,:], 0))
                model_field_interpolate_i = interpolator(
                    np.flip(LATITUDE_ARRAY), LONGITUDE_ARRAY)
                model_field_interpolate_i = np.flip(
                    model_field_interpolate_i, 0)
                self.model_field[key].append(model_field_interpolate_i)
            self.model_field[key] = np.asarray(self.model_field[key])

    def load_rain(self, file_name):
        """Read .nc file containing precipitation into memory

        To be called after loading model fields
        """
        rain_data = netCDF4.Dataset(file_name, "r", format="NETCDF4")
        rain = rain_data.variables["rr"]
        self.rain = []
        self.rain_units = rain.units

        assert(rain.dimensions[1] == "latitude")
        assert(rain.dimensions[2] == "longitude")

        #get the longitude and latitude
        latitude_array = np.round_(rain_data["latitude"], 2)
        longitude_array = np.round_(rain_data["longitude"], 2)
        latitude_array = np.flip(latitude_array)

        rain = ma.asarray(rain[:])
        rain = np.flip(rain, 1)

        #get the times
        time_array = rain_data["time"]
        time_array = netCDF4.num2date(time_array[:], time_array.units)

        self.mask = rain[0].mask
        #remove time points not covered by the model fields
        for i, time in enumerate(time_array):
            time = time.date()
            if time in self.time_array:
                self.rain.append(rain[i])
                assert(np.array_equal(self.mask, rain[i].mask))
        self.rain = ma.asarray(self.rain)

        rain_data.close()

    def load_topo(self, file_name):
        """Load topography data into memory

        Args:
            file_name: location of the .grib file
        """
        gdal_dataset = gdal.Open(file_name)
        raster_band = gdal_dataset.GetRasterBand(1)
        topo = raster_band.ReadAsArray() / GRAVITATIONAL_FIELD_STRENGTH
        grad = np.gradient(topo, LATITUDE_TOPO_ARRAY, LONGITUDE_TOPO_ARRAY)
        grad = np.sqrt(np.square(grad[0]) + np.square(grad[1]))

        #crop the topography data
        lat_index = []
        long_index = []
        for lat_i in LATITUDE_TOPO_ARRAY:
            lat_index.append(np.any(np.isclose(LATITUDE_ARRAY, lat_i)))
        for long_i in LONGITUDE_TOPO_ARRAY:
            long_index.append(np.any(np.isclose(LONGITUDE_ARRAY, long_i)))

        lat_index = np.where(lat_index)[0]
        long_index = np.where(long_index)[0]
        topo = topo[lat_index[0]:lat_index[-1]+1,
                    long_index[0]:long_index[-1]+1]
        grad = grad[lat_index[0]:lat_index[-1]+1,
                    long_index[0]:long_index[-1]+1]

        self.topography["elevation"] = topo
        self.topography["gradient"] = grad

        self.normalise_topography()

    def normalise_topography(self):
        """Store the normalisation of self.topography, to have mean 0, std 1, in
            self.topography_normalise
        """
        self.topography_normalise = {}
        for key, value in self.topography.items():
            topo_i = value.copy()
            shift = np.mean(topo_i)
            scale = np.std(topo_i, ddof=1)
            self.topography_normalise[key] = (topo_i - shift) / scale

    def find_nearest_latitude_longitude(self, coordinates):
        """Find nearest latitude and longitude

        Given coordinates of a place, returns the nearest latitude and longitude
            for a specific grid

        Args:
            coordinates: 2-element [latitude, longitude]
        Return:
            2-element array, latitude_index pointer to latitude_array and
                longitude_index pointer to longitude_array
        """
        #find longitude and latitude from the grid closest to coordinates
        min_longitude_error = float("inf")
        min_latitude_error = float("inf")
        #find latitude
        for i, latitude in enumerate(LATITUDE_ARRAY):
            latitude_error = abs(latitude - coordinates[0])
            if min_latitude_error > latitude_error:
                latitude_index = i
                min_latitude_error = latitude_error
        #find longitude
        for i, longitude in enumerate(LONGITUDE_ARRAY):
            longitude_error = abs(longitude - coordinates[1])
            if min_longitude_error > longitude_error:
                longitude_index = i
                min_longitude_error = longitude_error
        return(latitude_index, longitude_index)

    def get_latitude_longitude_city(self, city):
        return self.find_nearest_latitude_longitude(CITY_LOCATION[city])

    def get_latitude_longitude_random(self, rng):
        latitude_index = rng.randint(0, len(LATITUDE_ARRAY))
        longitude_index = rng.randint(0, len(LONGITUDE_ARRAY))
        return(latitude_index, longitude_index)

    def get_latitude_longitude_random_mask(self, rng):
        mask_index = np.where(np.logical_not(mask))
        random_index = rng.randint(0, len(mask_index[0]))
        latitude_index = mask_index[0][random_index]
        longitude_index = mask_index[1][random_index]
        return(latitude_index, longitude_index)

    def get_data(self, latitude_index, longitude_index):
        model_field = self.get_model_field(latitude_index, longitude_index)
        rain = self.get_rain(latitude_index, longitude_index)
        return (model_field, rain)

    def get_model_field(self, latitude_index, longitude_index):
        data_frame = {}
        for model_field_name, value in self.model_field.items():
            data_frame[model_field_name] = value[
                :, latitude_index, longitude_index]
        return pd.DataFrame(data_frame)

    def get_rain(self, latitude_index, longitude_index):
        return self.rain[:, latitude_index, longitude_index]

    def get_data_city(self, city):
        latitude_index, longitude_index = self.get_latitude_longitude_city(city)
        return self.get_data(latitude_index, longitude_index)

    def get_model_field_city(self, city):
        latitude_index, longitude_index = self.get_latitude_longitude_city(city)
        return self.get_model_field(latitude_index, longitude_index)

    def get_rain_city(self, city):
        latitude_index, longitude_index = self.get_latitude_longitude_city(city)
        return self.get_rain(latitude_index, longitude_index)

    def get_data_random(self, rng):
        latitude_index, longitude_index = (
            self.get_latitude_longitude_random_mask(rng))
        return self.get_data(latitude_index, longitude_index)

    def get_model_field_random(self, rng):
        latitude_index, longitude_index = self.get_latitude_longitude_random(
            rng)
        return self.get_model_field(latitude_index, longitude_index)

    def get_rain_random(self, rng):
        latitude_index, longitude_index = (
            self.get_latitude_longitude_random_mask(rng))
        return self.get_rain(latitude_index, longitude_index)

    def crop(self, lat, long):
        """Crop the data to a smaller area

        Args:
            lat: 2 array, used to index dimension 1
            long: 2 array, used to index dimension 2
        """
        for key, model_field in self.model_field.items():
            self.model_field[key] = model_field[
                :, lat[0]:lat[1], long[0]:long[1]]
        self.rain = self.rain[:, lat[0]:lat[1], long[0]:long[1]]
        self.mask = self.rain[0].mask
        for key in self.topography:
            self.topography[key] = self.topography[key][
                lat[0]:lat[1], long[0]:long[1]]
            self.topography_normalise[key] = self.topography_normalise[key][
                lat[0]:lat[1], long[0]:long[1]]

    def trim(self, time):
        """Trim the data of time points

        Args:
            time: 2 array, used to index time (dimension 0)
        """
        for key, model_field in self.model_field.items():
            self.model_field[key] = model_field[time[0]:time[1], :, :]
            self.model_field_coarse[key] = model_field[time[0]:time[1], :, :]
        self.rain = self.rain[time[0]:time[1], :, :]
        self.time_array = self.time_array[time[0]:time[1]]

    def trim_by_year(self, year_start, year_end):
        """Trim the data of time points, specified by the year

        Args:
            year_start: the year at the start of the trimmed data, inclusive
            year_end: the end year of the trimmed data, exclusive
        """
        #year_start inclusive
        #year_end exclusive
        array_index = np.zeros(len(self), dtype=np.bool);
        year_array = np.arange(year_start, year_end);
        for year in year_array:
            for i, date in enumerate(self.time_array):
                if date.year == year:
                    array_index[i] = True

        for key, model_field in self.model_field.items():
            self.model_field[key] = model_field[array_index, :, :]
        self.rain = self.rain[array_index, :, :]

        self.time_array = np.asarray(self.time_array)
        self.time_array = self.time_array[array_index]
        self.time_array = self.time_array.tolist()

    def generate_unmask_rain(self):
        for lat_i, long_i in self.generate_unmask_coordinates():
            yield self.rain[:, lat_i, long_i]

    def generate_unmask_coordinates(self):
        for lat_i in range(self.mask.shape[0]):
            for long_i in range(self.mask.shape[1]):
                if not self.mask[lat_i, long_i]:
                    yield [lat_i, long_i]

    def __len__(self):
        return len(self.time_array)

class GeneratorNc(object):
    """To generate the n_read_per_day readings for each day

    Used to smooth the n_read_per_day readings to get daily readings
    """

    def __init__(self, dataset, keys, n_read_per_day, time_array):
        self.dataset = dataset
        self.keys = keys
        self.n_read_per_day = n_read_per_day
        self.time_array = time_array
        self.n_time = len(time_array)

    def __iter__(self):
        for i in range(0, self.n_time, self.n_read_per_day):
            for key in self.keys:
                yield np.asarray(self.dataset[key][i:i+4])

class GeneratorGrib(object):
    """To generate the n_read_per_day readings for each day

    Used to smooth the n_read_per_day readings to get daily readings
    """

    def __init__(
        self, raster_array, raster_pointer, n_read_per_day, time_array):
        """
        Args:
            raster_array: gdal object
            raster_pointer: dictionary of pointers for GetRasterBand
        """
        self.raster_array = raster_array
        self.raster_pointer = raster_pointer
        self.n_read_per_day = n_read_per_day
        self.time_array = time_array
        self.n_time = len(time_array)

    def __iter__(self):
        #for each day
        for i_time in range(0, self.n_time, self.n_read_per_day):
            date = self.time_array[i_time].date()
            for i in range(self.n_read_per_day-1):
                assert(date == self.time_array[i_time+i+1].date())
            #for each model field
            for raster_pointer in self.raster_pointer.values():
                #get all readings for this day
                model_field_day = []
                for i_read in range(self.n_read_per_day):
                    raster = self.raster_array.GetRasterBand(
                        raster_pointer[i_time+i_read])
                    meta_data = raster.GetMetadata()
                    time_i = int(
                        meta_data["GRIB_VALID_TIME"].lstrip().rsplit()[0])
                    date_i = datetime.datetime.fromtimestamp(time_i).date()
                    assert(date_i == date)
                    model_field_day.append(raster.ReadAsArray())
                assert(len(model_field_day) == self.n_read_per_day)
                model_field_day = np.asarray(model_field_day)
                yield model_field_day
