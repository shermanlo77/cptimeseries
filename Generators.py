import tensorflow as tf
import netCDF4
from netCDF4 import Dataset
import gdal
import numpy as np

#TODO:(akanni-ade): add ability to return long/lat variable  implement long/lat

"""
    Example of how to use
    import Generators

    rr_ens file 
    _filename = "Data/Rain_Data/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_djf_uk.nc"
    rain_gen = Generator_rain(_filename, all_at_once=True)
    data = next(iter(grib_gen))

    Grib Files
    _filename = 'Data/Rain_Data/ana_coarse.grib'
    grib_gen = Generators.Generator_grib(fn=_filename, all_at_once=True)
    data = next(iter(grib_gen))

    Grib Files Location:
    _filename = 'Data/Rain_Data/ana_coarse.grib'
    grib_gen = Generators.Generator_grib(fn=_filename, all_at_once=True)
    arr_long, arr_lat = grib_gen.locaiton()
    #now say you are investingating the datum x = data[15,125]
    #   to get the longitude and latitude you must do
    long, lat = arr_long(15,125), arr_lat(15,125)


"""
class Generator():
    
    def __init__(self, fn = "", all_at_once=False, train_size=0.75, channel=None ):
        self.generator = None
        self.all_at_once = all_at_once
        self.fn = fn
        self.channel = channel
    
    def yield_all(self):
        pass

    def yield_iter(self):
        pass

    def long_lat(self):
        pass

    def __call__(self):
        if(self.all_at_once):
            return self.yield_all()
        else:
            return self.yield_iter()
    

class Generator_rain(Generator):
    def __init__(self, **generator_params):
        super(Generator_rain, self).__init__(**generator_params)

    def yield_all(self):
        with Dataset(self.fn, "r", format="NETCDF4") as f:
            _data = f.variables['rr'][:]
            yield _data
    
    def yield_iter(self):
        with Dataset(self.fn, "r", format="NETCDF4") as f:
            for chunk in f.variables['rr']:
                yield chunk
        

class Generator_grib(Generator):
    """
        Creates a generator for the various grib files
    
        :param all_at_once: whether to return all data, or return data in batches

        :param channel: the desired channel of information in the grib file
            Default None, then concatenate all channels together and return
            If an integer return this band
    """

    def __init__(self, **generator_params):

        super(Generator_grib, self).__init__(**generator_params)

        self.ds = gdal.Open(self.fn)
        self.channel_count = self.ds.RasterCount
        

    def yield_all(self):
        li_channel_data = []
        if(self.channel==None):
            for i in range(1, self.channel_count+1):
                channel_data = self.ds.GetRasterBand(i).ReadAsArray()
                li_channel_data.append(channel_data)
            channel_data_all = np.stack( li_channel_data, axis=-1)
            yield channel_data_all
        else:
            channel_data = self.ds.GetRasterBand(self.channel).ReadAsArray()
            yield channel_data
    
    def yield_iter(self):
        raise NotImplementedError
        #TODO:(akanni-ade) Consider implementing if the grib files become significantly larger when Peter's add more data

    def location(self):
        """
        Returns a 2 1D arrays
            arr_long: Longitudes
            arr_lat: Latitdues
        Example of how to use:


        """
        GT = self.ds.GetGeoTransform()
        indices = np.indices( self.ds.GetRasterBand(1).ReadAsArray().shape )
        xp = GT[0] + indices[1]*GT[1] + indices[0]*GT[2]   
        yp = GT[3] + indices[0]*GT[4] + indices[1]*GT[5] 

        #shifting to centre of pixel
        xp += GT[1]/2
        yp += GT[4]/2
        return xp, yp
         


    


