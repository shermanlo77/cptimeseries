import tensorflow as tf
from netCDF4 import 
import netCDF4
class Generator():

    def __init__(self, fn = "", all_at_once=False, train_size=0.75 ):
        self.generator = None
        self.all_at_once = all_at_once
        self.fn = fn
        self.train_size = train_size
    
    def __call__(self):
        pass
    

class Generator_rain(Generator):

    def __init__(self, **generator_params):
        super(Generator_rain, self).__init__(**generator_params)

    def __call__(self):
        with netCDF4.Dataset(self.fn, "r", format="NETCDF4") as f:
            for chunk in f.variables['rr']
                yield chunk

class Generator_(Generator):

    def __init__(self, **generator_params):
        super(Generator_rain, self).__init__(**generator_params)

    def __call__(self):


