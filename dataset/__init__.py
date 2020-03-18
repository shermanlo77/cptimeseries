from os import path
import pathlib

import joblib

from dataset import data
from dataset.data import AnaDualExample0
from dataset.data import AnaInterpolate1
from dataset.data import ANGLE_RESOLUTION 
from dataset.data import CITY_LOCATION
from dataset.data import GRAVITATIONAL_FIELD_STRENGTH
from dataset.data import LATITUDE_ARRAY
from dataset.data import LONGITUDE_ARRAY
from dataset.data import LATITUDE_TOPO_ARRAY
from dataset.data import LONGITUDE_TOPO_ARRAY
from dataset.data import LATITUDE_COARSE_ARRAY
from dataset.data import LONGITUDE_COARSE_ARRAY
from dataset.data import RADIUS_OF_EARTH
from dataset.data import RESOLUTION
from dataset.isle_of_man import IsleOfMan
from dataset.isle_of_man import IsleOfManTest
from dataset.isle_of_man import IsleOfManTraining
from dataset.london80 import London80
from dataset.london80 import LondonSimulated80

def init_ana_interpolate():
    ana = data.Data()
    path_here = pathlib.Path(__file__).parent.absolute()
    ana.load_model_field(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "ana_input_1.nc"))
    ana.load_rain(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"))
    ana.load_topo(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "topo_0.1_degree.grib"))
    joblib.dump(ana, path.join(path_here, data.AnaInterpolate1.__name__+".gz"))

def init_ana_dual_example_0():
    ana = data.DataDualGrid()
    path_here = pathlib.Path(__file__).parent.absolute()
    ana.load_model_field_coarse_example(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "ana_coarse.grib"))
    ana.load_rain(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"))
    ana.load_topo(path.join(path_here, "..", "Data", "Rain_Data_Nov19", "topo_0.1_degree.grib"))
    joblib.dump(ana, path.join(path_here, data.AnaDualExample0.__name__+".gz"))
