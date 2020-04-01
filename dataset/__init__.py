from os import path
import pathlib

import joblib

from dataset import data
from dataset.data import AnaDualExample1
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
