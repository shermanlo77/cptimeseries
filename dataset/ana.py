from os import path
import pathlib

from dataset import data

PATH_TO_MODEL_FIELD = path.join(pathlib.Path(__file__).parent.absolute(), "..", "Data", "Rain_Data_Mar20")
MODEL_FIELD_FILE_0 = path.join(PATH_TO_MODEL_FIELD, "ana_cpdn_new_0.grib")
MODEL_FIELD_FILE_1 = path.join(PATH_TO_MODEL_FIELD, "ana_cpdn_new_1.grib")
MODEL_FIELD_FILE_2 = path.join(PATH_TO_MODEL_FIELD, "ana_cpdn_new_2.grib")
MODEL_FIELD_FILE_3 = path.join(PATH_TO_MODEL_FIELD, "ana_cpdn_new_3.grib")
MODEL_FIELD_FILE_4 = path.join(PATH_TO_MODEL_FIELD, "ana_cpdn_new_4.grib")
PATH_TO_RAIN = path.join(pathlib.Path(__file__).parent.absolute(), "..", "Data", "Rain_Data_Nov19")
RAIN_FILE = path.join(PATH_TO_RAIN, "rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc")
PATH_TO_TOPO = PATH_TO_RAIN
TOPO_FILE = path.join(PATH_TO_TOPO, "topo_0.1_degree.grib")

class AnaDualTraining(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_model_field(MODEL_FIELD_FILE_0)
        self.load_model_field(MODEL_FIELD_FILE_1)
        self.load_model_field(MODEL_FIELD_FILE_2)
        self.interpolate_model_field()
        self.load_rain(RAIN_FILE)
        self.load_topo(TOPO_FILE)

class AnaDual10Training(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_model_field(MODEL_FIELD_FILE_2)
        self.interpolate_model_field()
        self.load_rain(RAIN_FILE)
        self.load_topo(TOPO_FILE)

class AnaDual1Training(AnaDual10Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.trim_by_year(1999, 2000);

class AnaDual2Training(AnaDual10Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.trim_by_year(1998, 2000);

class AnaDual5Training(AnaDual10Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.trim_by_year(1995, 2000);

class AnaDualTest(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_model_field(MODEL_FIELD_FILE_3)
        self.load_model_field(MODEL_FIELD_FILE_4)
        self.interpolate_model_field()
        self.load_rain(RAIN_FILE)
        self.load_topo(TOPO_FILE)

class AnaDual10Test(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_model_field(MODEL_FIELD_FILE_3)
        self.interpolate_model_field()
        self.load_rain(RAIN_FILE)
        self.load_topo(TOPO_FILE)
