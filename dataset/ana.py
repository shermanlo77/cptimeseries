from os import path
import pathlib

from dataset import data

#training and test range for small data
TRAINING_RANGE_1 = [0, 3653]
TEST_RANGE_1 = [3653, 4018]

class AnaInterpolate1(data.Data):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_model_field(path.join(dir_to_data, "ana_input_1.nc"))
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))

class AnaDualExample1(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_model_field_interpolate_to_coarse(
            path.join(dir_to_data, "ana_input_1.nc"))
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))

class AnaDual1Training(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Mar20")
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_0.grib"))
        self.interpolate_model_field()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))

class AnaDual1Test(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Mar20")
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_2.grib"))
        self.interpolate_model_field()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))
        self.trim([3653-365, 3653])

class AnaDual5Test(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Mar20")
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_2.grib"))
        self.interpolate_model_field()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))
        self.trim([3653-5*365-1, 3653])

class AnaDual10Training(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Mar20")
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_1.grib"))
        self.interpolate_model_field()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))

class AnaDual10Test(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Mar20")
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_2.grib"))
        self.interpolate_model_field()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))

class AnaDualTraining(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Mar20")
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_0.grib"))
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_1.grib"))
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_2.grib"))
        self.interpolate_model_field()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))

class AnaDualTest(data.DataDualGrid):

    def __init__(self):
        super().__init__()

    def load_data(self):
        path_here = pathlib.Path(__file__).parent.absolute()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Mar20")
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_3.grib"))
        self.load_model_field(path.join(dir_to_data, "ana_cpdn_new_4.grib"))
        self.interpolate_model_field()
        dir_to_data = path.join(path_here, "..", "Data", "Rain_Data_Nov19")
        self.load_rain(path.join(dir_to_data, "rr_ens_mean_0.1deg_reg_v20"
            ".0e_197901-201907_uk.nc"))
        self.load_topo(path.join(dir_to_data, "topo_0.1_degree.grib"))
