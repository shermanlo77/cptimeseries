import os
import sys

import joblib

sys.path.append("..")
import dataset

def main():
    data = dataset.Data()
    data.load_model_field(os.path.join("..", "Data", "Rain_Data_Nov19", "ana_input_1.nc"))
    data.load_rain(os.path.join("..", "Data", "Rain_Data_Nov19", "rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"))
    joblib.dump(data, os.path.join("ana_input_1.gz"))
    
if __name__ == "__main__":
    main()
