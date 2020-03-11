from os import path
import pathlib

import joblib

import dataset

def main():
    
    path_here = pathlib.Path(__file__).parent.absolute()
    downscale = joblib.load(path.join(path_here, "downscale.gz"))
    test_set = dataset.IsleOfManTest()
    forecast = downscale.forecast(test_set, 1000)
    joblib.dump(forecast, path.join(path_here, "forecast.gz"))

if __name__ == "__main__":
    main()
