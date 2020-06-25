# dataset
* Copyright (c) 2020 Sherman Lo
* MIT LICENSE

Under the hood code for handling `.nc` and `.grib` files and formatting it for `compound_poisson`.

The main code is `data.py`. It converts `.nc` and `.grib` files to a custom `.gz` file using `joblib` and saves it onto disk. Most of the datasets require at least 16 GB of RAM to do this.

Once the `.gz` file has been saved, data is read straight from the `.gz` file without the `.nc` and `.grib` files, saving time.

The main difference when using the `.gz` files rather than the `.nc` and `.grib` files is that the entire data is loaded onto RAM. This trades computational time for RAM. This should be noted for developers if there is not sufficient memory for larger datasets.

## List of Datasets

* Single locations
    * Simulated London
    * London
    * Cardiff
    * (other cities have been coded in)
* Multiple locations
    * Isle of Man (6 spatial points) (useful for debugging code)
    * Wales (about ~350 spatial points)
    * British Isles (about ~3000 spatial points) (referred to as `Ana` which stands for reanalysis)

## List of objects

* Single locations
    * `LondonSimulatedTraining` 1980-1989 inclusive
    * `LondonTraining` 1980-1989 inclusive
    * `LondonTest` 1990-1999 inclusive
    * `CardiffTraining` 1979-1999 inclusive
    * `CardiffTrainingHalf` 1990-1999 inclusive
    * `CardiffTest` 2000-2019 inclusive
* Multiple locations
    * `IsleOfManTraining` 1980-1989 inclusive
    * `IsleOfManTest` 1990-1999 inclusive
    * `Wales10Training` 1980-1989 inclusive
    * `Wales10Test` 1990-1999 inclusive
    * `WalesTraining` 1979-1999 inclusive
    * `WalesTest` 2000-2019 inclusive
    * `AnaDual10Training` 1980-1989 inclusive
    * `AnaDual10Test` 1990-1999 inclusive
    * `AnaDualTraining` 1979-1999 inclusive
    * `AnaDualTest` 2000-2019 inclusive
