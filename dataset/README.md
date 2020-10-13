# dataset
* Copyright (c) 2020 Sherman Lo
* MIT LICENSE

Under the hood code for handling `.nc` and `.grib` files and formatting it for `compound_poisson`.

The main superclasses are `data.Data` and `data.DataDualGrid`. It reads precipitation, model fields and tomography data from `.nc` and `.grib` files and converts them into custom `.gz` files which are saved onto a drive using `joblib`. At least 16 GB of RAM is required to do this.

Once the `.gz` file has been saved, data is read straight from the `.gz` file without the `.nc` and `.grib` files. This enables the entire data to be loaded onto RAM. This should be noted for developers who finds insufficient memory for larger datasets.

For a single location, the superclass `location.Location` is used.

See or modify the global constants in `ana` to specify the location where the `.nc` and `.grib` files should be read. The `.gz` files are saved in this package.

Implementations of these superclasses are found in the remaining modules and listed below.

## List of Datasets

* Single location
    * Simulated London
    * London
    * Cardiff
    * (Belfast, Dublin and Edinburgh have been coded in)
* Multiple locations
    * Isle of Man (6 spatial points) (useful for debugging code or as a small example)
    * Wales (305 spatial points)
    * British Isles (4806 spatial points), referred to as `Ana` in code

## List of Classes

* Single locations
    * `LondonSimulatedTraining` 1990-1999 inclusive
    * `LondonTraining` 1990-1999 inclusive
    * `LondonTest` 2000-2009 inclusive
    * `Cardiff1Training` 1999
    * `Cardiff2Training` 1998-1999 inclusive
    * `Cardiff5Training` 1995-1999 inclusive
    * `Cardiff10Training` 1990-1999 inclusive
    * `CardiffTraining` 1979-1999 inclusive
    * `CardiffTest` 2000-2019 inclusive
* Multiple locations
    * `IsleOfManTraining` 1990-1999 inclusive
    * `IsleOfManTest` 2000-2009 inclusive
    * `Wales1Training` 1999
    * `Wales2Training` 1998-1999 inclusive
    * `Wales5Training` 1995-1999 inclusive
    * `Wales10Training` 1990-1999 inclusive
    * `WalesTraining` 1979-1999 inclusive
    * `Wales10Test` 2000-2009 inclusive
    * `WalesTest` 2000-2019 inclusive
    * `AnaDual1Training` 1999
    * `AnaDual2Training` 1998-1999 inclusive
    * `AnaDual5Training` 1995-1999 inclusive
    * `AnaDual10Training` 1990-1999 inclusive
    * `AnaDualTraining` 1979-1999 inclusive
    * `AnaDual10Test` 2000-2009 inclusive
    * `AnaDualTest` 2000-2019 inclusive
