import os
import sys

import matplotlib.pyplot as plot

sys.path.append(os.path.join("..", ".."))
import compound_poisson as cp
import dataset

def main():
    
    try:
        os.mkdir("figure")
    except(FileExistsError):
        pass
    
    directory = "figure"
    london = dataset.London80()
    model_field = london.model_field
    rain = london.rain
    time_series = cp.TimeSeries(model_field, rain)
    time_series.time_array = london.time_array
    cp.print.time_series(time_series, directory)
    
    for i in range(time_series.n_model_field):
        plot.figure()
        plot.plot(time_series.time_array, time_series.x[:, i])
        plot.xlabel("Time")
        plot.ylabel(time_series.model_field_name[i])
        plot.savefig(os.path.join(directory, "model_field_" + str(i) + ".pdf"))
        plot.close()
    
if __name__ == "__main__":
    main()
