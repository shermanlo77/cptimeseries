import matplotlib.pyplot as plot
from get_data import LondonSimulation
from print_figure import print_time_series

def main():
    directory = "../figures/london/simulation/"
    london = LondonSimulation()
    time_series = london.get_time_series_training()
    print_time_series(time_series, directory)
    
    for i in range(time_series.n_model_fields):
        plot.figure()
        plot.plot(time_series.time_array, time_series.x[:, i])
        plot.xlabel("Time")
        plot.ylabel(time_series.model_field_name[i])
        plot.savefig(directory + "model_field_" + str(i) + ".pdf")
        plot.close()
    
if __name__ == "__main__":
    main()
