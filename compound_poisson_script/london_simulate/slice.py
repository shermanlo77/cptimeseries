import os
import sys

import joblib
import matplotlib.pyplot as plot
import numpy as np
import numpy.random as random

sys.path.append(os.path.join("..", ".."))
import compound_poisson as cp
import dataset

def main():
    
    rng = random.RandomState(np.uint32(3976111046))
    name = "slice"
    result_dir = "result"
    figure_directory = os.path.join("figure", name)
    
    if not os.path.isdir("result"):
        os.path.mkdir("result")
    if not os.path.isdir("figure"):
        os.path.mkdir("figure")
    if not os.path.isdir(figure_directory):
        os.path.mkdir(figure_directory)
    
    result_file = os.path.join(result_dir, name + ".gz")
    if not os.path.isfile(result_file):
        london = dataset.London80()
        model_field, rain = london.get_data_training()
        time_series = cp.TimeSeriesSlice(model_field, rain, (5, 5), (5, 5))
        time_series.time_array = london.get_time_training()
        time_series.rng = rng
        time_series.fit()
        joblib.dump(time_series, result_file)
    else:    
        time_series = joblib.load(result_file)
    
    print_chain(time_series, figure_directory)
    
def print_chain(time_series, directory):
    parameter_name = time_series.get_parameter_vector_name()
    
    chain = np.asarray(time_series.parameter_mcmc.sample_array)
    for i in range(time_series.n_parameter):
        chain_i = chain[:,i]
        plot.figure()
        plot.plot(chain_i)
        plot.ylabel(parameter_name[i])
        plot.xlabel("Sample number")
        plot.savefig(
            os.path.join(directory, "chain_parameter_" + str(i) + ".pdf"))
        plot.close()
    
    chain = []
    z_chain = np.asarray(time_series.z_mcmc.sample_array)
    for z in z_chain:
        chain.append(np.mean(z))
    plot.figure()
    plot.plot(chain)
    plot.ylabel("Mean of latent variables")
    plot.xlabel("Sample number")
    plot.savefig(os.path.join(directory, "chain_z.pdf"))
    plot.close()
    
    plot.figure()
    plot.plot(np.asarray(time_series.parameter_mcmc.n_reject_array))
    plot.ylabel("Number of rejects in parameter slicing")
    plot.xlabel("Parameter sample number")
    plot.savefig(os.path.join(directory, "n_reject_parameter.pdf"))
    plot.close()

    plot.figure()
    plot.plot(np.asarray(time_series.z_mcmc.slice_width_array))
    plot.ylabel("Latent variable slice width")
    plot.xlabel("Latent variable sample number")
    plot.savefig(os.path.join(directory, "slice_width_z.pdf"))
    plot.close()

if __name__ == "__main__":
    main()
