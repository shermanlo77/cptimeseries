import joblib
import matplotlib.pyplot as plot
import numpy as np
from get_data import London
from print_figure import print_time_series, print_forecast

def main():
    
    time_series = joblib.load("results/mcmc_london.zlib")
    time_series.burn_in = 90000
    
    directory = "../figures/london/mcmc/"
    london = London()
    
    parameter_name = time_series.get_parameter_vector_name()
    
    chain = np.asarray(time_series.parameter_sample)
    for i in range(time_series.n_parameter):
        chain_i = chain[:,i]
        plot.figure()
        plot.plot(chain_i)
        plot.ylabel(parameter_name[i])
        plot.xlabel("Sample number")
        plot.savefig(directory + "chain_parameter_" + str(i) + ".pdf")
        plot.close()
    
    chain = []
    for i in range(len(time_series.z_sample)):
        chain.append(np.mean(time_series.z_sample[i]))
    plot.figure()
    plot.plot(chain)
    plot.ylabel("Mean of latent variables")
    plot.xlabel("Sample number")
    plot.savefig(directory + "chain_z.pdf")
    plot.close()
    
    plot.figure()
    plot.plot(np.asarray(time_series.accept_reg_array))
    plot.ylabel("Acceptance rate of parameters")
    plot.xlabel("Parameter sample number")
    plot.savefig(directory + "accept_parameter.pdf")
    plot.close()

    plot.figure()
    plot.plot(np.asarray(time_series.accept_z_array))
    plot.ylabel("Acceptance rate of latent variables")
    plot.xlabel("Latent variable sample number")
    plot.savefig(directory + "accept_z.pdf")
    plot.close()
    
    print_forecast(time_series, 
                   london.get_rain_training(), 
                   np.asarray(london.get_model_field_test()),
                   london.get_rain_test(),
                   directory)
    
    time_series.simulate()
    print_time_series(time_series, directory + "fitted_")

if __name__ == "__main__":
    main()
