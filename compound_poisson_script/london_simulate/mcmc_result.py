import os
import sys

import joblib
import matplotlib.pyplot as plot
import numpy as np

sys.path.append(os.path.join("..", ".."))
import compound_poisson as cp
import dataset

def main():
    
    time_series = joblib.load(os.path.join("result", "mcmc.gz"))
    directory = os.path.join("figure", "mcmc")
    try:
        os.mkdir(directory)
    except(FileExistsError):
        pass
    
    london = dataset.LondonSimulated80()
    true_parameter = london.time_series.get_parameter_vector()
    parameter_name = time_series.get_parameter_vector_name()
    
    chain = np.asarray(time_series.parameter_mcmc.sample_array)
    for i in range(time_series.n_parameter):
        chain_i = chain[:,i]
        plot.figure()
        plot.plot(chain_i)
        plot.hlines(true_parameter[i], 0, len(chain)-1)
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
    plot.plot(np.asarray(time_series.parameter_mcmc.accept_array))
    plot.ylabel("Acceptance rate of parameters")
    plot.xlabel("Parameter sample number")
    plot.savefig(os.path.join(directory, "accept_parameter.pdf"))
    plot.close()

    plot.figure()
    plot.plot(np.asarray(time_series.z_mcmc.accept_array))
    plot.ylabel("Acceptance rate of latent variables")
    plot.xlabel("Latent variable sample number")
    plot.savefig(os.path.join(directory, "accept_z.pdf"))
    plot.close()
    

if __name__ == "__main__":
    main()
