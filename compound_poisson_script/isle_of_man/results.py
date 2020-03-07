import os
import sys

import joblib
import matplotlib.pyplot as plot
import numpy as np
from numpy import ma
import pdb

sys.path.append(os.path.join("..", ".."))
import compound_poisson as cp
import dataset

def main():
    
    downscale = joblib.load("downscale.gz")
    if not os.path.isdir("figure"):
        os.mkdir("figure")
    if not os.path.isdir(os.path.join("figure", "chain")):
        os.mkdir(os.path.join("figure", "chain"))
    
    chain = np.asarray(downscale.parameter_mcmc.sample_array)
    area_unmask = downscale.area_unmask
    parameter_name = (
        downscale.time_series_array[0][0].get_parameter_vector_name())
    for i in range(downscale.n_parameter):
        chain_i = chain[:, i*area_unmask : (i+1)*area_unmask]
        plot.plot(chain_i)
        plot.xlabel("sample number")
        plot.ylabel(parameter_name[i])
        plot.savefig(
            os.path.join("figure", "chain", "parameter_" + str(i) + ".pdf"))
        plot.close()
    
    chain = np.asarray(downscale.precision_mcmc.sample_array)
    for i in range(chain.shape[1]):
        chain_i = chain[:, i]
        plot.plot(chain_i)
        plot.xlabel("sample number")
        plot.ylabel("parameter precision " + str(i))
        plot.savefig(
            os.path.join("figure", "chain", "precision_" + str(i) + ".pdf"))
        plot.close()
    
    chain = np.asarray(downscale.gp_mcmc.sample_array)
    plot.plot(chain)
    plot.xlabel("sample number")
    plot.ylabel("gp precision")
    plot.savefig(
        os.path.join("figure", "chain", "gp_precision.pdf"))
    plot.close()
    
    chain = []
    for lat_i in range(downscale.shape[0]):
        for long_i in range(downscale.shape[1]):
            if not downscale.mask[lat_i, long_i]:
                time_series = downscale.time_series_array[lat_i][long_i]
                chain.append(np.mean(time_series.z_mcmc.sample_array, 1))
    plot.plot(np.transpose(np.asarray(chain)))
    plot.xlabel("sample number")
    plot.ylabel("mean z")
    plot.savefig(
        os.path.join("figure", "chain", "z.pdf"))
    plot.close()

if __name__ == "__main__":
    main()
