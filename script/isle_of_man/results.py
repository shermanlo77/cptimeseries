import os
from os import path
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

def main():
    
    path_here = pathlib.Path(__file__).parent.absolute()
    downscale = joblib.load(path.join(path_here, "downscale.gz"))
    
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    chain_dir = path.join(figure_dir, "chain")
    if not path.isdir(chain_dir):
        os.mkdir(chain_dir)
    
    chain = np.asarray(downscale.parameter_mcmc.sample_array)
    area_unmask = downscale.area_unmask
    parameter_name = (
        downscale.time_series_array[0][0].get_parameter_vector_name())
    for i in range(downscale.n_parameter):
        chain_i = chain[:, i*area_unmask : (i+1)*area_unmask]
        plt.plot(chain_i)
        plt.xlabel("sample number")
        plt.ylabel(parameter_name[i])
        plt.savefig(path.join(chain_dir, "parameter_" + str(i) + ".pdf"))
        plt.close()
    
    chain = np.asarray(downscale.precision_mcmc.sample_array)
    for i in range(chain.shape[1]):
        chain_i = chain[:, i]
        plt.plot(chain_i)
        plt.xlabel("sample number")
        plt.ylabel("parameter precision " + str(i))
        plt.savefig(path.join(chain_dir, "precision_" + str(i) + ".pdf"))
        plt.close()
    
    chain = np.asarray(downscale.gp_mcmc.sample_array)
    plt.plot(chain)
    plt.xlabel("sample number")
    plt.ylabel("gp precision")
    plt.savefig(path.join(chain_dir, "gp_precision.pdf"))
    plt.close()
    
    chain = []
    for lat_i in range(downscale.shape[0]):
        for long_i in range(downscale.shape[1]):
            if not downscale.mask[lat_i, long_i]:
                time_series = downscale.time_series_array[lat_i][long_i]
                chain.append(np.mean(time_series.z_mcmc.sample_array, 1))
    plt.plot(np.transpose(np.asarray(chain)))
    plt.xlabel("sample number")
    plt.ylabel("mean z")
    plt.savefig(path.join(chain_dir, "z.pdf"))
    plt.close()

if __name__ == "__main__":
    main()
