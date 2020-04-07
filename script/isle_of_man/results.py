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
    downscale.read_memmap()

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

    chain = np.asarray(downscale.parameter_gp_mcmc.sample_array)
    key = downscale.parameter_gp_target.state.keys()
    for i, key in enumerate(downscale.parameter_gp_target.state):
        chain_i = chain[:, i]
        plt.plot(chain_i)
        plt.xlabel("sample number")
        plt.ylabel(key)
        plt.savefig(path.join(chain_dir, key + ".pdf"))
        plt.close()

    chain = []
    for time_series in downscale.generate_unmask_time_series():
        chain.append(np.mean(time_series.z_mcmc.sample_array, 1))
    plt.plot(np.transpose(np.asarray(chain)))
    plt.xlabel("sample number")
    plt.ylabel("mean z")
    plt.savefig(path.join(chain_dir, "z.pdf"))
    plt.close()

if __name__ == "__main__":
    main()
