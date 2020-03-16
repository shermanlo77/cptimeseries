import os
from os import path
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import dataset

def main():
    
    path_here = pathlib.Path(__file__).parent.absolute()
    figure_dir = path.join(path_here, "figure")
    if not path.isdir(figure_dir):
        os.mkdir(figure_dir)
    
    data = dataset.AnaInterpolate1()
    n_unmask = np.sum(np.logical_not(data.mask))
    area = data.mask.shape[0] * data.mask.shape[1]
    
    correlation = np.zeros((n_unmask, n_unmask))
    
    lat_i_array, long_i_array = np.where(np.logical_not(data.mask))
    rain = data.rain
    for i in range(n_unmask):
        print(i)
        lat_i = lat_i_array[i]
        long_i = long_i_array[i]
        correlation[i,i] = 1
        for j in range(i+1, n_unmask):
            lat_j = lat_i_array[j]
            long_j = long_i_array[j]
            correlation_ij = np.corrcoef(rain[:, lat_i, long_i],
                                         rain[:, lat_j, long_j])
            correlation[i,j] = correlation_ij[0,1]
            correlation[j,i] = correlation[i,j]
    
    plt.figure()
    plt.imshow(correlation)
    plt.colorbar()
    plt.savefig(path.join(figure_dir, "rain_correlation.pdf"))
    plt.close()
    

if __name__ == "__main__":
    main()
