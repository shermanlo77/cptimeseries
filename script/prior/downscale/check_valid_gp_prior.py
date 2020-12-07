import numpy as np
from scipy import linalg

import compound_poisson
import dataset

def main():
    downscale = compound_poisson.Downscale(dataset.AnaDual10Training())

    precision_array = np.linspace(250, 300, 50)
    for precision in precision_array:
        cov_chol = downscale.square_error.copy()
        cov_chol *= -precision / 2
        cov_chol = np.exp(cov_chol)
        print("Precision: " + str(precision))
        try:
            cov_chol = linalg.cholesky(cov_chol, True)
            print("Success")
        except:
            print("Fail")



if __name__ == "__main__":
    main()
