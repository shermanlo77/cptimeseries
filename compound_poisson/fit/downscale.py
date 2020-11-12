import compound_poisson
from compound_poisson import multiprocess
from compound_poisson import forecast
from compound_poisson.fit import fitter

class Fitter(fitter.Fitter):

    def __init__(self, downscale_class, directory=""):
        super().__init__(downscale_class, directory)

    def initial_fit(self, dataset, seed, n_sample=None, pool=None):
        #dataset is Data object (see dataset module)
        if pool is None:
            pool = multiprocess.Serial()
        downscale = self.model_class(dataset, (5, 5))
        downscale.set_rng(seed)
        downscale.set_memmap_dir(self.result_dir)
        if not n_sample is None:
            downscale.n_sample = n_sample
        downscale.fit(pool)
        return downscale

    def print_mcmc(self, downscale, dataset=None, pool=multiprocess.Serial):
        #arg dataset not used
        downscale.print_mcmc(self.figure_dir, pool)

    def do_forecast(self, downscale, dataset, n_simulation, pool):
        #dataset is Data object (see dataset module)
        downscale.forecast(dataset, n_simulation, pool)

    def print_forecast(self, downscale, dataset, pool):
        #dataset is Data object (see dataset module)
        printer = forecast.print.Downscale(
            downscale.forecaster, self.figure_dir, pool)
        printer.print()

class FitterDownscale(Fitter):

    def __init__(self, directory=""):
        super().__init__(compound_poisson.MultiSeries, directory)

################################################################################
#                              DEPRECATED
################################################################################

class FitterDownscaleDual(Fitter):

    def __init__(self, directory=""):
        super().__init__(compound_poisson.DownscaleDual, directory)
