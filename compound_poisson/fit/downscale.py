import compound_poisson
from compound_poisson import multiprocess
from compound_poisson import forecast
from compound_poisson.fit import fitter


class Fitter(fitter.Fitter):

    def __init__(self, downscale_class, directory="", suffix=None):
        super().__init__(downscale_class, directory, suffix)
        self.use_gp = False
        self.topo_key = None

    def initial_fit(self, dataset, seed, n_sample=None, pool=None):
        # dataset is Data object (see dataset module)
        if pool is None:
            pool = multiprocess.Serial()
        downscale = self.model_class(dataset, (5, 5))
        downscale.set_rng(seed)
        downscale.set_memmap_dir(self.result_dir)
        if n_sample is not None:
            downscale.n_sample = n_sample
        downscale.fit(pool)
        return downscale

    def print_mcmc(self, downscale, dataset=None, pool=multiprocess.Serial):
        # arg dataset not used
        downscale.print_mcmc(self.figure_dir, pool)

    def set_post_gp(self, use_gp, topo_key=None):
        self.use_gp = use_gp
        self.topo_key = topo_key

    def do_forecast(self, downscale, dataset, n_simulation, pool):
        # dataset is Data object (see dataset module)
        if self.topo_key is None:
            downscale.forecast(dataset, n_simulation, pool, self.use_gp)
        else:
            downscale.forecast(
                dataset, n_simulation, pool, self.use_gp, self.topo_key)

    def print_forecast(self, downscale, dataset, pool):
        # dataset is Data object (see dataset module)
        printer = forecast.print.Downscale(
            downscale.forecaster, self.figure_dir, pool)
        printer.print()


class FitterDownscale(Fitter):

    def __init__(self, directory="", suffix=None):
        super().__init__(compound_poisson.Downscale, directory, suffix)


class FitterMultiSeries(Fitter):

    def __init__(self, directory="", suffix=None):
        super().__init__(compound_poisson.MultiSeries, directory, suffix)


class FitterDownscaleDeepGp(Fitter):

    def __init__(self, directory="", suffix=None):
        super().__init__(compound_poisson.DownscaleDeepGp, directory, suffix)
