import multiprocessing

from abcpy import backends

N_PROCESSESS = None

class Pool(object):

    def __init__(self):
        self.pool = None
        self.is_mpi = False
        try:
            self.pool = backends.BackendMPI()
            self.is_mpi = True
        except ValueError:
            self.pool = multiprocessing.Pool(N_PROCESSESS)

    def map(self, function, parameters):
        if self.is_mpi:
            pds = self.pool.parallelize(parameters)
            pds_map = self.pool.map(function, pds)
            results = self.pool.collect(pds_map)
        else:
            results = self.pool.map(function, parameters)
        return results

    def join(self):
        if not self.is_mpi:
            self.pool.join()

    def close(self):
        if not self.is_mpi:
            self.pool.close()
