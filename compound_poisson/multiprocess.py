import multiprocessing

from mpi4py import MPI
from mpi4py import futures

N_PROCESSESS = None
CHUNKSIZE = 1

class Pool(object):

    def __init__(self):
        self.pool = None
        self.is_mpi = False

        comm = MPI.COMM_WORLD
        if comm.size > 1:
            self.pool = futures.MPIPoolExecutor()
            self.is_mpi = True
        else:
            self.pool = multiprocessing.Pool(N_PROCESSESS)

    def map(self, function, parameters):
        results = self.pool.map(function, parameters)
        if self.is_mpi:
            results = list(results)
        return results

    def join(self):
        if not self.is_mpi:
            self.pool.join()

    def close(self):
        if not self.is_mpi:
            self.pool.close()
