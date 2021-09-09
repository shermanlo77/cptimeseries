"""Wrapper classes for different multiprocess objects such as MPI.
"""

from mpi4py import futures
import multiprocessing


class Serial(object):
    """Does tasks in serial (for debugging purposes)
    """

    def __init__(self):
        pass

    def map(self, function, parameters):
        results = []
        for parameter in parameters:
            results.append(function(parameter))
        return results

    def broadcast(self, value):
        return Broadcast(value)

    def join(self):
        pass


class MPIPoolExecutor(Serial):
    """Uses mpi4py.futures.MPIPoolExecutor

    To call, use, for example,: mpiexec -n 8 python3 -m mpi4py.futures
        script.py
    """

    def __init__(self):
        self.pool = futures.MPIPoolExecutor()

    def map(self, function, parameters):
        results = self.pool.map(function, parameters)
        results = list(results)
        return results

    def join(self):
        self.pool.shutdown()


class Pool(Serial):
    """Uses multiprocessing.Pool

    Number of workers defined by the global variable N_PROCESSESS, default is
        None which uses the default number of workers
    """

    def __init__(self, n_process=None):
        self.pool = multiprocessing.Pool(n_process)

    def map(self, function, parameters):
        results = self.pool.map(function, parameters)
        return results

    def join(self):
        self.pool.close()
        self.pool.join()


class Broadcast(object):
    """Dummy Broadcast object
    """

    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value
