"""Base abstract superclass is Mcmc. Given a Target distribution, it can get and
    store MCMC samples in a memmap. Memmap are chosen to reduce memory
    consumption.

Also contains the class ReadOnlyFromSlice. Instances are dummy objects which
    can read MCMC samples from a larger memmap. It does this by wrapping around
    a Mcmc object which has that larger memmap. The suffix FromSlice is a
    reference to the Python slice object, not to be confused with slice
    sampling.

The function do_gibbs_sampling() does Gibbs sampling when given an array of Mcmc
    objects.

Mcmc
    <- mcmc_parameter.Rwmh
    <- mcmc_parameter.Elliptical
    <- mcmc_z.ZRwmh
    <- mcmc_z.ZSlice
    <- mcmc_z.ZMcmcArray

Mcmc
    <>1- target.Target

compound_poisson.time_series_mcmc.TimeSeriesMcmc
    <>-1..* Mcmc
compound_poisson.downscale.Downscale
    <>-1..* Mcmc
"""

import datetime
import math
import os
from os import path

import numpy as np

class Mcmc(object):
    """Abstract class for MCMC

    Given a Target distribution, it can sample and store MCMC samples in a
        memmap.
    Memmap are chosen to reduce memory consumption. To handle the memmap, they
        should be read (using read_to_write_memmap() or read_memmap()) and del
        aka closed (using del_memmap()) to avoid having too many files opened
        where appropriate. Instantiating a MCMC object will instantiate a
        memmap already opened for you.
    See the numpy.memmap documentation
        https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        for more information on the memmap used here.
    The posterior distribution (otherwise known as the target distribution),
        likelihood and the state (position vector of the current state) can be
        obtained via a wrapper class Target. This is passed via the constructor.

    To use MCMC to read/write from a larger memmap, do not pass n_sample and
        memmap_dir via the constructor. After instantiation, call
        set_memmap_slice() to set where to read the memmap and slice it to view
        a requested range of dimensions. This is useful for a single location
        analysis from a larger multi-location inference for example.

    How to use:
        Pass the Target distribution (and other required parameters) through
            the constructor. The current state of the MCMC chain is stored
            in self.state and the MCMC samples are stored as a memmap in
            self.sample_array. The memmap is allocated drive space in the
            constructor.
        Call step() to do a MCMC step, updates self.state and save that sample
            or state to the memmap.
        Call sample() to do a MCMC step and updates self.state without saving
             the sample or state to the memmap.
        Call add_to_sample() to save the state to the memmap (can be repeated).
        To store more MCMC samples than the allocated space for the memmap in
            self.sample_array, call extend_memmap().

    Implemented methods:
        append(): add a MCMC sample to sample_array
        __len__(): length of sample_array, CAUTION: this is not the number of
            MCMC samples during sampling
        __getitem__(): get MCMC sample from sample_array

    Methods to implement:
        sample() to modify self.state after a MCMC step

    Attributes:
        dtype: type of what is being sampled, used to read and write memmap
        n_sample: number of MCMC samples
        memmap_path: location of the memmap
        memmap_path_old: temporary member variable, location of the original
            memmap after calling extend_memmap()
        target: Target object which can evalute the posterior and obtain the
            state vector
        rng: numpy.random.RandomState object
        n_dim: the length of the state vector
        state: the current state vector
        sample_array: memmap, array of state vector representing the chain
            dim 0: for each sample
            dim 1: for each dimension
        sample_pointer: during sampling, 0th dimensional pointer of sample_array
            to where to store the next MCMC sample
        slice_index: optional slice object, if not None only use specific index
            of dimension 1 for sample_array. To be set after instantiation.
            Useful for reading MCMC samples from a larger sampling scheme, eg
            a location from a multi-location inference.
        n_dim_parent: optional, size of dimension 1 when initally reading
            the memmap in memmap
    """

    def __init__(self,
                 dtype,
                 target=None,
                 rng=None,
                 n_sample=None,
                 memmap_dir=None):
        """
        Args:
            dtype: type of what is being sampled, eg numpy.int32, numpy.float64
            target: Target object containing the posterior distribution. Passing
                None is reserved by ZMcmcArray because in Downscale, each
                location has a dummy ZSlice object which does not do any MCMC
                sampling by itself.
            rng: numpy.random.RandomState object
            n_sample: number of MCMC samples to sample. If None, a memmap is not
                instantiate and MCMC samples cannot be stored without further
                action.
            memmap_dir: where the memmap lives
        """

        self.dtype = dtype
        self.n_sample = None
        self.memmap_path = memmap_dir
        self.memmap_path_old = None
        self.target = target
        self.rng = rng
        self.n_dim = None
        self.state = None
        self.sample_array = None
        self.sample_pointer = 0
        self.slice_index = None
        self.n_dim_parent = None

        if not target is None:
            n_dim = target.get_n_dim()
            self.state = target.get_state()

            if not n_sample is None:
                self.instantiate_memmap(memmap_dir, n_sample, n_dim)

    def instantiate_memmap(self, directory, n_sample, n_dim):
        """Instantiate the member variable sample_array as a memmmap

        Instantiate the member variable sample_array as a memmmap. Also assign
            member variables n_sample and n_dim. The file name of the memmap
            is a combination of the self.target type, time and memory address
            to form a unique and identifiable name.

        Args:
            directory:
            n_sample: number of MCMC samples to store, this sets the 0th
                dimension of the sample_array memmap
            n_dim: number of dimensions the state has (or the length of the
                state vector), this sets the 1st dimension of the sample_array
                memmap
        """
        file_name = self.get_target_class()
        file_name = self.make_memmap_file_name(file_name)
        self.n_sample = n_sample
        self.n_dim = n_dim
        self.memmap_path = path.join(directory, file_name)
        self.sample_array = np.memmap(self.memmap_path,
                                      self.dtype,
                                      "w+",
                                      shape=(n_sample, n_dim))

    def get_target_class(self):
        """Return the full name , including the module, of the type of
            self.target
        """
        return (type(self.target).__module__.split(".")[-1]
            + "_" + type(self.target).__name__)

    def make_memmap_file_name(self, name):
        """Return a propose file name for memmap for self.sample_array

        File name consist of: underscore, class name, datetime and memory
            address with extension .dat
        """
        datetime_id = str(datetime.datetime.now())
        datetime_id = datetime_id.replace("-", "")
        datetime_id = datetime_id.replace(":", "")
        datetime_id = datetime_id.replace(" ", "")
        datetime_id = datetime_id[0:14]
        file_name = ("_" + type(self).__name__ + "_" + name + "_" + datetime_id
            + "_" + str(id(self)) + ".dat")
        return file_name

    def extend_memmap(self, n_sample):
        """Extend sample_array to store more MCMC samples

        Extend self.sample_array to store more MCMC samples by instantiating a
            new memmap with a larger 0th dimension. Copy values from the old
            memmap to the new memmap. This can be slow for large samples.
        Does not delete the old memmap but does keep track of the path of the
            old memmap by storing it in self.memmap_path_old.
        Copying from one memmap to another is a design choice in case something
            goes wrong during a MCMC run, eg time out on a server.

        Args:
            n_sample: new length of sample_array
        """
        if n_sample > self.n_sample:
            n_sample_old = self.n_sample
            self.read_memmap()
            sample_array_old = self.sample_array
            self.memmap_path_old = self.memmap_path
            self.instantiate_memmap(path.dirname(self.memmap_path),
                                    n_sample,
                                    self.n_dim)
            self.sample_array[0:n_sample_old] = sample_array_old
            #del is required to stop reading the file of the old memmap
            del sample_array_old

    def delete_old_memmap(self):
        """DANGEROUS: Actually deletes the file containing the old MCMC samples
        """
        if path.exists(self.memmap_path_old):
            os.remove(self.memmap_path_old)

    def read_to_write_memmap(self):
        """Read the memmap (using r+ mode) and assign it to self.sample_array.
            Allows the memmap to read and write.
        """
        self.load_memmap("r+")

    def read_memmap(self):
        """Read only the memmap and assign it to self.sample_array.
        """
        self.load_memmap("r")

    def load_memmap(self, mode):
        """Load the memmap

        If n_dim_parent exist, then load a memmap from a larger inference
            scheme. This is then sliced in the 1st dimension according to
            self.slice_index

        Args:
            mode: eg "r", "r+"
        """
        shape = None
        if self.n_dim_parent is None:
            shape = (self.n_sample, self.n_dim)
        else:
            shape = (self.n_sample, self.n_dim_parent)

        self.sample_array = np.memmap(self.memmap_path,
                                      self.dtype,
                                      mode,
                                      shape=shape)
        #slice the first dimension
        if not self.slice_index is None:
            sliced_sample_array = self.sample_array[:, self.slice_index]
            del self.sample_array
            self.sample_array = sliced_sample_array

    def del_memmap(self):
        """Delete reference to the memmap

        Close the file to the memmap. This does not actually delete the file.
        """
        del self.sample_array
        self.sample_array = None

    def add_to_sample(self):
        """Add a MCMC sample to sample_array.

        Copy the state vector and append it to sample_array. This adds a
            duplicate to sample_array. Used when doing a MCMC step or when
            sampling another component in Gibbs sampling.
        """
        if not self.sample_array is None:
            self.append(self.state.copy())

    def update_state(self):
        """Update state

        Update the target distribution with the current state of the chain, used
            to eg evaluate the log likelihood at this current state
        """
        self.target.update_state(self.state)

    def get_log_likelihood(self):
        """Return the log likelihood
        """
        return self.target.get_log_likelihood()

    def get_log_target(self):
        """Get log target

        Return the log posterior
        """
        return self.target.get_log_target()

    def save_state(self):
        """Save state

        Save a copy of the current state vector in self.target. This can be
            recovered using revert_state(), eg. a rejection step.
        """
        self.target.save_state()

    def revert_state(self):
        """Revert state

        Revert the state vector and the target back to what it was before
            calling save_state(). Used for eg. a rejection step.
        """
        self.target.revert_state()
        self.state = self.target.get_state()

    def sample(self):
        """Sample once from the posterior

        Sample once from the posterior and update state
        """
        raise NotImplementedError("Sampling scheme inc Mcmc not implemented")

    def step(self):
        """Do a MCMC step

        Sample once from the posterior and append it to sample_array
        """
        self.sample()
        self.add_to_sample()

    def is_accept_step(self, log_target_before, log_target_after):
        """Metropolis-Hastings accept and reject

        Args:
            log_target_before: log likelihood at the state now
            log_target_after log likelihood of the proposed state

        Returns:
            True if this is the accept step, else False
        """
        accept_prob = math.exp(log_target_after - log_target_before)
        if self.rng.rand() < accept_prob:
            return True
        else:
            return False

    def discard_sample(self, n_keep):
        """ONLY FOR DEBUGGING PURPOSES: Discard initial mcmc samples to save
            hard disk space

        For each mcmc, make a new memmap and store the last n_keep mcmc samples.
            As a precaution, you will have to delete the old memmap file
            yourself.

        Args:
            n_keep: number of mcmc samples to keep (from the end of the chain)
        """
        self.read_memmap()
        sample_array_old = self.sample_array
        self.memmap_path_old = self.memmap_path
        self.instantiate_memmap(path.dirname(self.memmap_path),
                                n_keep,
                                self.n_dim)
        self.sample_array[:] = sample_array_old[-n_keep:len(sample_array_old)]
        self.n_sample = n_keep
        self.sample_pointer = self.n_sample
        del sample_array_old
        self.del_memmap()

    def set_memmap_slice(
        self, n_sample, n_dim_parent, memmap_path, slice_index):
        """Set the member variables n_sample, n_dim_parent, memmap_path and
            slice_index so that it can read and write to/from an already existing
            larger memmap.

        Args:
            n_sample: number of samples, size of the 0th dimension of
                sample_array
            n_dim_parent: TOTAL number of dimensions, size of the 1st
                dimension of sample_array before slicing
            memmap_path: the location of the memmap
            slice_index: slice object, which of the 1st dimension to use
        """
        self.n_sample = n_sample
        self.n_dim_parent = n_dim_parent
        self.memmap_path = memmap_path
        self.slice_index = slice_index

    def return_self(self):
        """Return itself, used by ReadOnlyFromSlice
        """
        return self

    def __len__(self):
        """Length of sample_array, CAUTION: this is not the number of MCMC
            samples during sampling, use self.sample_pointer for that.
        """
        return len(self.sample_array)

    def __getitem__(self, index):
        return self.sample_array[index]

    def append(self, state):
        #cast state to the appropriate type for memmap
        if self.dtype.__name__ == "int32":
            state = state.astype(self.dtype)
        self.sample_array[self.sample_pointer] = state
        self.sample_pointer += 1

def do_gibbs_sampling(mcmc_array, n_sample, rng, gibbs_weight,
        is_initial_sample=True):
    """Do Gibbs sampling

    Given the mcmc of all components, do a Gibbs sampling. One sample is
        obtained by choosing one component at random and that component does a
        mcmc step, the rest of the components stay where they are, aka random
        scan.
    Requires the Mcmc objects in the mcmc_array to open their memmaps
        beforehand. They are all del or closed afterwards.

    Args:
        mcmc_array: array of mcmc objects
        n_sample: number of samples to be obtained
        rng: random number generator
        gibbs_weight: array of probabilities up to a constant, same length as
            mcmc_array, each element correspond to a component, proportional to
            probability that component gets sampled in a Gibbs step
        is_initial_sample: boolean, the initial value is a sample if True
    """
    prob_sample = np.asarray(gibbs_weight)
    prob_sample = prob_sample / np.sum(prob_sample)
    #initial value is a sample
    if is_initial_sample:
        print("Sample initial")
        for mcmc in mcmc_array:
            mcmc.add_to_sample()
        n_sample -= 1
    #Gibbs sampling
    for i_step in range(n_sample):
        print("Sample", i_step)
        #random scan
        mcmc_index = rng.choice(len(mcmc_array), 1, p=prob_sample)
        for i_mcmc, mcmc in enumerate(mcmc_array):
            if i_mcmc == mcmc_index:
                mcmc.step()
            else:
                mcmc.add_to_sample()
    #close the files for all mcmc objects
    for mcmc in mcmc_array:
        mcmc.del_memmap()
