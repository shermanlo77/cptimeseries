import datetime
import math
import os
from os import path

import numpy as np

class Mcmc(object):
    """Abstract class for MCMC

    Posterior distribution (otherwise known as the target distribution),
        likelihood and the state (position vector of the current state) can be
        obtained via a wrapper class Target. A Target object and a rng is to be
        passed via the constructor.
    Methods append, __len__, __getitem__ implemented which uses sample_array

    Attributes:
        target: wrapper object which can evalute the posterior and obtain the
            state vector
        rng: numpy.random.RandomState object
        n_dim: the length of the state vector
        state: the current state vector
        sample_array: array of state vector, representing the chain
    """

    def __init__(self,
                 dtype,
                 target=None,
                 rng=None,
                 n_sample=None,
                 memmap_dir=None):
        #None target to be used by ZMcmcArray
        #None n_sample won't store mcmc sample
        #Potential development: None memmap_path to use np.darray
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

        if not target is None:
            n_dim = target.get_n_dim()
            self.state = target.get_state()

            if not n_sample is None:
                self.instantiate_memmap(memmap_dir, n_sample, n_dim)

    def instantiate_memmap(self, directory, n_sample, n_dim):
        #instantiate memmap to member variable, also assign member variables
            #n_sample and n_dim
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
        return (type(self.target).__module__.split(".")[-1]
            + "_" + type(self.target).__name__)

    def make_memmap_file_name(self, name):
        """Prefix underscore, class name, append datetime and memory address and
            .dat
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
        """Extend sample_array to store more mcmc, makes a new memmap and copies
            values from the old memmap

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
            del sample_array_old

    def delete_old_memmap(self):
        """DANGEROUS: Actually delete the file containing the old MCMC samples
        """
        if path.exists(self.memmap_path_old):
            os.remove(self.memmap_path_old)

    def read_to_write_memmap(self):
        self.sample_array = np.memmap(self.memmap_path,
                                      self.dtype,
                                      "r+",
                                      shape=(self.n_sample, self.n_dim))

    def del_memmap(self):
        """Delete reference to the memmap
        """
        del self.sample_array
        self.sample_array = None

    def read_memmap(self):
        self.sample_array = np.memmap(self.memmap_path,
                                      self.dtype,
                                      "r",
                                      shape=(self.n_sample, self.n_dim))

    def add_to_sample(self):
        """Add to sample

        Copy the state vector and append it to sample_array. This adds a
            duplicate to sample_array. Used for rejection step or when sampling
            another component in Gibbs sampling
        """
        if not self.sample_array is None:
            self.append(self.state.copy())

    def update_state(self):
        """Update state

        Update the target distribution with the current state of the chain
        """
        self.target.update_state(self.state)

    def get_log_likelihood(self):
        """Get log likelihood
        """
        return self.target.get_log_likelihood()

    def get_log_target(self):
        """Get log target

        Return the log posterior
        """
        return self.target.get_log_target()

    def save_state(self):
        """Save state

        Save a copy of the current state vector. This can be recovered using
            revert_state()
        """
        self.target.save_state()

    def revert_state(self):
        """Revert state

        Revert the state vector and the target back to what it was before
            calling save_state
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
            log_target_before
            log_target_after

        Returns:
            True if this is the accept step, else False
        """
        accept_prob = math.exp(log_target_after - log_target_before)
        if self.rng.rand() < accept_prob:
            return True
        else:
            return False

    def discard_sample(self, n_keep):
        """Discard initial mcmc samples to save hard disk space

        For each mcmc, make a new memmap and store the last n_keep mcmc samples.
            For saftey reasons, you will have to delete the old memmap file
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

    def __len__(self):
        return len(self.sample_array)

    def __getitem__(self, index):
        return self.sample_array[index]

    def append(self, state):
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
        mcmc_index = rng.choice(len(mcmc_array), 1, p=prob_sample)
        for i_mcmc, mcmc in enumerate(mcmc_array):
            if i_mcmc == mcmc_index:
                mcmc.step()
            else:
                mcmc.add_to_sample()
    for mcmc in mcmc_array:
        mcmc.del_memmap()
