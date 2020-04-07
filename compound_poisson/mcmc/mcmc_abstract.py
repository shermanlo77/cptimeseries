import datetime
import math
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

    def __init__(self, dtype=None, n_sample=None, memmap_path=None, target=None,
        rng=None):
        #none target is used for wrapper mcmc such as ZMcmcArray
        self.dtype = dtype
        self.n_sample = n_sample
        self.memmap_path = memmap_path
        self.target = target
        self.rng = rng
        self.n_dim = None
        self.state = None
        self.sample_array = None
        self.sample_pointer = 0

        if not target is None:
            self.n_dim = self.target.get_n_dim()
            self.state = target.get_state()

            datetime_id = str(datetime.datetime.now())
            datetime_id = datetime_id.replace("-", "")
            datetime_id = datetime_id.replace(":", "")
            datetime_id = datetime_id.replace(" ", "")
            datetime_id = datetime_id[0:14]
            file_name = ("_" + type(self).__name__ + type(target).__name__
                + "_" + datetime_id + "_" + str(id(self)) + ".dat")
            self.memmap_path = path.join(memmap_path, file_name)
            self.sample_array = np.memmap(self.memmap_path,
                                          dtype,
                                          "w+",
                                          shape=(self.n_sample, self.n_dim))

    def load_to_write(self):
        self.sample_array = np.memmap(self.memmap_path,
                                      self.dtype,
                                      "r+",
                                      shape=(self.n_sample, self.n_dim))

    def del_memmap(self):
        del self.sample_array

    def load_memmap(self):
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

    def __len__(self):
        return len(self.sample_array)

    def __getitem__(self, index):
        return self.sample_array[index]

    def append(self, state):
        if self.dtype.__name__ == "int32":
            state = state.astype(self.dtype)
        self.sample_array[self.sample_pointer] = state
        self.sample_pointer += 1

def do_gibbs_sampling(mcmc_array, n_sample, rng):
    #initial value is a sample
    print("Sample 0")
    for mcmc in mcmc_array:
        mcmc.add_to_sample()
    #Gibbs sampling
    for i_step in range(n_sample-1):
        print("Sample", i_step+1)
        #select random component
        mcmc_index = rng.randint(0, len(mcmc_array))
        for i_mcmc, mcmc in enumerate(mcmc_array):
            if i_mcmc == mcmc_index:
                mcmc.step()
            else:
                mcmc.add_to_sample()
    for mcmc in mcmc_array:
        mcmc.del_memmap()
