"""Implementations of MCMC algorithms targetting discrete target distributions

MCMC algorithms design for targetting the latent variables z. Random walk
    Metropolis-Hastings is implemented in ZRwmh. Slice sampling is implemented
    in ZSlice, see Neal, R. M. (2003). Slice sampling. The Annals of
    Statistics, 31(3):705–741.

mcmc_abstract.Mcmc
    <- ZRwmh
    <- ZSlice
    <- ZMcmcArray

By design...
compound_poisson.downscale.Downscale
    <>-1 ZMcmcArray
but in terms of pointers/references, the following applies as well...
ZMcmcArray
    <>-1 compound_poisson.downscale.Downscale
"""

import math

import numpy as np

from compound_poisson.mcmc import mcmc_abstract


class ZRwmh(mcmc_abstract.Mcmc):
    """Metropolis-Hastings for the latent variables z

    Uses a non-adaptive proposal on integer space. The proposal distribution is
        the multivariate uniform distribution, sampling each and every z.

    For more attributes, see the superclass
    Attributes:
        z_parameter: proposal probability a single z moves
        n_propose: number of proposals
        n_accept: number of accept steps
        accept_array: acceptance rate at each proposal
    """

    def __init__(self, target, rng, n_sample=None, memmap_dir=None):
        super().__init__(np.int32, target, rng, n_sample, memmap_dir)
        self.z_parameter = 1 / target.get_n_dim()
        self.n_propose = 0
        self.n_accept = 0
        self.accept_array = []

    # implemented
    def sample(self):
        """Use Metropolis Hastings to sample z

        Uniform prior, proposal is step back, stay or step forward. Probability
            a single z moves is self.z_parameter.
        """
        # make a deep copy of the z, use it in case of rejection step
        self.save_state()
        state_before = self.state.copy()
        log_target_before = self.get_log_likelihood()
        # count number of times z moves to 1, also subtract if z moves from 1
        # use because the transition is non-symetric at 1
        n_transition_to_one = 0
        # for each z, proposal
        for i in range(len(self.state)):
            z = self.state[i]
            if z != 0:
                self.state[i] = self.propose_z(z)
                if z == 1 and self.state[i] == 2:
                    n_transition_to_one -= 1
                elif z == 2 and self.state[i] == 1:
                    n_transition_to_one += 1
        try:
            # metropolis hastings
            # needed for get_joint_log_likelihood()
            self.update_state()
            log_target_after = self.get_log_likelihood()
            # to cater for non-symetric transition
            log_target_after += n_transition_to_one * math.log(2)
            if not self.is_accept_step(log_target_before, log_target_after):
                # rejection step, replace the z before this MCMC step
                self.revert_state()
            else:
                # accept step only if at least one z has moved
                if not np.array_equal(state_before, self.state):
                    self.n_accept += 1
        # treat numerical errors as a rejection
        except(ValueError, OverflowError):
            self.revert_state()
        # keep track of acceptance rate
        self.n_propose += 1
        self.accept_array.append(self.n_accept/self.n_propose)

    def propose_z(self, z):
        """Return a proposed z

        Return a transitioned z. Probability that it will move is
            self.proposal_z_parameter, otherwise stay where it is. Cannot move
            to 0.

        Args:
            z: z at this step of MCMC

        Returns:
            proposed z
        """
        if self.rng.rand() < self.z_parameter:
            if z == 1:  # cannot move to 0, so move to 2
                return 2
            elif self.rng.rand() < 0.5:
                return z+1
            else:
                return z-1
        else:
            return z


class ZSlice(mcmc_abstract.Mcmc):
    """Slice sampling on the z latent variables

    Does slice sampling for a randomly selected z from TimeSeries. See Neal
        (2003) for the slice samplng algorithm. Note: this implementation is
        not a generalised one.

    For more attributes, see the superclass
    Attributes:
    """

    def __init__(self, target, rng, n_sample=None, memmap_dir=None):
        super().__init__(np.int32, target, rng, n_sample, memmap_dir)
        self.n_propose = 0
        self.slice_width_array = []
        self.non_zero_index = np.nonzero(self.target.time_series.y_array)[0]

    def sample(self):
        """Use slice sampling

        Implemented
        See Neal (2003)
        Select a random z_t in the time series, then move it using slice
            sampling
        """
        # pick a random non-zero latent variable to sample
        t = self.rng.choice(self.non_zero_index)
        # get the latent variable and the log likelihood
        z_t = self.state[t]
        ln_l_before = self.get_log_likelihood()
        # sample the vertical line
        ln_y = math.log(self.rng.rand()) + ln_l_before
        # proposal range contains the limits of what integers to propose, both
        # excluse for now
        proposal_range = [z_t, z_t+1]

        # keep decreasing proposal_range[0] until it is zero or when the log
        # likelihood is less than ln_y
        # NOTE: proposal_range is exclusive for now
        ln_l_propose = ln_l_before
        is_in_slice = True
        while is_in_slice:
            proposal_range[0] -= 1
            if proposal_range[0] == 0:
                is_in_slice = False
            else:
                # break when there is a numerical problem with this proposal
                try:
                    self.state[t] = proposal_range[0]
                    self.update_state()
                    ln_l_propose = self.get_log_likelihood()
                    if ln_l_propose < ln_y:
                        is_in_slice = False
                except(ValueError, OverflowError):
                    is_in_slice = False
        # NOTE: now make proposal range left inclusive
        proposal_range[0] += 1

        # keep increasing proposal_range[1] until the log likelihood is less
        # than ln_y
        # NOTE: proposal_range is left inclusive and right exclusive for the
        # rest of this method
        ln_l_propose = ln_l_before
        is_in_slice = True
        while is_in_slice:
            # break when there is a numerical problem with this proposal
            try:
                self.state[t] = proposal_range[1]
                self.update_state()
                ln_l_propose = self.get_log_likelihood()
                if ln_l_propose < ln_y:
                    is_in_slice = False
                else:
                    proposal_range[1] += 1
            except(ValueError, OverflowError):
                is_in_slice = False

        # set the proposed z
        self.state[t] = self.rng.randint(proposal_range[0], proposal_range[1])
        self.update_state()
        self.slice_width_array.append(proposal_range[1] - proposal_range[0])
        self.n_propose += 1


class ZMcmcArray(mcmc_abstract.Mcmc):
    """MCMC for multiple ZSlice objects, eg one for each location

    Does MCMC on all z in Downscale. Does this by accessing the member
        variables of the Downscale object.

    Attributes:
        downscale: Downscale object containing array of TimeSeries objects, one
            for each location
    """

    def __init__(self, downscale):
        """
        Args:
            downscale: Downscale object
        """
        super().__init__(np.int32)
        self.downscale = downscale
        n_dim = len(downscale) * downscale.area_unmask
        self.instantiate_memmap(
            downscale.memmap_dir, downscale.n_sample, n_dim)

    # override
    def get_target_class(self):
        return ""

    # implemented
    def sample(self):
        """All locations in self.downscale to sample z (in parallel)
        """
        time_series_array = self.downscale.pool.map(
            static_sample_z, self.downscale.generate_unmask_time_series())
        self.downscale.replace_unmask_time_series(time_series_array)

    # override
    def add_to_sample(self):
        """Override to extract z from all locations and append it
        """
        z_sample = []
        for time_series in self.downscale.generate_unmask_time_series():
            z_sample.append(time_series.z_array.copy())
        z_sample = np.asarray(np.concatenate(z_sample), self.dtype)
        self.append(z_sample)


def static_sample_z(time_series):
    time_series.z_mcmc.sample()
    return time_series
