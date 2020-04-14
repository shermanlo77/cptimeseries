import math

import numpy as np
from numpy import linalg
from scipy import stats

from compound_poisson.mcmc import target

class TargetModelField(target.Target):
    """Density information for the model fields for a time point

    Attributes:
        downscale: pointer to parent
        time_step: time point
        prior_mean: vector, Gaussian process mean, for each model field and
            point in space, length n_total_parameter
        model_field_shift: array, normalisation value for each model field
        model_field_scale: array, normalisation value for each model field
        area_unmask: number of points on the find grid
        n_model_field: number of model fields
        n_total_parameter: number of model fields x area_unmask
    """

    def __init__(self, downscale, time_step):
        self.downscale = downscale
        self.time_step = time_step
        self.prior_mean = None
        self.model_field_shift = []
        self.model_field_scale = []
        self.area_unmask = downscale.area_unmask
        self.n_model_field = downscale.n_model_field
        self.n_total_parameter = self.area_unmask * self.n_model_field

        self.prior_mean = np.empty(self.n_total_parameter)
        for i, model_field_coarse in enumerate(
            downscale.model_field_coarse.values()):
            #model_field_shift is mean of coarse model fields
            #model_field_scale is std of coarse model_fields
            #prior_mean is constant at the mean
            self.model_field_shift.append(
                np.mean(model_field_coarse[time_step]))
            self.model_field_scale.append(
                np.std(model_field_coarse[time_step], ddof=1))
            self.prior_mean[i*self.area_unmask:(i+1)*self.area_unmask] = (
                self.model_field_shift[i])

    def get_n_dim(self):
        #implmented
        return self.n_total_parameter

    def get_state(self):
        #implemented
        return self.downscale.get_model_field(self.time_step)

    def update_state(self, state):
        #implemented
        self.downscale.set_model_field(state, self.time_step)
        #required for likelihood evaluation
        self.downscale.update_all_cp_parameters()

    def get_log_prior(self):
        #required for mcmc on the gaussian process parameters
        gp_target = self.downscale.model_field_gp_target
        z = self.get_state()
        z -= self.prior_mean
        ln_prior_term = []
        chol = gp_target.cov_chol
        precision = gp_target.state["precision"]
        #evaluate the Normal density for each model_field, each model_field has
            #a covariance matrix which represent the correlation in space. There
            #is no correlation between different model_field. In other words,
            #if a vector contains all model_field for all points in space, the
            #covariance matrix has a block diagonal structure. Evaluating
            #the density with a block digonal covariance matrix can be done
            #using a for loop
        for i in range(self.n_model_field):
            #z is the parameter - mean
            z_i = z[i*self.area_unmask : (i+1)*self.area_unmask]

            z_i = linalg.lstsq(chol, z_i, rcond=None)[0]
            ln_prior_term.append(-0.5 * precision * np.dot(z_i, z_i))
        #reminder: det(cX) = c^d det(X)
        #reminder: det(L*L) = det(L) * det(L)
        #reminder: 1/sqrt(precision) = standard deviation or scale
        ln_det_cov = (2*np.sum(np.log(np.diagonal(chol))
            - self.area_unmask * math.log(precision)))
        return np.sum(ln_prior_term) - 0.5 * self.n_model_field * ln_det_cov

    def set_mean(self, mean):
        #used when the gp precision changes which changes the mean
        self.prior_mean = mean

    def get_prior_mean(self):
        #implemented
        return self.prior_mean

    def simulate_from_prior(self, rng):
        #implemented
        gp_target = self.downscale.model_field_gp_target
        precision = gp_target.state["precision"]
        cov_chol = gp_target.cov_chol
        state_rng = self.simulate_from_prior_given_gp(precision, cov_chol, rng)
        return state_rng[0]

    def simulate_from_prior_given_gp(self, precision, cov_chol, rng):
        """Simulate from the prior given gp precision parameters

	    Args:
            precision: precision to scale cov_chol
            cov_chol: cholesky of the covariance matrix
            rng: random number generator

        Return:
            2 array, 0th element simulation, 1st element rng after being used
        """
        model_field_vector = np.empty(self.area_unmask * self.n_model_field)
        scale = 1/math.sqrt(precision)
        #simulate each model_field, correlation only in space, not between
            #model_fields
        for i in range(self.n_model_field):
            model_field_i = np.asarray(rng.normal(size=self.area_unmask))
            model_field_i = np.matmul(cov_chol, model_field_i)
            model_field_i *= self.model_field_scale[i]
            model_field_vector[i*self.area_unmask : (i+1)*self.area_unmask] = (
                model_field_i)
        model_field_vector *= scale
        model_field_vector += self.prior_mean
        return [model_field_vector, rng]

class TargetModelFieldArray(target.Target):
    """Contains array of TargetModelField objects, one for each time step

    Mimics the target.Target class, uses density information from all
        TargetModelField objects. __iter__(), __len__(), __getitem__(),
        __setitem__() implemented.

    Attributes:
        downscale: pointer to parent
        model_field_target_array: array of TargetModelField objects
        rng_array: array of rng, one for each time point
        area_unmask: number of points on fine grid
        n_model_field: number of model fields
        n_parameter_i: number of dimensions for a time point
    """

    def __init__(self, downscale):
        self.downscale = downscale
        self.model_field_target_array = []
        self.rng_array = downscale.spawn_rng(len(downscale))
        self.area_unmask = downscale.area_unmask
        self.n_model_field = downscale.n_model_field
        self.n_parameter_i = self.area_unmask * self.n_model_field

        for i in range(len(downscale)):
            self.model_field_target_array.append(TargetModelField(downscale, i))

    def get_n_dim(self):
        #implemented
        n_parameter = 0
        for target in self:
            n_parameter += target.get_n_dim()
        return n_parameter

    def get_state(self):
        #implemented
        #stack all model fields on top of each other
        state = []
        for target in self:
            state.append(target.get_state())
        state = np.concatenate(state)
        return state

    def update_state(self, state):
        #implemented
        for i, target in enumerate(self):
            state_i = state[i*self.n_parameter_i : (i+1)*self.n_parameter_i]
            self.downscale.set_model_field(state_i, i)
        self.downscale.update_all_cp_parameters()

    def get_log_likelihood(self):
        #self.downscale.update_all_cp_parameters() call befordhand required
        return self.downscale.get_log_likelihood()

    def get_log_prior(self):
        #implemented
        ln_l = []
        for target in self:
            ln_l.append(target.get_log_prior())
        ln_l = np.sum(ln_l)
        return ln_l

    def get_log_target(self):
        #implemented
        return self.get_log_likelihood() + self.get_log_prior()

    def update_mean(self):
        """Update the mean for each model field for each time point

        Update the mean for each model field for each time point. Done by
            instantiating a GpRegression object and calling regress. Required as
            a change in the gp precision parameters will change the mean.
        """
        gp_array = []
        gp_target = self.downscale.model_field_gp_target
        pool = self.downscale.pool
        #distribute k_11 and k_12 to all workers
        k_11 = pool.broadcast(gp_target.k_11)
        k_12 = pool.broadcast(gp_target.k_12)
        for target in self:
            gp_array.append(GpRegressionMessage(target, k_11, k_12))
        mean = self.downscale.pool.map(GpRegressionMessage.regress, gp_array)
        self.set_mean(mean)

    def set_mean(self, mean):
        """Set the mean for each time step
        """
        for i, target in enumerate(self):
            mean_i = mean[i*self.n_parameter_i : (i+1)*self.n_parameter_i]
            target.set_mean(mean_i)

    def simulate_from_prior(self, rng):
        #implemented
        #provided rng not used as this is to be parallised
        message_array = []
        pool = self.downscale.pool
        gp_target = self.downscale.model_field_gp_target

        #perpare message
        cov_chol = pool.broadcast(gp_target.cov_chol)
        precision = pool.broadcast(gp_target.state["precision"])
        for i, target in enumerate(i, self):
            #use spawned rng
            rng = self.rng_array[i]
            message_array.append(
                GpSimulateMessage(target, cov_chol, precision, rng))

        #returns array of 2 array containing simulated model field and rng after
            #use, need to extract them and put them accordingly
        state_rng_array = pool.map(
            GpSimulateMessage.simulate_from_prior, message_array)
        state = []
        self.rng_array = []
        for state_rng in state_rng_array:
            state.append(state_rng[0])
            self.rng_array.append(state_rng[1])
        state = np.concatenate(state)
        return state

    def get_prior_mean(self):
        #implemented
        #stack model fields on top of each other
        mean = []
        for target in self:
            mean.append(target.get_prior_mean())
        mean = np.concatenate(mean)
        return mean

    def __iter__(self):
        return iter(self.model_field_target_array)

    def __len__(self):
        return len(self.model_field_target_array)

    def __getitem__(self, index):
        return self.model_field_target_array[index]

    def __setitem__(self, index, value):
        self.model_field_target_array[index] = value

class TargetGp(target.Target):
    """Contains gp precision parameters and density information

    Attributes:
        downscale: pointer to parent
        prior: dictionary of scipy.stats prior distributions
        state: dictionary of the gp precision parameters
        state_before: copy of state, used by mcmc
        model_field_coarse: dictionary of model fields on the coarse grid
        topography_coarse: dictionary of normalised topography on the coarse
            grid
        topography: dictionary of normalised topography on the fine grid
        mask: boolean matrix with shape for the fine grid, true area in the grid
            is on water, ie no rain data on water
        square_error_coarse: square error of normalise topography on coarse
            grid, matrix dimension n_coarse x n_coarse
        square_error_fine: square error of normalise topography on fine grid,
            matrix dimension area_unmask x area_unmask
        square_error_coarse_fine: square error of normalise topography between
            fine and coarse grid, matrix dimension n_coarse x area_unmask
        cov_chol: cholesky of kernel matrix (not covariance matrix), dimensions
            area_unmask x area_unmask
        k_11: kernel matrix, dimensions n_coarse x n_coarse
        k_12: kernel matrix, dimensions n_coarse x area_unmask
        k_22: kernel matrix, dimensions area_unmask x area_unmask
        mean_before: used by mcmc, copy of self.downscale.model_field_target
            .get_prior_mean()
        cov_chol_before: used by mcmc, copy of cov_chol
    """

    def __init__(self, downscale):
        self.downscale = downscale
        self.prior = {
            "precision": get_precision_prior(),
            "gp_precision": target.get_gp_precision_prior(),
            "reg_precision": get_regulariser_precision_prior(),
        }
        self.state = {}
        self.state_before = None

        self.model_field_coarse = downscale.model_field_coarse
        self.topography_coarse = downscale.topography_coarse_normalise
        self.topography = downscale.topography_normalise

        self.mask = downscale.mask
        self.square_error_coarse = np.zeros(
            (downscale.n_coarse, downscale.n_coarse))
        self.square_error_fine = downscale.square_error
        self.square_error_coarse_fine = np.empty(
            (downscale.n_coarse, downscale.area_unmask))
        self.cov_chol = None
        self.k_11 = np.identity(downscale.n_coarse)
        self.k_12 = np.zeros((downscale.n_coarse, downscale.area_unmask))
        #k_22 is not needed, only required to calculate posterior covariance

        self.mean_before = None
        self.cov_chol_before = None

        #set initalse state to be the prior mean
        for key, prior in self.prior.items():
            self.state[key] = prior.mean()

        n_coarse = downscale.n_coarse

        #work out square error matrices
        for topo_i in self.topography_coarse.values():
            topo_i = topo_i.flatten()
            for i in range(n_coarse):
                for j in range(i+1, n_coarse):
                    self.square_error_coarse[i, j] += math.pow(
                        topo_i[i] - topo_i[j], 2)
                    self.square_error_coarse[j, i] = (
                        self.square_error_coarse[i,j])

        unmask = np.logical_not(self.mask).flatten()
        for key, topo_fine in self.topography.items():
            topo_coarse = self.topography_coarse[key].flatten()
            topo_fine = topo_fine.flatten()
            topo_fine = topo_fine[unmask]
            for i in range(n_coarse):
                for j in range(downscale.area_unmask):
                    self.square_error_coarse_fine[i, j] += math.pow(
                        topo_coarse[i] - topo_fine[j], 2)

    def get_n_dim(self):
        #implemented
        return len(self.prior)

    def get_state(self):
        #implemented
        return np.asarray(list(self.state.values()))

    def update_state(self, state):
        #implemented
        for i, key in enumerate(self.state):
            self.state[key] = state[i]
        #when updating the gp precision parameters, the posterior mean and
            #covariance changes, update them
        #update covariance first as the method updates the member variables k_11
            #and k_12, required for updating the mean
        self.save_cov_chol()
        self.downscale.model_field_target.update_mean()

    def get_log_likelihood(self):
        #implemented
        return self.downscale.model_field_target.get_log_prior()

    def get_log_target(self):
        #implemented
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        ln_pdf = []
        for key, prior in self.prior.items():
            ln_pdf.append(prior.logpdf(self.state[key]))
        return np.sum(ln_pdf)

    def save_state(self):
        #implemented
        #make copy of gp precisions, posterior covariance and posterior mean
            #of all moedel fields for all time steps
        self.state_before = self.state.copy()
        self.cov_chol_before = self.cov_chol.copy()
        self.mean_before = self.downscale.model_field_target.get_prior_mean()

    def revert_state(self):
        #implemeneted
        #revert posterior covariance and posterior mean
        self.state = self.state_before
        self.cov_chol_before = self.cov_chol_before
        self.downscale.model_field_target.set_mean(self.mean_before)

    def simulate_from_prior(self, rng):
        #implemented
        prior_simulate = []
        for prior in self.prior.values():
            prior.random_state = rng
            prior_simulate.append(prior.rvs())
        return np.asarray(prior_simulate)

    def get_kernel_matrix(self, square_error):
        """Return kernel matrix given matrix of squared errors
        """
        kernel_matrix = square_error.copy()
        kernel_matrix *= -self.state["gp_precision"] / 2
        kernel_matrix = np.exp(kernel_matrix)
        return kernel_matrix

    def save_cov_chol(self):
        """Update member variables k_11, k_12 and cov_chol
        """
        self.k_11 = self.get_kernel_matrix(self.square_error_coarse)
        self.regularise_kernel(self.k_11)
        self.k_12 = self.get_kernel_matrix(self.square_error_coarse_fine)

        k_11_chol = linalg.cholesky(self.k_11)
        cov_chol = linalg.lstsq(k_11_chol, self.k_12, rcond=None)[0]
        cov_chol = np.matmul(cov_chol.T, cov_chol)
        cov_chol = self.get_kernel_matrix(self.square_error_fine) - cov_chol
        cov_chol = linalg.cholesky(cov_chol)
        self.cov_chol = cov_chol

    def regularise_kernel(self, kernel):
        """Add 1/sqrt(reg_precision) * identity matrix to given matrix
        """
        regulariser = 1/math.sqrt(self.state["reg_precision"])
        for i in range(kernel.shape[0]):
            kernel[i, i] += regulariser

class GpRegressionMessage(object):
    """For doing gaussian process regression

    For doing gaussian process regression and return the mean, designed to be
        temporary and used in distributed memory

    Attributes:
        model_field_coarse: array of matrices containing model fields on the
            coarse grid, one model field for each entry
        shift_array: array of normalising factors, one for each model field
        scale_array: array of normalising factors, one for each model field
        k_11: broadcast, kernel matrix, dimensions n_coarse x n_coarse
        k_12: broadcast, kernel matrix, dimensions n_coarse x area_unmask
    """

    def __init__(self, model_field_target, k_11, k_12):
        """
        Args:
            model_field_target: TargetModelField object (not
                TargetModelFieldArray)
            k_11: kernel matrix, dimensions n_coarse x n_coarse
            k_12: kernel matrix, dimensions n_coarse x area_unmask
        """
        self.model_field_coarse = []
        self.shift_array = model_field_target.model_field_shift
        self.scale_array = model_field_target.model_field_scale
        self.k_11 = k_11
        self.k_12 = k_12
        #extracted model fields for this particular time point
        for model_field in (
            model_field_target.downscale.model_field_coarse.values()):
            self.model_field_coarse.append(
                model_field[model_field_target.time_step])

    def regress(self):
        """Return posterior mean of the Gaussian process regression

        Return posterior mean of the Gaussian process regression for each model
            field, returns vector, model fields are stacked on top of each other
        """
        mean = [] #array of means for each model field
        k_11 = self.k_11.value()
        k_12 = self.k_12.value()
        k_12_t = k_12.T
        #model fields have covariance matrix over space, but different model
            #fields are not correlated with each other, thus regress for each
            #model fields indepedently
        for i, model_field_i in enumerate(self.model_field_coarse):
            model_field_i = model_field_i.flatten()
            model_field_i -= self.shift_array[i]
            model_field_i /= self.scale_array[i]
            mean_i = np.matmul(
                k_12_t, linalg.lstsq(k_11, model_field_i, rcond=None)[0])
            mean_i *= self.scale_array[i]
            mean_i += self.shift_array[i]
            mean.append(mean_i)
        return np.concatenate(mean)

class GpSimulateMessage(object):
    """For simulating model fields from the prior

    For simulating model fields from the prior for a given time point, designed
        to be temporary and used in distributed memory.

    Attributes:
        prior_mean: vector, Gaussian process mean, for each model field and
            point in space, length n_total_parameter
        model_field_shift: array, normalisation value for each model field
        model_field_scale: array, normalisation value for each model field
        area_unmask: number of points on the find grid
        n_model_field: number of model fields
        n_total_parameter: number of model fields x area_unmask
        cov_chol: broadcast, cholesky of kernel matrix (not covariance matrix),
            dimensions area_unmask x area_unmask
        precision: broadcast, used to scale cov_chol
        rng: random number generator
    """

    def __init__(self, model_field_target, cov_chol, precision, rng):
        """
        Args:
            model_field_target: TargetModelField for a time point
            cov_chol: broadcast of cov_chol
            precision: broadcast of gp_target.state["precision"]
            rng: random number generator
        """
        self.prior_mean = model_field_target.prior_mean
        self.model_field_shift = model_field_target.model_field_shift
        self.model_field_scale = model_field_target.model_field_scale
        self.area_unmask = model_field_target.area_unmask
        self.n_model_field = model_field_target.n_model_field
        self.cov_chol = cov_chol
        self.precision = preicison
        self.rng = rng

    def simulate_from_prior(self):
        """Return simulation from the prior

        Return:
            two array, 0th element simulated model field, 1st element rng after
                use
        """
        #implemented
        precision = self.precision.value()
        cov_chol = self.cov_chol.value()
        state_rng = TargetModelField.simulate_from_prior_given_gp(
            self, precision, cov_chol, self.rng)
        return [model_field_vector, rng]

def get_precision_prior():
    return stats.gamma(a=4/3, scale=3)

def get_regulariser_precision_prior():
    return stats.gamma(a=2, scale=1)
