"""Gaussian process"""
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from covariance import Covariance


FUZZ = 1e-7


class UnsupportedDerivativeOrder(Exception):
    """Raised when the derivative order is not supported"""
    pass


class GaussianProcess:
    """Gaussian process"""

    def __init__(self, covariance: Covariance, noise=1e-7,
                 list_observations=None, list_y=None):
        """Init

        :param covariance: instance of a covariance class
        :param noise: noise
        :param list_observations: list of observation
        :param list_y: list of evaluation of the function to interpolate
        """

        self.covariance = covariance
        self.list_observations = list_observations if list_observations else []
        self.list_y = list_y if list_y else []
        self.n_observation = len(self.list_observations)
        self.cov_matrix = np.zeros((0, 0))
        self.noise = noise

    @staticmethod
    def _center_data(list_y):
        """Center the list

        :param list_y: input list
        :return: tuple with the centered list and the empirical mean
        """

        assert isinstance(list_y, list)

        mean = np.mean(list_y)
        centered_list_y = list_y - mean

        return centered_list_y, mean

    def _compute_covariance_matrix(self, list_obs_1, list_obs_2):
        """Compute the covariance matrix between two lists of observations

        :param list_obs_1: first list of observation
        :param list_obs_2: second list of observation
        :return: covariance matrix between the elements of list_obs_1 and
         list_obs_2
        """

        assert isinstance(list_obs_1, list)
        assert isinstance(list_obs_2, list)

        cov_matrix = np.zeros((len(list_obs_1), len(list_obs_2)))
        cov_matrix_flat = [
            (i, j, self.covariance(xi, yj))
            for (i, xi) in enumerate(list_obs_1)
            for (j, yj) in enumerate(list_obs_2)
            ]
        for coord_value in cov_matrix_flat:
            cov_matrix[coord_value[:2]] = coord_value[2]

        return cov_matrix

    def _compute_covariance_matrix_pd(self, list_obs_1, list_obs_2, pd_dim,
                                      order=1):

        assert isinstance(list_obs_1, list)
        assert isinstance(list_obs_2, list)

        if order == 1:
            fction = self.covariance.compute_pd
        elif order == 2:
            fction = partial(self.covariance.compute_pdpd, j=pd_dim)
        else:
            raise UnsupportedDerivativeOrder

        cov_matrix = np.zeros((len(list_obs_1), len(list_obs_2)))
        cov_matrix_flat = [
            (i, j, fction(xi, yj, pd_dim))
            for (i, xi) in enumerate(list_obs_1)
            for (j, yj) in enumerate(list_obs_2)
            ]
        for coord_value in cov_matrix_flat:
            cov_matrix[coord_value[:2]] = coord_value[2]

        return cov_matrix

    def _gp_up_to_date(self):
        """Assert the Gaussian process is up to date"""

        n = len(self.list_observations)

        assert n == len(self.list_y)
        assert self.cov_matrix.shape[0] == n

    def _order_observations(self):
        """Order the observation

        Can be useful to stabilize the covariance matrix
        """

        list_observations_y = zip(self.list_observations, self.list_y)
        list_observations_y = sorted(
            list_observations_y,
            key=lambda obs_y: np.linalg.norm(np.array(obs_y[0]))
        )
        self.list_observations = [obs for obs, y in list_observations_y]
        self.list_y = [y for obs, y in list_observations_y]

    def add_observation(self, x, y):
        """Add an observation

        :param x: parameter where the function to interpolate is evaluated
        :param y: value of the interpolated function
        """

        self.list_observations.append(x)
        self.list_y.append(y)
        self.n_observation += 1

    def covariance_matrix(self):
        """Compute the covariance matrix of the Gaussian process

        :return: covariance matrix between the observation of the Gaussian
         process
        """

        self._order_observations()
        self.cov_matrix = self._compute_covariance_matrix(
            self.list_observations, self.list_observations)

        self.cov_matrix += np.diag(np.array([self.noise] * self.n_observation))

        return self.cov_matrix

    def likelihood(self):
        """Likelihood of the Gaussian process

        :return: log likelihood of the Gaussian process
        """

        # assert the Gaussian process is up to date
        self._gp_up_to_date()

        noise_penalization_term = -1 / 2 * np.log(
            np.linalg.det(self.cov_matrix))

        y = np.linalg.solve(self.cov_matrix, self.list_y)
        y = np.array(self.list_y) @ y
        data_fidelity_term = -1 / 2 * y

        nbr_obs_term = - self.n_observation * np.log(2 * np.pi)
        likelihood = (
            noise_penalization_term + data_fidelity_term + nbr_obs_term
        )
        return likelihood

    def mean(self, x, derivative=False, i=None):
        """Compute the conditional mean of the Gaussian process

        Knowing the observations, compute the value of the mean of the Gaussian
        process at x

        :param x: parameter where to evaluate the mean of the Gaussian process
        :param derivative: boolean, whether or not to compute the derivative
         of sigma
        :param i: dimension along which to compute the derivative if True
        :return: interpolated value at x which is the mean of the Gaussian
         process (i.e. mean of the posterior probability of the Gaussian
         process knowing the observations)
        """

        assert isinstance(x, list)
        assert len(x) > 0
        assert isinstance(x[0], tuple)

        if derivative:
            assert 0 <= i < len(x[0])
            cov_function = partial(
                self._compute_covariance_matrix_pd, pd_dim=i
            )
        else:
            cov_function = self._compute_covariance_matrix

        # assert the Gaussian process is up to date
        self._gp_up_to_date()

        # Compute the correlation between the parameter x and the observation
        current_cov = cov_function(x, self.list_observations)
        # Solve the linear system
        centered_list_y, mean = self._center_data(self.list_y)
        y = np.linalg.solve(self.cov_matrix, centered_list_y)
        # Assert the resolution of the linear system went well
        assert np.allclose(np.array(centered_list_y), self.cov_matrix @ y)

        return mean + current_cov @ y

    def sample(self, x):
        """Sample a Gaussian process on x

        :param x: parameter where to sample the Gaussian process
        :return: a sample from the Gaussian process
        """

        assert isinstance(x, list)
        assert len(x) > 0
        assert isinstance(x[0], tuple)

        mean = self.mean(x)
        sigma = self.sigma(x)

        d, u = np.linalg.eig(sigma)
        assert np.allclose(u@np.diag(d)@u.T, sigma)

        d = np.real(d)
        d[d < FUZZ] = FUZZ
        d_sqrt = np.sqrt(d)

        sample = np.random.normal(loc=0, scale=1, size=len(x))
        sample = mean + u@np.diag(d_sqrt)@sample

        return sample

    def sigma(self, x, derivative=False, i=None):
        """Compute the conditional variance of the Gaussian process

        Knowing the observations, compute the value of the variance of the
        Gaussian process at x

        :param x: parameter where to evaluate the mean of the Gaussian process
        :param derivative: boolean, whether or not to compute the derivative
         of sigma
        :param i: dimension along which to compute the derivative if True
        :return: variance of the Gaussian process at x (i.e. mean of the
         posterior probability of the Gaussian process knowing the
         observations)
        """

        assert isinstance(x, list)
        assert len(x) > 0
        assert isinstance(x[0], tuple)
        if derivative:
            assert 0 <= i < len(x[0])
            auto_cov_function = partial(
                self._compute_covariance_matrix_pd, pd_dim=i, order=2
            )
            cov_function = partial(
                self._compute_covariance_matrix_pd, pd_dim=i
            )
        else:
            auto_cov_function = self._compute_covariance_matrix
            cov_function = self._compute_covariance_matrix

        # assert the Gaussian process is up to date
        self._gp_up_to_date()

        current_sigma = auto_cov_function(x, x)
        # Compute the correlation between the parameter x and the observation
        current_cov = cov_function(
            x, self.list_observations
        )
        # Solve the linear system
        y = np.linalg.solve(self.cov_matrix, current_cov.T)
        # Assert the resolution of the linear system went well
        assert np.allclose(current_cov.T, self.cov_matrix @ y)

        return current_sigma - current_cov @ y


class GaussianProcess1d(GaussianProcess):
    """1D Gaussian process"""

    def _estimate_gp(self, list_x):
        """Estimate the mean and variance of the Gaussian process

        :param list_x: points where to do the estimation
        :return: tuple with the mean and variance estimated at the points
         in list_x
        """

        assert isinstance(list_x, list)

        mean = self.mean(list_x)
        sigma = np.squeeze(
            np.array([self.sigma([x]) for x in list_x])
        )

        return mean, sigma

    def plot(self, list_x, ymin, ymax, n_samples=3, confidence_band=True):
        """Plotting utility

        :param list_x: list of point where to evaluate the interpolation
        :param ymin: minimum y axis value for the plot
        :param ymax: maximum y axis value for the plot
        :param n_samples: number of samples from the Gaussian process to plot
        :param confidence_band: boolean, whether or not to plot the confidence
         band
        """

        assert isinstance(list_x, list)

        mean, sigma = self._estimate_gp(list_x)

        for _ in range(n_samples):
            plt.plot(list_x, self.sample(list_x), color='black', linewidth='1')
        if confidence_band:
            plt.plot(list_x, mean + 2*np.sqrt(sigma), color='g',
                     linewidth='2')
            plt.plot(list_x, mean - 2*np.sqrt(sigma), color='g',
                     linewidth='2')
        plt.plot(list_x, mean, color='b', linewidth='3')
        plt.scatter(
            self.list_observations, self.list_y, s=50, facecolors='white',
            zorder=3
        )
        plt.axis([list_x[0][0], list_x[-1][0], ymin, ymax])


class GaussianProcess2d(GaussianProcess):
    """2D Gaussian process"""

    def _estimate_gp(self, list_x, list_y):
        """Estimate the Gaussian process mean and variance

        :param list_x: list of point on the first dimension where to evaluate
         the interpolation
        :param list_y: list of point on the second dimension where to evaluate
         the interpolation
        :return: two arrays of size (len(list_x), len(list_y)) with the
         estimation of the mean and variance of the Gaussian process
        """

        assert isinstance(list_x, list)
        assert isinstance(list_y, list)
        assert len(list_x) == len(list_y)

        n = len(list_x)

        mean = np.zeros((n, n))
        variance = np.zeros((n, n))

        mean_sigma_flat = [
            (i, j, self.mean([(xi, yj)])[0], self.sigma([(xi, yj)])[0])
            for (i, xi) in enumerate(list_x)
            for (j, yj) in enumerate(list_y)
            ]
        for coord_value in mean_sigma_flat:
            mean[coord_value[:2]] = coord_value[2]
            variance[coord_value[:2]] = coord_value[3]

        return mean, variance

    def _plot_data(self, fig, ax, *, img, x, y, xmin, xmax, ymin,
                   ymax, title, vmin=None, vmax=None):
        """Plotting utility"""

        im = ax.imshow(
            img, cmap='viridis', extent=[xmin, xmax, ymin, ymax],
            interpolation='none', vmin=vmin, vmax=vmax
        )
        fig.colorbar(im, ax=ax)
        ax.scatter(y, x, c=self.list_y, cmap='viridis', s=50)
        ax.set_title(title)

    def plot(self, list_x, list_y):
        """Plotting utility

        :param list_x: list of point on the first dimension where to evaluate
         the interpolation
        :param list_y: list of point on the second dimension where to evaluate
         the interpolation
        """

        assert isinstance(list_x, list)
        assert isinstance(list_y, list)
        assert len(list_x) == len(list_y)

        estimation, sigma = self._estimate_gp(list_x, list_y)
        x = [x[0] for x in self.list_observations]
        y = [x[1] for x in self.list_observations]

        fig, ax = plt.subplots(2, 1)

        self._plot_data(
            fig, ax[0],
            img=estimation, x=x, y=y,
            xmin=list_x[0], xmax=list_x[-1],
            ymin=list_y[0], ymax=list_y[-1],
            vmin=0.0, vmax=1.0, title='Estimation'
        )
        self._plot_data(
            fig, ax[1],
            img=np.sqrt(sigma), x=x, y=y,
            xmin=list_x[0], xmax=list_x[-1],
            ymin=list_y[0], ymax=list_y[-1], title='Standard Deviation'
        )
