"""Likelihood optimizer"""
from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize

from covariance import Covariance
from gaussian_process import GaussianProcess


class LikelihoodOptimizer(object, metaclass=ABCMeta):
    """Likelihood optimizer"""

    def __init__(self, covariance_class, list_observations, list_y,
                 initial_guess=None, noise=1e-7):
        """Init

        :param covariance_class: class of the covariance
        :param list_observations: list of observations
        :param list_y: list of evaluation of the function to interpolate
        :param initial_guess: initial guess for the optimization
        :param noise: noise
        """

        self.covariance_class = covariance_class
        self.list_observations = list_observations
        self.list_y = list_y
        self.initial_guess = initial_guess
        self.noise = noise

    def _get_current_likelihood(self, covariance_param):
        """Compute likelihood

        :param covariance_param: parameters used to instantiate the covariance
        :return: likelihood of the current Gaussian process
        """

        gp = self.instanciate_gp(covariance_param)
        gp.covariance_matrix()
        return gp.likelihood()

    @abstractmethod
    def _instantiate_covariance(self, covariance_param):
        """Instantiate the covariance class with covariance_param"""

        raise NotImplementedError

    def instanciate_gp(self, covariance_param):
        """Instantiate the Gaussian process

        :param covariance_param: parameters used to instantiate the covariance
        :return: Gaussian process instance
        """

        cov = self._instantiate_covariance(covariance_param)
        assert isinstance(cov, Covariance)

        gp = GaussianProcess(
            cov, list_observations=self.list_observations,
            list_y=self.list_y, noise=self.noise
        )

        return gp

    def maximum_likelihood(self):
        """Maximize the likelihood

        :return: results of the optimization
        """

        # Define the function which is optimized: we minimize the negative log
        # likelihood
        def likelihood_optimization_func(param):
            return -self._get_current_likelihood(param)

        res = minimize(likelihood_optimization_func, self.initial_guess,
                       method='COBYLA', options={'disp': True})
        self.initial_guess = res.x
        return res


class SECovLikelihoodOptimizer(LikelihoodOptimizer):
    """Likelihood optimizer for the squared exponential covariance"""

    def _instantiate_covariance(self, covariance_param):
        """Instantiate the covariance

        :param covariance_param: parameters used to instantiate the covariance
        :return: Covariance instance
        """

        return self.covariance_class(covariance_param[0], covariance_param[1:])
