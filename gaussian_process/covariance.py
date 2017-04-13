"""Covariance"""
from abc import ABCMeta, abstractmethod

import numpy as np


class Covariance(object, metaclass=ABCMeta):
    """Base class for the covariance function"""

    def __call__(self, x, y):
        return self.compute(x, y)

    @abstractmethod
    def compute(self, x, y):
        """Evaluate the covariance function"""

        raise NotImplementedError

    @abstractmethod
    def compute_pd(self, x, y, **kwargs):
        """Evaluate the partial derivative of the covariance function"""

        raise NotImplementedError

    @abstractmethod
    def compute_pdpd(self, x, y, **kwargs):
        """Second order partial derivative of the covariance function"""

        raise NotImplementedError


class SquaredExponential(Covariance):
    """Squared exponential covariance function"""

    def __init__(self, w0, w1):
        """Init

        :param w0: variance parameter
        :param w1: correlation parameter
        """

        assert isinstance(w1, np.ndarray)

        # We take the absolute value to make sure the parameters are positive
        # (this is a lay way to enforce constraint during optimization)
        self.w0 = abs(w0)
        self.w1 = abs(w1)

    def compute(self, x, y):
        """Evaluate the covariance function

        :param x: first parameter
        :param y: secon parameter
        :return: covariance between x and y
        """

        assert len(x) == len(y)
        assert len(x) == len(self.w1)

        x_array = np.array(x)
        y_array = np.array(y)
        euclidean_norm = np.linalg.norm(
            x_array / self.w1 - y_array / self.w1
        ) ** 2

        return self.w0 * np.exp(-euclidean_norm / 2)

    def compute_pd(self, x, y, i):
        """Evaluate the partial derivative of the covariance function

        The partial derivative is evaluated between x and y at x_i
        where x = [x_0, x_1, ...., x_n]

        :param x: first parameter
        :param y: second parameter
        :param i: dimension of x where the partial derivative is evaluated
        :return: partial derivative of the covariance between x and y
        """
        print(i)
        return self.compute(x, y) * (-(x[i] - y[i]) / (self.w1[i]**2))

    def compute_pdpd(self, x, y, i, j):
        """Second order partial derivative of the covariance function

        The partial derivative is evaluated between x and y at x_i and y_j
        where x = [x_0, x_1, ...., x_n], and y = [y_0, y_1, ...., y_n]

        :param x: first parameter
        :param y: second parameter
        :param i: dimension of x where the partial derivative is evaluated
        :param j: dimension of y where the partial derivative is evaluated
        :return: second order partial derivative between x and y
        """

        common_term = self.compute_pd(x, y, i=i)

        if i == j:
            first_term = self.compute(x, y) * y[i] / self.w1[i] ** 2
            result = first_term + common_term * (x[j] - y[j]) / self.w1[j] ** 2
        else:
            result = common_term * (x[j] - y[j]) / self.w1[j] ** 2

        return result
