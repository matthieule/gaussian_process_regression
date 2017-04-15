"""Simple 1D example"""
import numpy as np
import matplotlib.pyplot as plt

from gaussian_process.covariance import SquaredExponential
from gaussian_process.gp import GaussianProcess1d
from gaussian_process.optimizer import SECovLikelihoodOptimizer
from gaussian_process.util import get_logger

np.random.seed(0)


@np.vectorize
def density(x):

    y = np.sin(x) + x/2

    return y


def example_interpolation(xmin, xmax, ymin, ymax,
                          w0, w1, title, n=10):

    list_observations, list_y = get_observations(xmin, xmax, n)

    # Define a Gaussian process with the optimized parameters
    cov = SquaredExponential(w0, np.array([w1]))
    gp = GaussianProcess1d(
        covariance=cov, list_observations=list_observations,
        list_y=list_y
    )
    gp.covariance_matrix()

    # Plot the interpolation
    x_vec = np.linspace(xmin, xmax, 50)
    x_vec = [(x,) for x in x_vec]
    gp.plot(list_x=x_vec, ymin=ymin, ymax=ymax, confidence_band=False)
    plt.savefig(title, dpi=150, bbox_inches='tight')
    plt.close()


def get_observations(xmin, xmax, n):
    """Get a list of random observation

    :param xmin: minimum x value where to plot the density
    :param xmax: maximum x value where to plot the density
    :param n: number of observation
    :return: two lists of the same length. One list of the observation points,
     and one list of the evaluation of the density at the observation points
    """

    list_observations = []
    list_y = []
    for _ in range(n):
        i = np.random.uniform(low=xmin, high=xmax)
        list_observations.append((i,))
        list_y.append(density(i))

    return list_observations, list_y


def plot_density(xmin, xmax, ymin, ymax, n):
    """Plot the density

    :param xmin: minimum x value where to plot the density
    :param xmax: maximum x value where to plot the density
    :param n: number of evaluation point on the x and y axis
    """

    x_vec = np.linspace(xmin, xmax, n)
    density_value = density(x_vec)

    plt.plot(x_vec, density_value, color='r', linewidth='3')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title('Real Density')
    plt.savefig('figures/1d_density.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main"""

    logger = get_logger()

    xmin = 0
    xmax = 2*np.pi
    ymin = 0
    ymax = 3.0

    np.random.seed(0)
    example_interpolation(
        xmin, xmax, ymin, ymax, 0.5, 0.3, 'figures/small_covariance'
    )
    np.random.seed(0)
    example_interpolation(
        xmin, xmax, ymin, ymax, 0.5, 0.5, 'figures/large_covariance'
    )

    plot_density(xmin, xmax, ymin, ymax, 20)
    np.random.seed(0)
    list_observations, list_y = get_observations(xmin, xmax, 10)

    # Optimize the parameters of the covariance function
    logger.info('Optimize the Gaussian process')
    opt = SECovLikelihoodOptimizer(
        SquaredExponential, list_observations,
        list_y, initial_guess=np.array([1.0, 1.0]), noise=1e-3
    )
    res = opt.maximum_likelihood()
    logger.info(res)

    # Define a Gaussian process with the optimized parameters
    logger.info('Define the final Gaussian process')
    cov = SquaredExponential(res.x[0], res.x[1:])
    gp = GaussianProcess1d(
        covariance=cov, list_observations=list_observations,
        list_y=list_y
    )
    gp.covariance_matrix()

    # Plot the interpolation
    logger.info('Plot the interpolation')
    x_vec = np.linspace(xmin, xmax, 50)
    x_vec = [(x,) for x in x_vec]
    gp.plot(list_x=x_vec, ymin=ymin, ymax=ymax, n_samples=0)
    plt.savefig('figures/1d_estimation.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    main()
