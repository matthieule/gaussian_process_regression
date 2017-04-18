"""Simple 2D example"""
import matplotlib.pyplot as plt
import numpy as np

from gaussian_process.covariance import SquaredExponential
from gaussian_process.gp import GaussianProcess2d
from gaussian_process.optimizer import SECovLikelihoodOptimizer
from gaussian_process.util import get_logger

np.random.seed(0)


@np.vectorize
def density(x, y):
    """Density to estimate

    :param x: first parameter
    :param y: second parameter
    :return: value of the density at (x, y)
    """

    a = np.exp(-(1-(x**2/2+y**2/0.5))**2/0.5)

    return a


def get_observations(xmin, xmax, ymin, ymax, n):
    """Get a list of random observation

    :param xmin: minimum x value where to plot the density
    :param xmax: maximum x value where to plot the density
    :param ymin: minimum y value where to plot the density
    :param ymax: maximum y value where to plot the density
    :param n: number of observation
    :return: two lists of the same length. One list of the observation points,
     and one list of the evaluation of the density at the observation points
    """

    list_observations = []
    list_y = []
    for _ in range(n):
        i = np.random.uniform(low=xmin, high=xmax)
        j = np.random.uniform(low=ymin, high=ymax)
        list_observations.append((i, j))
        list_y.append(density(i, j))

    return list_observations, list_y


def plot_density(xmin, xmax, ymin, ymax, n):
    """Plot the density

    :param xmin: minimum x value where to plot the density
    :param xmax: maximum x value where to plot the density
    :param ymin: minimum y value where to plot the density
    :param ymax: maximum y value where to plot the density
    :param n: number of evaluation point on the x and y axis
    """

    x_vec = np.linspace(xmin, xmax, n)
    y_vec = np.linspace(ymin, ymax, n)
    coords = np.meshgrid(x_vec, y_vec, indexing='ij')

    density_value = density(coords[0], coords[1])

    plt.imshow(
        density_value, cmap='viridis',
        extent=[xmin, xmax, ymin, ymax], vmin=0.0, vmax=1.0
    )
    plt.title('Real Density')
    plt.savefig('figures/2d_density.png', dpi=150, bbox_inches='tight')


def main():
    """Main"""

    logger = get_logger()

    # Define the admissible domain
    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    # Plot the real density
    logger.info('Plot the density to estimate')
    plot_density(xmin, xmax, ymin, ymax, 50)

    # Gather a list of observation
    logger.info('Gather some observations')
    list_observations, list_y = get_observations(xmin, xmax, ymin, ymax, 80)

    # Optimize the parameters of the covariance function
    logger.info('Optimize the Gaussian process')
    opt = SECovLikelihoodOptimizer(
        SquaredExponential, list_observations,
        list_y, initial_guess=np.array([1.0, 1.0, 1.0]), noise=1e-3
    )
    res = opt.maximum_likelihood()
    logger.info(res)

    # Define a Gaussian process with the optimized parameters
    logger.info('Define the final Gaussian process')
    cov = SquaredExponential(res.x[0], res.x[1:])
    gp = GaussianProcess2d(
        covariance=cov, list_observations=list_observations,
        list_y=list_y, noise=1e-3
    )
    gp.covariance_matrix()

    # Plot the interpolation
    logger.info('Plot the interpolation')
    n = 50
    x_vec_bis = np.linspace(xmin, xmax, n)
    y_vec_bis = np.linspace(ymin, ymax, n)
    gp.plot(x_vec_bis.tolist(), y_vec_bis.tolist())
    plt.savefig('figures/2d_estimation.png', dpi=150, bbox_inches='tight')

if __name__ == '__main__':

    main()
