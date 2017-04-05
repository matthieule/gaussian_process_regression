## Gaussian Process for Regresssion

Implementation of Gaussian process for regression

The code is mainly based on three classes:
- `GaussianProcess`: the Gaussian process holding the observations and computing the regression
- `Covariance`: the covariance function used by the Gaussian process
- `LikelihoodOptimizer`: the likelihood optimizer of the covariance parameters. I took a simple black-box optimization approach which does not require the likelihood derivatives.

Example scripts are in the `script` folder

## 1D Example

In this 1D example, we want to regress the simple 1D function:

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")



## 2D Example

## References:
- Rasmussen, C.E., 2006. Gaussian processes for machine learning.
- Rasmussen, C.E., Bernardo, J.M., Bayarri, M.J., Berger, J.O., Dawid, A.P., Heckerman, D., Smith, A.F.M. and West, M., 2003. Gaussian processes to speed up hybrid Monte Carlo for expensive Bayesian integrals. In Bayesian Statistics 7 (pp. 651-659).
