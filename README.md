## Gaussian Process for Regresssion

Implementation of Gaussian process for regression

The code is mainly based on three classes:
- `GaussianProcess`: the Gaussian process holding the observations and computing the regression
- `Covariance`: the covariance function used by the Gaussian process
- `LikelihoodOptimizer`: the likelihood optimizer of the covariance parameters. I took a simple black-box optimization approach which does not require the likelihood derivatives.

Example scripts are in the `script` folder

## Using the Code

Example usage can be found in `script/example_1d.py` and `script/example_2d.py`.
To extend the code with new covariance function, one would need to:
- Define a subclass of `Covariance` implementing the abstract methods
- Define a subclass of `LikelihoodOptimizer` implementing the method `_instantiate_covariance` which details how to instantiate the covariance from an array input parameter

Some notes:
- The list of observation `list_observations` is assumed to be a list of tuple corresponding to the position in the parameter space where the function is evaluated
- The list of evaluation `list_y` is assumed to be a list of scalar corresponding to the evaluation of the target function on the parameter in `list_observations`
- We assume that `list_observations` and `list_y` have the same order

## To Do

- [ ] Link to the tensorflow implementation of GP for "real" work
- Extend the bibliography


## 1D Example

In this 1D example, we want to regress the simple 1D function:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/1d_density.png" alt="alt text" width=500px>

We use a Gaussian process with the squared exponential covariance function:

`cov(x, y) = w0 exp(-1/2*(x-y)^2/w1^2)`

The parameter `w1` corresponds to the correlation between the data point: the larger it is, the larger the point are assumed correlated.
Here is an example of the Gaussian process interpolation using 10 random data points and `w1 = 0.2`:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/small_covariance.png" alt="alt text" width=500px>

The blue line is the mean of the interpolating Gaussian process, the black lines are samples from the Gaussian process, the white scattered points are the observations.
Here is the same example but with `w1 = 0.5`, where we can see the effect of increasing the assumed correlation between data points:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/large_covariance.png" alt="alt text" width=500px>

We can see that there is probably a "sweet spot" for which the covariance parameters are more reasonable. We compute those by maximum likelihood. Here is the resulting regression:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/1d_estimation.png" alt="alt text" width=500px>

The blue line is the mean of the interpolating Gaussian process, the green lines are the +/- two standard deviation.

All the pictures from this example can be reproduced by running

```python
python script/example_1d.py 
```

## 2D Example

In this 2D example, we want to regress the simple 2D function:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/2d_density.png" alt="alt text" width=500px>

Here is the result of the regression using a squared exponential covariance function, and 80 data points:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/2d_estimation.png" alt="alt text" width=1200px>

From left to right, and top to bottom: mean of the Gaussian process, derivative along the first axis of the mean, derivative along 
the second axis of the mean, standard deviation of the Gaussian process, derivative along the first axis of the standard deviation, derivative along 
the second axis of the standard deviation, empirical standard deviation using 50 samples of the 2D Gaussian process. The fact that the analytical
standard deviation and the empirical one look similar is a good sanity check to make sure the standard deviation and the samples are correctly computed.

We also computed the empirical derivative of the mean and standard deviation using `np.gradient` to check
the sanity of our results:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/2d_estimation_empirical.png" alt="alt text" width=1200px>

All the pictures from this example (except the last one ;)) can be reproduced by running

```python
python script/example_2d.py 
```

## References:
- [Rasmussen, C.E., 2006. Gaussian processes for machine learning.](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
- [Rasmussen, C.E., Bernardo, J.M., Bayarri, M.J., Berger, J.O., Dawid, A.P., Heckerman, D., Smith, A.F.M. and West, M., 2003. Gaussian processes to speed up hybrid Monte Carlo for expensive Bayesian integrals. In Bayesian Statistics 7 (pp. 651-659).](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/pdfs/pdf2080.pdf)
