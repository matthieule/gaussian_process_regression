## Gaussian Process for Regresssion

Implementation of Gaussian process for regression

The code is mainly based on three classes:
- `GaussianProcess`: the Gaussian process holding the observations and computing the regression
- `Covariance`: the covariance function used by the Gaussian process
- `LikelihoodOptimizer`: the likelihood optimizer of the covariance parameters. I took a simple black-box optimization approach which does not require the likelihood derivatives.

Example scripts are in the `script` folder

## 1D Example

In this 1D example, we want to regress the simple 1D function:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/1d_density.png" alt="alt text" width=500px>

We use a Gaussian process with the squared exponential covariance function:

`cov(x, y) = w0 exp(-1/2*(x-y)^2/w1^2)`

The parameter `w1` corresponds to the correlation between the data point: the larger it is, the larger the point are assumed correlated.
Here is an example of the Gaussian process interpolation using 10 random data points and `w1 = 0.3`:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/small_covariance.png" alt="alt text" width=500px>

The blue line is the mean of the interpolating Gaussian process, the black lines are samples from the Gaussian process, the white scattered points are the observations.
Here is the same example but with `w1 = 0.5`, where we can see the effect of increasing the assumed correlation between data points:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/large_covariance.png" alt="alt text" width=500px>

We can see that there is probably a "sweet spot" for which the covariance parameters are more reasonable. We compute those by maximum likelihood. Here is the resulting regression:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/1d_estimation.png" alt="alt text" width=500px>

The blue line is the mean of the interpolating Gaussian process, the green lines are the +/- one variance.

All the pictures from this example can be reproduced by running

```python
python script/example_1d.py 
```

## 2D Example

In this 2D example, we want to regress the simple 2D function:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/2d_density.png" alt="alt text" width=500px>

Here is the result of the regression using a squared exponential covariance function, and 60 data points:

<img src="https://github.com/matthieule/gaussian_process_regression/blob/master/figures/2d_estimation.png" alt="alt text" width=600px>

All the pictures from this example can be reproduced by running

```python
python script/example_2d.py 
```

## References:
- [Rasmussen, C.E., 2006. Gaussian processes for machine learning.](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
- [Rasmussen, C.E., Bernardo, J.M., Bayarri, M.J., Berger, J.O., Dawid, A.P., Heckerman, D., Smith, A.F.M. and West, M., 2003. Gaussian processes to speed up hybrid Monte Carlo for expensive Bayesian integrals. In Bayesian Statistics 7 (pp. 651-659).](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/pdfs/pdf2080.pdf)
