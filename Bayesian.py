import numpy as np

lst = [1, 2, 3]
type(lst)

m = np.array(lst)
type(m)

# Multidimensional arrays
m = np.array([[1, 2, 3], [4, 5, 6]])
m
# find dimensions (rows, columns)

m.shape
# evenly spaced values in a ginven interval
m = np.arange(0, 24, 2)
m
m1 = m.reshape(4, 3) # reshape array to be 4x3
m1
m.resize(2, 6)
m
m = np.linspace(0, 2, 9)
m

# scipy
import scipy.stats as stats

# the general pattern to access functions related to probability distributions is
# $$ \text{scipy.stats.<distribution family>.<function>}$$

stats.beta.cdf(0.1, 2, 3) # evaluates the CDF of a beta(2, 3) random variable at 0.1.
stats.beta.cdf(0.2, 2, 3) # evaluates the CDF of a beta(2, 3) random variable at 0.1.

# generate 10 random sample from a normal (Gaussian) random variable with mean 2 and standard deviation 3

stats.norm.rvs(2, 3, size=100)

# Examples

import arviz

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import pymc3 as pm
import pystan

# import tensorflow as tf
plt.style.use('seaborn-darkgrid')

pm.__version__  # if not 3.5, run `%pip install pymc3==3.5`

##############################################
# prioir x likelihood = posterior
##############################################
success = 6
tosses = 9

# define grid
grid_points = 100

# define grid
p_grid = np.linspace(0, 1, grid_points)

# compute likelihood at each point in the grid
likelihood = stats.binom.pmf(success, tosses, p_grid)
plt.plot(stats.binom.pmf(success, tosses, p_grid))

prior = np.repeat(1, grid_points)

# compute product of likelihood and prior
unstd_posterior = likelihood * prior


def computePosterior(likelihood, prior):
    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior
    unstd_posterior
    # standardize posterior
    posterior = unstd_posterior/unstd_posterior.sum()

    plt.figure(figsize=(17, 3))
    ax1 = plt.subplot(131)
    ax1.set_title("Prior")
    plt.plot(p_grid, prior)

    ax2 = plt.subplot(132)
    ax2.set_title("Likelihood")
    plt.plot(p_grid, likelihood)

    ax3 = plt.subplot(133)
    ax3.set_title("Posterior")
    plt.plot(p_grid, posterior)
    plt.show()

    return posterior

prior1 = np.repeat(1, grid_points)
posterior1 = computePosterior(likelihood, prior1)

prior2 = 2* (p_grid >= 0.5).astype(int)
posterior2 = computePosterior(likelihood, prior2)

prior3 = np.exp(- 5 * abs(p_grid - 0.5))
posterior3 = computePosterior(likelihood, prior3)

sum(posterior1[(p_grid > 0.48) * (p_grid < 0.52)])

sum(posterior2[(p_grid > 0.48) * (p_grid < 0.52)])

sum(posterior3[(p_grid > 0.48) * (p_grid < 0.52)])

# The Monte Carlo method
# Find approximate value of $\pi.$

def in_circle(x, y, r):
    return math.sqrt(x**2 + y**2) <= r**2

in_circle(3,4,2.1)


def approx_pi(r, n):
    xs, ys, cols = [], [], []

    count = 0

    for i in range(n):
        x = np.random.uniform(0, r, 1)
        y = np.random.uniform(0, r, 1)
        xs.append(x)
        ys.append(y)

        if in_circle(x, y, r):
            count += 1
            cols.append("red")
        else:
            cols.append("steelblue")

    pi_appr = round(4 * count / n, 3)

    plt.figure(figsize=(2, 2))
    plt.scatter(xs, ys, c=cols, s=2)
    plt.title("pi (approximately) = " + str(pi_appr))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return pi_appr

# example
# estiamte the integral $\int_0^1 e^x dx$.

x = np.linspace(0, 1, 100)
plt.plot(x, np.exp(x))
pts = np.random.uniform(0, 1, (100, 2))
pts[:, 1] *= np.e
count = 0
cols = ['steelblue'] * 100
for i in range(100):
    if pts[i, 1] > np.exp(pts[i, 0]):  # acceptance / rejection step
        cols[i] = 'red'
        count += 1
plt.scatter(pts[:, 0], pts[:, 1], c=cols)
plt.xlim([0, 1])
plt.ylim([0, np.e]);
print(count)

# analytic solution
import sumpy
import mpmath
from sympy import *
from sympy import symbols, integrate, exp
x = symbols('x')
expr = integrate(exp(x), (x,0,1))
expr.evalf()
# numerical quadrature
from scipy import integrate
integrate.quad(exp, 0, 1)

# Monte Carlo approximation
for n in 10**np.array([1,2,3,4,5,6,7,8]):
    pts = np.random.uniform(0, 1, (n, 2))
    pts[:, 1] *= np.e
    count = np.sum(pts[:, 1] < np.exp(pts[:, 0]))
    volume = np.e * 1 # volume of region
    sol = (volume * count)/n
    print('%10d %.6f' % (n, sol))
    print(volume)
    print(count)

# The frequentist way - numerically
n = 4
h = 3
p = h/n
rv = stats.binom(n, p)
mu = rv.mean()
mu
p

# The Bayesian way
# Beta distribution:
# $$ \text{Beta}_\theta(a,b)  = C * \theta^{(a-1)} (1 - \theta)^{(b-1)} $$

# Exercise
# compute posterior distribution analytically
a, b = 10, 10                   # hyperparameters
prior = stats.beta(a, b)        # prior
post = stats.beta(h+a, n-h+b)   # posterior


def beta_binomial(n, h, a, b):
    # frequentist
    p = h / n
    rv = stats.binom(n, p)
    mu = rv.mean()

    # Bayesian
    prior = stats.beta(a, b)
    post = stats.beta(h + a, n - h + b)

    thetas = np.linspace(0, 1, 200)
    plt.figure(figsize=(8, 6))
    plt.plot(thetas, prior.pdf(thetas), label='Prior', c='blue')
    plt.plot(thetas, post.pdf(thetas), label='Posterior', c='red')
    plt.plot(thetas, n * stats.binom(n, thetas).pmf(h), label='Likelihood', c='green')
    plt.axvline((h + a - 1) / (n + a + b - 2), c='red', linestyle='dashed', alpha=0.4, label='MAP')
    plt.axvline(mu / n, c='green', linestyle='dashed', alpha=0.4, label='MLE')
    plt.xlim([0, 1])
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.ylabel('Density', fontsize=16)
    plt.legend();

beta_binomial(100, 80, 10, 10)

# Metropolis-Hastings random walk algorithm

# - Start with an initial guess for $\theta$
# - Chose a new proposed value as $\theta_p = \theta + \delta \theta, \delta \theta \sim N(0, \sigma).$
# Here we have chosen the proposal distribution to be $N(0, \sigma).$
# - If $g$ is the posterior probability, calculate the ratio $\rho = \frac{g(\theta_p \mid X)}{g(\theta \mid X)}$
# - (adjust for symmetry of the proposal distribution)
# - If $\rho \ge 1,$ accept $\theta = \theta_p;$ if $\rho < 1,$ accept  $\theta = \theta_p$ with probability $p,$ other wise keep $\theta = \theta.$ (This step is done with the help of the standard Uniform distribution)

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def target(likelihood, prior, n, h, theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        return likelihood(n, theta).pmf(h)*prior.pdf(theta)

n = 100
h = 61
a = 10
b = 10
likelihood = stats.binom
prior = stats.beta(a, b)
sigma = 0.3

naccept = 0
theta = 0.1
niters = 1000

samples = np.zeros(niters+1)
samples[0] = theta

for i in range(niters):
    theta_p = theta + stats.norm(0, sigma).rvs()
    rho = min(1, target(likelihood, prior, n, h, theta_p)/target(likelihood, prior, n, h, theta ))
    u = np.random.uniform()
    if u < rho:
        naccept += 1
        theta = theta_p
    samples[i+1] = theta

nmcmc = len(samples)//2
print("Portion of accepted steps = " + str(naccept/niters))

post = stats.beta(h+a, n-h+b)
thetas = np.linspace(0, 1, 200)

plt.figure(figsize=(8, 6))
rlt.hist(samples[nmcmc:], 20, histtype='step', density=True, linewidth=1, label='Distribution of posterior samples');
plt.hist(prior.rvs(nmcmc), 40, histtype='step', density=True, linewidth=1, label='Distribution of prior samples');
plt.plot(thetas, post.pdf(thetas), c='red', linestyle='--', alpha=0.5, label='True posterior')
plt.xlim([0,1]);
plt.legend(loc='best');


import numpy as np

lst = [1, 2, 3]
type(lst)

m = np.array(lst)
type(m)

# Multidimensional arrays
m = np.array([[1, 2, 3], [4, 5, 6]])
m
# find dimensions (rows, columns)

m.shape
# evenly spaced values in a ginven interval
m = np.arange(0, 24, 2)
m
m1 = m.reshape(4, 3) # reshape array to be 4x3
m1
m.resize(2, 6)
m
m = np.linspace(0, 2, 9)
m

# scipy
import scipy.stats as stats

# the general pattern to access functions related to probability distributions is
# $$ \text{scipy.stats.<distribution family>.<function>}$$

stats.beta.cdf(0.1, 2, 3) # evaluates the CDF of a beta(2, 3) random variable at 0.1.
stats.beta.cdf(0.2, 2, 3) # evaluates the CDF of a beta(2, 3) random variable at 0.1.

# generate 10 random sample from a normal (Gaussian) random variable with mean 2 and standard deviation 3

stats.norm.rvs(2, 3, size=100)

# Examples

import arviz

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import pymc3 as pm
import pystan

# import tensorflow as tf
plt.style.use('seaborn-darkgrid')

pm.__version__  # if not 3.5, run `%pip install pymc3==3.5`

##############################################
# prioir x likelihood = posterior
##############################################
success = 6
tosses = 9

# define grid
grid_points = 100

# define grid
p_grid = np.linspace(0, 1, grid_points)

# compute likelihood at each point in the grid
likelihood = stats.binom.pmf(success, tosses, p_grid)
plt.plot(stats.binom.pmf(success, tosses, p_grid))

prior = np.repeat(1, grid_points)

# compute product of likelihood and prior
unstd_posterior = likelihood * prior


def computePosterior(likelihood, prior):
    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior
    unstd_posterior
    # standardize posterior
    posterior = unstd_posterior/unstd_posterior.sum()

    plt.figure(figsize=(17, 3))
    ax1 = plt.subplot(131)
    ax1.set_title("Prior")
    plt.plot(p_grid, prior)

    ax2 = plt.subplot(132)
    ax2.set_title("Likelihood")
    plt.plot(p_grid, likelihood)

    ax3 = plt.subplot(133)
    ax3.set_title("Posterior")
    plt.plot(p_grid, posterior)
    plt.show()

    return posterior

prior1 = np.repeat(1, grid_points)
posterior1 = computePosterior(likelihood, prior1)

prior2 = 2* (p_grid >= 0.5).astype(int)
posterior2 = computePosterior(likelihood, prior2)

prior3 = np.exp(- 5 * abs(p_grid - 0.5))
posterior3 = computePosterior(likelihood, prior3)

sum(posterior1[(p_grid > 0.48) * (p_grid < 0.52)])

sum(posterior2[(p_grid > 0.48) * (p_grid < 0.52)])

sum(posterior3[(p_grid > 0.48) * (p_grid < 0.52)])

# The Monte Carlo method
# Find approximate value of $\pi.$

def in_circle(x, y, r):
    return math.sqrt(x**2 + y**2) <= r**2

in_circle(3,4,2.1)


def approx_pi(r, n):
    xs, ys, cols = [], [], []

    count = 0

    for i in range(n):
        x = np.random.uniform(0, r, 1)
        y = np.random.uniform(0, r, 1)
        xs.append(x)
        ys.append(y)

        if in_circle(x, y, r):
            count += 1
            cols.append("red")
        else:
            cols.append("steelblue")

    pi_appr = round(4 * count / n, 3)

    plt.figure(figsize=(2, 2))
    plt.scatter(xs, ys, c=cols, s=2)
    plt.title("pi (approximately) = " + str(pi_appr))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return pi_appr

# example
# estiamte the integral $\int_0^1 e^x dx$.

x = np.linspace(0, 1, 100)
plt.plot(x, np.exp(x))
pts = np.random.uniform(0, 1, (100, 2))
pts[:, 1] *= np.e
count = 0
cols = ['steelblue'] * 100
for i in range(100):
    if pts[i, 1] > np.exp(pts[i, 0]):  # acceptance / rejection step
        cols[i] = 'red'
        count += 1
plt.scatter(pts[:, 0], pts[:, 1], c=cols)
plt.xlim([0, 1])
plt.ylim([0, np.e]);
print(count)

# analytic solution
import sumpy
import mpmath
from sympy import *
from sympy import symbols, integrate, exp
x = symbols('x')
expr = integrate(exp(x), (x,0,1))
expr.evalf()
# numerical quadrature
from scipy import integrate
integrate.quad(exp, 0, 1)

# Monte Carlo approximation
for n in 10**np.array([1,2,3,4,5,6,7,8]):
    pts = np.random.uniform(0, 1, (n, 2))
    pts[:, 1] *= np.e
    count = np.sum(pts[:, 1] < np.exp(pts[:, 0]))
    volume = np.e * 1 # volume of region
    sol = (volume * count)/n
    print('%10d %.6f' % (n, sol))
    print(volume)
    print(count)

# The frequentist way - numerically
n = 4
h = 3
p = h/n
rv = stats.binom(n, p)
mu = rv.mean()
mu
p

# The Bayesian way
# Beta distribution:
# $$ \text{Beta}_\theta(a,b)  = C * \theta^{(a-1)} (1 - \theta)^{(b-1)} $$

# Exercise
# compute posterior distribution analytically
a, b = 10, 10                   # hyperparameters
prior = stats.beta(a, b)        # prior
post = stats.beta(h+a, n-h+b)   # posterior


def beta_binomial(n, h, a, b):
    # frequentist
    p = h / n
    rv = stats.binom(n, p)
    mu = rv.mean()

    # Bayesian
    prior = stats.beta(a, b)
    post = stats.beta(h + a, n - h + b)

    thetas = np.linspace(0, 1, 200)
    plt.figure(figsize=(8, 6))
    plt.plot(thetas, prior.pdf(thetas), label='Prior', c='blue')
    plt.plot(thetas, post.pdf(thetas), label='Posterior', c='red')
    plt.plot(thetas, n * stats.binom(n, thetas).pmf(h), label='Likelihood', c='green')
    plt.axvline((h + a - 1) / (n + a + b - 2), c='red', linestyle='dashed', alpha=0.4, label='MAP')
    plt.axvline(mu / n, c='green', linestyle='dashed', alpha=0.4, label='MLE')
    plt.xlim([0, 1])
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.ylabel('Density', fontsize=16)
    plt.legend();

beta_binomial(100, 80, 10, 10)

# Metropolis-Hastings random walk algorithm

# - Start with an initial guess for $\theta$
# - Chose a new proposed value as $\theta_p = \theta + \delta \theta, \delta \theta \sim N(0, \sigma).$
# Here we have chosen the proposal distribution to be $N(0, \sigma).$
# - If $g$ is the posterior probability, calculate the ratio $\rho = \frac{g(\theta_p \mid X)}{g(\theta \mid X)}$
# - (adjust for symmetry of the proposal distribution)
# - If $\rho \ge 1,$ accept $\theta = \theta_p;$ if $\rho < 1,$ accept  $\theta = \theta_p$ with probability $p,$ other wise keep $\theta = \theta.$ (This step is done with the help of the standard Uniform distribution)


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def target(likelihood, prior, n, h, theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        return likelihood(n, theta).pmf(h)*prior.pdf(theta)

n = 100
h = 61
a = 10
b = 10
likelihood = stats.binom
prior = stats.beta(a, b)
sigma = 0.3

naccept = 0
theta = 0.1
niters = 10000

samples = np.zeros(niters+1)
samples[0] = theta

for i in range(niters):
    theta_p = theta + stats.norm(0, sigma).rvs()
    rho = min(1, target(likelihood, prior, n, h, theta_p)/target(likelihood, prior, n, h, theta ))
    u = np.random.uniform()
    if u < rho:
        naccept += 1
        theta = theta_p
    samples[i+1] = theta

nmcmc = len(samples)//2
print("Portion of accepted steps = " + str(naccept/niters))

post = stats.beta(h+a, n-h+b)
thetas = np.linspace(0, 1, 20)

plt.figure(figsize=(8, 6))
plt.hist(samples[nmcmc:], 20, histtype='step', density=True, linewidth=1, label='Distribution of posterior samples');
plt.hist(prior.rvs(nmcmc), 40, histtype='step', density=True, linewidth=1, label='Distribution of prior samples');
plt.plot(thetas, post.pdf(thetas), c='red', linestyle='--', alpha=0.5, label='True posterior')
plt.xlim([0,1]);
plt.legend(loc='best');

# Convergence diagnostics# Convergence diagnostics
# Rigorous way of assesing convergence is an unsolved problems.
# But there are several tool we can use to convice ourselves that an MCMC
# has converged, such as
# - trace plots need to look stationary
# - parallel chain should carry similar information

def mh_coin(niters, n, h, theta, likelihood, prior, sigma):
    samples = [theta]
    while len(samples) < niters:
        theta_p = theta + stats.norm(0, sigma).rvs()
        rho = min(1, target(likelihood, prior, n, h, theta_p) / target(likelihood, prior, n, h, theta))
        u = np.random.uniform()
        if u < rho:
            theta = theta_p
        samples.append(theta)

    return samples

n = 100
h = 61

likelihood= stats.binom
prior = stats.beta(a, b)
sigma = 0.05
niters = 100

chains = [mh_coin(niters, n, h, theta, likelihood, prior, sigma) for theta in np.arange(0.1, 1, 0.2)]
chains
type(chains)
len(chains)

# Compare multiple chains
plt.figure(figsize=(8, 6))

for chain in chains:
    plt.plot(chain, '-o')

plt.xlim([0, niters])
plt.ylim([0, 1]);

# Probabilistic programming languages (PPLs)

# PPLs accessible via or native to Python include
#
# Python-native:
# - **PyMC3**
# - PyMC4
# - PyRo
# - **Edward2**
#
# API for Stan in Python:
# - **PyStan**

# We will consider PPLs available via Python in breadth and depth: first, we will implement the same simple model in several PPLs (breadth), and then will dive into one PPLs by considering more examples.
# PyMC3
# https://docs.pymc.io/
# PyMC3 syntaxis
import pymc3 as pm

# Model creation
with pm.Model() as model:
    # Model definition
    pass

# Unobserved random variables
with pm.Model():
    x = pm.Normal('x', mu=0, sd=1)

# Observed random variables

with pm.Model() as model:
    obs = pm.Normal('obs', mu=3, sd=1, observed=np.random.randn(100))
