import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
from sklearn.metrics import log_loss
import pandas as pd
import math

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def linear_regression(data, targets=None):
    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 10))  # Intercept
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)))  # Coefficients
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5))  # Noise std dev

    mu = alpha + jnp.dot(data, beta)  # Linear predictor

    return numpyro.sample("obs", dist.Normal(mu, sigma), obs=targets)


def log_likelihood_linear(alpha, beta, sigma, data, obs):
    mu = alpha + jnp.dot(data, beta)
    log_likelihood = dist.Normal(mu, sigma).log_prob(obs)
    return log_likelihood.sum()


def sample_prior_linear(data, num_samples):
    prior_samples = {}
    D = data.shape[1]

    prior_samples["beta"] = dist.Cauchy(jnp.zeros(D), 10 * jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
    prior_samples["alpha"] = dist.Cauchy(0, 2.5).sample(random.PRNGKey(0), (num_samples,))
    prior_samples["sigma"] = dist.HalfCauchy(5).sample(random.PRNGKey(0), (num_samples,))

    return prior_samples


def sample_posterior(data, observed_data, num_samples):
    D = data.shape[1]

    kernel = NUTS(linear_regression)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(42), data, observed_data)

    # Get the posterior samples
    posterior_samples = mcmc.get_samples()
    # import arviz as az
    # import matplotlib.pyplot as plt
    # data_plot = az.from_numpyro(mcmc)
    # az.plot_trace(data_plot, compact=True, figsize=(15, 25))
    # plt.show()

    return posterior_samples


def calculate_log_marginal(num_samples, samples, data, observed_data):
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_linear(samples["alpha"][i], samples["beta"][i],
                                                                          samples["sigma"][i], data, observed_data))
    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    # print('marginal', log_marginal_likelihood)

    return log_marginal_likelihood


path = '/home/nandia/PycharmProjects/PythonProject/kernel-test-bias/src/data_dok.csv'

def data_preds(path):
    df = pd.read_csv(path)
    return df

df = data_preds(path)
# Observational data where s == 1
Do = df[df['s'] == 1].copy()
# List of columns from x1 to x13
x_cols = [f'x{i}' for i in range(1, 14)]

# New DataFrame with 't' first, then x1 to x13
data_obs = Do[['t'] + x_cols].copy()
labels_obs = Do['y']

# Experimental data where s == 0
De = df[df['s'] == 0].copy()
# New DataFrame with 't' first, then x1 to x13
data_exp = De[['t'] + x_cols].copy()
labels_exp = De['y']

data_obs = np.array(data_obs)
labels_obs = np.array(labels_obs)

data_exp = np.array(data_exp)
labels_exp = np.array(labels_exp)



"""Let's assume that Z1 is a latent and that MB(Y)=(T,Z2)"""
P_Hzc = {}
P_Hz_not_c = {}
P_HZ = {}
P_HZC = {}

correct_IMB = 0
num_samples = 1000

"""Searching in these subsets of MB"""
subset_list = [tuple(range(14))]

for set in subset_list:
    reg_variables = set
    sub_data = data_obs[:, reg_variables]
    exp_sub_data = data_exp[:, reg_variables]

    posterior_samples = sample_posterior(sub_data, labels_obs, num_samples)

    prior_samples = sample_prior_linear(exp_sub_data, num_samples)

    marginal = calculate_log_marginal(num_samples, prior_samples, exp_sub_data, labels_exp)
    # print('Marginal {} from experimental sampling:'.format(reg_variables), marginal)
    """P(De|Do, Hzc_)"""
    P_Hz_not_c[reg_variables] = marginal

    marginal = calculate_log_marginal(num_samples, posterior_samples, exp_sub_data, labels_exp)
    # print('Marginal {} from observational sampling:'.format(reg_variables), marginal)
    """P(De|Do, Hzc)"""
    P_Hzc[reg_variables] = marginal

    if P_Hzc[reg_variables] > P_Hz_not_c[reg_variables]:
        print("CMB", reg_variables)
        print('P_HZ', P_Hzc[reg_variables])
        print('P_HZc', P_Hz_not_c[reg_variables])

        CMB = reg_variables

    #logg sum exp trick instead of:  P_HZ[set] = math.exp(P_Hzc[set]) / (math.exp(P_Hzc[set]) + math.exp(P_Hz_not_c[set]))
    diff = P_Hzc[set] - P_Hz_not_c[set]
    from scipy.special import expit

    P_HZ[set] = expit(diff)

    P_HZC[set] = 1 - P_HZ[set]
    print('PHz for set', set, P_HZ[set])
    print('PHzc for set', set, P_HZC[set])
