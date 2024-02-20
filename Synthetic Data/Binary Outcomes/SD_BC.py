"""
1. Binary outcome
2. Both binary and continuous independent variables
3. Process with standardizing the data
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import numpy
from itertools import combinations
import operator
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import time

st = time.time()

def Generate_Experimental_Data(sample_size):

    #beta_Y > T*Y + A*Y + Z1*T
    beta_Y = jnp.array([2.5, 1.9, 1.7])
    e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    Z1 = 0 + 15 * random.normal(random.PRNGKey((numpy.random.randint(100))), (sample_size, 1))
    A = 0 + 10 * random.normal(random.PRNGKey((numpy.random.randint(100))), (sample_size, 1))
    Z3 = 0 + 7 * random.normal(random.PRNGKey((numpy.random.randint(100))), (sample_size, 1))
    Z4 = 0 + 3 * random.normal(random.PRNGKey((numpy.random.randint(100))), (sample_size, 1))
    Z5 = dist.Bernoulli(probs=0.8).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
    Z6 = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))

    T = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
    data_T_A = jnp.concatenate((T, A, Z1), axis=1)
    logits_Y = jnp.sum(beta_Y * data_T_A + e, axis=-1)
    Y = dist.Bernoulli(logits=logits_Y).sample(random.PRNGKey((numpy.random.randint(100))))

    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((T, A, Z1, Z3, Z4, Z5, Z6), axis=1)

    return data, labels

# 1. Generate Experimental Data

Ne = 1000
data_exp, labels_exp = Generate_Experimental_Data(Ne)

# 3. Standardize continuous variables
"""Standardize continuous variable before use your model for finding
ground truth coefficients from experimental data"""
data_continuous = data_exp[:, 1:5]
data_continuous = (data_continuous - data_continuous.mean()) * (0.5 / data_continuous.std()) #standardization

var_0 = data_exp[:, 0] - data_exp[:, 0].mean()
var_5 = data_exp[:, 5] - data_exp[:, 5].mean()
var_6 = data_exp[:, 6] - data_exp[:, 6].mean()
De_std = np.concatenate((np.array(var_0)[:, np.newaxis], data_continuous, var_5[:, np.newaxis],
                         var_6[:, np.newaxis]), axis=1)
print("Experimental Data after standardization", De_std)

# 2. Define the model
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def model(data, labels):

    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 2.5))
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 10 * jnp.ones(D)))
    logits = alpha + jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


#Run NUTS.
kernel = NUTS(model)
num_samples = 1000
mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
mcmc.run(
    rng_key_, De_std, labels_exp
)
mcmc.print_summary()

trace = mcmc.get_samples()
intercept = trace['alpha'].mean()
# slope = np.array([trace['beta'][:, 0].mean(), trace['beta'][:, 1].mean(), trace['beta'][:, 2].mean()])
"""Gia osoi einai oi coefficients tou slope pane kai ftiaxe mou ta mean tous se mia lista, anti na to kanw xeirokinita
opws to kanw panw se sxolio """
slope_list = []
for i in range(len(trace['beta'][0, :])):
    slope_list.append(trace['beta'][:, i].mean())

slope = np.array(slope_list)

# 2. Create Observational data
"""Pare oles tis stiles, dld oles tis metavlites ektos apo to T pou thelw na allaxo kai tha einai panta sthn 1h stili"""
data_except_T = De_std[:, 1:]

# data_numpy = np.array(De_std)
# T_new = []
# for i in range(len(data_numpy[:, 0])):
#     if data_numpy[i, 1] < 0.4:
#         T_new.append(0)
#     else:
#         T_new.append(1)

"""Edw ftiaxnw to T_new sumfona me logistic regression, kanonika oso peirazw to syntelesti tha prepei na allazei"""
A = data_exp[:, 1]
z = np.dot(A, 2.2)
def logistic(z):
    return 1 / (1 + np.exp(-z))

prob = logistic(z)
T_new = np.random.binomial(1, prob.flatten())
print(T_new)
T_new = T_new - T_new.mean()


#Add T_new in the first column
Do_std = np.concatenate((np.array(T_new)[:, np.newaxis], data_except_T), axis=1)
# obs_data = np.stack((T_new, A, Z1), axis=1)
print("Standardized Observational Data", Do_std)

print("mean intercept", intercept)
print("mean slope", slope)

#Based on ground truth model and the T_new create the Outcome in the observational data
e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (Ne, 1))
logits = jnp.sum(intercept + slope * Do_std + e, axis=-1)
labels_obs = dist.Bernoulli(logits=logits).sample(random.PRNGKey((numpy.random.randint(100))))



def binary_logistic_regression(data, labels):

    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 2.5))
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 10 * jnp.ones(D)))
    logits = alpha + jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


# Define a function to compute the log likelihood
def log_likelihood_calculation(alpha, beta, data, obs):
    logits = alpha + jnp.dot(data, beta)
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(obs)
    return log_likelihood.sum()

def sample_prior(data):

    prior_samples = {}
    D = data.shape[1]
    #paizei kai i paralagi me Cauchy(0, 2.5) gia tous coefs kai Cauchy(0, 10 ) gia to intercept
    prior_samples1 = dist.Cauchy(jnp.zeros(D), 2.5*jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
    prior_samples2 = dist.Cauchy(jnp.zeros(1), 10*jnp.ones(1)).sample(random.PRNGKey(0), (num_samples,))

    prior_samples["beta"] = prior_samples1
    prior_samples["alpha"] = prior_samples2

    return prior_samples

# Perform MCMC with NUTS to sample from the posterior
def sample_posterior(data, observed_data):

    D = data.shape[1]

    kernel = NUTS(binary_logistic_regression)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(1244), data, observed_data)


    # Get the posterior samples
    posterior_samples = mcmc.get_samples()

    data_plot = az.from_numpyro(mcmc)
    az.plot_trace(data_plot, compact=True, figsize=(15, 25))
    plt.show()

    return posterior_samples

# Calculate log likelihood for each posterior sample
def calculate_log_marginal(num_samples, samples, data, observed_data):
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["alpha"][i], samples["beta"][i],
                                                                               data, observed_data))

    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    return log_marginal_likelihood

def var_combinations(data):
    #how many variables we have
    num_variables = data.shape[1]
    column_list = list(map(lambda var: var, range(0, num_variables)))

    df = pd.DataFrame(data, columns=column_list)
    sample_list = df.columns.values[0:]
    list_comb = []
    for l in range(df.shape[1]):
        list_combinations = list(combinations(sample_list, df.shape[1]-l))
        for x in list_combinations:
            if x[0] == 0:
                list_comb.append(x)
    return list_comb

for i in range(1):

    data, observed_data = Do_std, labels_obs
    num_samples = 2000
    MB_Scores = {}
    IMB_Scores_obs = {}
    IMB_Scores_exp = {}

    list_comb = var_combinations(data)
    print(list_comb)

    for comb in range(len(list_comb)):
        reg_variables = list_comb[comb]

        sub_data = data[:, reg_variables]

        prior_samples = sample_prior(sub_data)

        marginal = calculate_log_marginal(num_samples, prior_samples, sub_data, observed_data)

        MB_Scores[reg_variables] = marginal

    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    print(MB_Scores)
    print(MB_Do)

    sample_list = list(MB_Do)

    """Searching for subsets of MB"""
    subset_list = []
    for s in range(len(sample_list)):
        list_combinations = list(combinations(sample_list, len(sample_list) - s))
        for x in list_combinations:
            if x[0] == 0:
                subset_list.append(x)
    print('The subsets of MB are {}'.format(subset_list))

    exp_data, exp_observed_data = De_std, labels_exp

    """For subsets of MB sample from experimental and observational data"""
    for j in range(len(subset_list)):
        reg_variables = subset_list[j]
        sub_data = data[:, reg_variables]
        exp_sub_data = exp_data[:, reg_variables]

        posterior_samples = sample_posterior(sub_data, observed_data)

        prior_samples = sample_prior(exp_sub_data)

        marginal = calculate_log_marginal(num_samples, prior_samples, exp_sub_data, exp_observed_data)
        # print('Marginal {} from experimental sampling:'.format(reg_variables), marginal)
        IMB_Scores_exp[reg_variables] = marginal

        marginal = calculate_log_marginal(num_samples, posterior_samples, exp_sub_data, exp_observed_data)
        # print('Marginal {} from observational sampling:'.format(reg_variables), marginal)

        IMB_Scores_obs[reg_variables] = marginal
        if IMB_Scores_obs[reg_variables] > IMB_Scores_exp[reg_variables]:
            print("CMB", reg_variables)

print(IMB_Scores_exp)
print(IMB_Scores_obs)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
