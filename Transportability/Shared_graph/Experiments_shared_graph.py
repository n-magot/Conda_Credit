"""Experimental section of SHARED graph UAI2025

In this example we simulate data: Do, De, Do* with binary outcome Y
- Do and De based on population  Π:
                                        X -> Y
                                        Z2 -> Y
                                        Z1 -> Y
                                    Confounders:
                                        Z2 -> X
                                    ,and Z2〜Bernoulli(0.3), Z1〜Bernoulli(0.9)

Z1 has different distribution in Π and Π*. In Π*: Z2〜Bernoulli(0.3), Z1〜Bernoulli(0.7, 0.4, 0.2)

intercept_Y= -0.5  # Intercept for Y
b_X_Y= -2  # Effect of X on Y
b_Z2_Y = 2.5  # Effect of Z2 on Y
b_Z1_Y = 3  # Effect of Z1 on Y
# Observational data coefficients
intercept_T_obs = 0  # Intercept for T
b_Z2_T_obs = 2  # Strong effect of Z2 on T (confounder)

"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
from itertools import combinations
from sklearn.metrics import log_loss
import pandas as pd
import math
import sys

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def Generate_DE_source(sample_size, intercept_Y, b_T_Y, b_Z2_Y,b_Z1_Y, random_seed=None):
    e = np.random.normal(size=sample_size)
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate independent binary variables T and Z2
    Z1 = np.random.binomial(1, 0.9, sample_size)  # Z1 ~ Bernoulli(0.7)
    Z2 = np.random.binomial(1, 0.3, sample_size)  # Z2 ~ Bernoulli(0.7)
    T = np.random.binomial(1, 0.5, sample_size)  # T ~ Bernoulli(0.5)

    # Generate Y based on the logistic model
    logit_Y = intercept_Y + b_T_Y * T + b_Z2_Y * Z2 + + b_Z1_Y * Z1 + e
    pr_Y = np.clip(1 / (1 + np.exp(-logit_Y)), 1e-6, 1 - 1e-6)
    Y = np.random.binomial(1, pr_Y, sample_size)

    X_exp = np.column_stack((T, Z1, Z2))

    return X_exp, Y


def Generate_Do_target(sample_size, intercept_Y, b_T_Y, b_Z2_Y, b_Z1_Y, intercept_T, b_Z2_T, random_seed=None):
    e = np.random.normal(size=sample_size)

    if random_seed is not None:
        np.random.seed(random_seed)

    Z1 = np.random.binomial(1, 0.6, sample_size)  # Z1 ~ Bernoulli(0.7)
    Z2 = np.random.binomial(1, 0.8, sample_size)  # Z2 ~ Bernoulli(0.3)


    # Generate T based on Z2
    logit_T = intercept_T + b_Z2_T * Z2
    prob_T = np.clip(1 / (1 + np.exp(-logit_T)), 1e-6, 1 - 1e-6)
    # print(prob_T)
    T = np.random.binomial(1, prob_T)

    # Generate Y based on the logistic model
    logit_Y = intercept_Y + b_T_Y * T + b_Z2_Y * Z2 + + b_Z1_Y * Z1 + e
    pr_Y = np.clip(1 / (1 + np.exp(-logit_Y)), 1e-6, 1 - 1e-6)
    Y = np.random.binomial(1, pr_Y, sample_size)

    X_obs = np.column_stack((T, Z1, Z2))

    return X_obs, Y

def Generate_DE_target(sample_size, intercept_Y, b_T_Y, b_Z2_Y,b_Z1_Y, random_seed=None):
    e = np.random.normal(size=sample_size)
    if random_seed is not None:
        np.random.seed(random_seed)

    Z1 = np.random.binomial(1, 0.6, sample_size)  # Z1 ~ Bernoulli(0.7)
    Z2 = np.random.binomial(1, 0.8, sample_size)  # Z2 ~ Bernoulli(0.3)
    T = np.random.binomial(1, 0.5, sample_size)  # T ~ Bernoulli(0.5)

    # Generate Y based on the logistic model
    logit_Y = intercept_Y + b_T_Y * T + b_Z2_Y * Z2 + + b_Z1_Y * Z1 + e
    pr_Y = np.clip(1 / (1 + np.exp(-logit_Y)), 1e-6, 1 - 1e-6)
    Y = np.random.binomial(1, pr_Y, sample_size)

    X_exp = np.column_stack((T, Z1, Z2))

    return X_exp, Y

def binary_logistic_regression(data, labels):
    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 10))
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)))
    logits = alpha + jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


def log_likelihood_calculation(alpha, beta, data, obs):
    logits = alpha + jnp.dot(data, beta)
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(obs)
    return log_likelihood.sum()


def sample_prior(data, num_samples):
    prior_samples = {}
    D = data.shape[1]

    prior_samples1 = dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
    prior_samples2 = dist.Cauchy(jnp.zeros(1), 10 * jnp.ones(1)).sample(random.PRNGKey(0), (num_samples,))

    prior_samples["beta"] = prior_samples1
    prior_samples["alpha"] = prior_samples2

    return prior_samples


def sample_posterior(data, observed_data, num_samples):
    D = data.shape[1]

    kernel = NUTS(binary_logistic_regression)
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
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["alpha"][i], samples["beta"][i],
                                                                               data, observed_data))
    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    # print('marginal', log_marginal_likelihood)

    return log_marginal_likelihood

Log_loss_alg = []
Log_loss_exp = []
Log_loss_obs = []
Log_loss_gt = []

No = 10000
Ne = 100
N_test = 1000

n_runs = 20  # Number of different datasets that you want to test the algorithm
columns = []
df = pd.DataFrame(columns=columns)

for k in range(n_runs):

    P_HAT_alg = {}
    P_HAT_exp = {}
    P_HAT_obs = {}
    P_HAT_gt = {}

    # Experimental data coefficients
    intercept_Y= -0.5# Intercept for Y
    b_T_Y= -2  # Effect of T on Y
    b_Z2_Y = 2.5  # Effect of Z2 on Y
    b_Z1_Y = 3  # Effect of Z1 on Y

    rs = np.random.seed(None)

    # Generate experimental data
    data_exp, labels_exp = Generate_DE_source(
        sample_size=Ne,
        intercept_Y=intercept_Y,
        b_T_Y=b_T_Y,
        b_Z1_Y=b_Z1_Y,
        b_Z2_Y=b_Z2_Y,
        random_seed= rs
    )
    # Generate experimental data
    data_test, labels_test = Generate_DE_target(
        sample_size=N_test,
        intercept_Y=intercept_Y,
        b_T_Y=b_T_Y,
        b_Z1_Y=b_Z1_Y,
        b_Z2_Y=b_Z2_Y,
        random_seed=rs
    )

    # Observational data coefficients
    intercept_T_obs = 0  # Intercept for T
    b_Z2_T_obs = 2  # Strong effect of Z2 on T (confounder)

    # Generate observational data
    data_obs, labels_obs = Generate_Do_target(
        sample_size=No,
        intercept_Y=intercept_Y,
        b_T_Y=b_T_Y,
        b_Z1_Y=b_Z1_Y,
        b_Z2_Y=b_Z2_Y,
        intercept_T=intercept_T_obs,
        b_Z2_T=b_Z2_T_obs,
        random_seed=rs
    )

    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier()
    model.fit(data_obs, labels_obs)
    # display the relative importance of each attribute
    print(model.feature_importances_)
    model.fit(data_exp, labels_exp)
    # display the relative importance of each attribute
    print(model.feature_importances_)
    print('how many 1 in observational', np.count_nonzero(labels_obs == 1))
    print('how many 1 in experimental', np.count_nonzero(labels_exp == 1))

    """Let's assume that Z1 is a latent and that MB(Y)=(T,Z2)"""
    """Let's assume that C is a latent confounder and that MB(Y)=(T,Z2)"""
    MB = (0, 2)
    sample_list = list(MB)
    P_Hzc = {}
    P_Hz_not_c = {}
    P_HZ = {}
    P_HZC = {}

    correct_IMB = 0
    num_samples = 10000

    """Searching for subsets of MB that contain the treatment T"""
    subset_list = [MB]

    for set in subset_list:
        reg_variables = set
        sub_data = data_obs[:, reg_variables]
        exp_sub_data = data_exp[:, reg_variables]

        posterior_samples = sample_posterior(sub_data, labels_obs, num_samples)

        prior_samples = sample_prior(exp_sub_data, num_samples)

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
            CMB = reg_variables

        test_sub_data = data_test[:, reg_variables]

        # Calclulate P_hat(Y) without multiply them with P(HZ|De,Do)
        """1. Assuming Hzc we can use both observational and experimental data"""
        rng_key = random.PRNGKey(np.random.randint(100))
        rng_key, rng_key_ = random.split(rng_key)
        combined_X = np.concatenate((sub_data, exp_sub_data), axis=0)  # Concatenate row-wise
        combined_y = np.concatenate((labels_obs, labels_exp), axis=0)  # Concatenate row-wise
        kernel = NUTS(binary_logistic_regression)
        num_samples = 1000
        mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
        mcmc.run(
            rng_key_, combined_X, combined_y
        )

        trace = mcmc.get_samples()
        intercept_alg = trace['alpha'].mean()
        slope_list = []
        for i in range(len(trace['beta'][0, :])):
            slope_list.append(trace['beta'][:, i].mean())

        slope_alg = np.array(slope_list)
        print('intercept and slope from observational + experimental:', intercept_alg, slope_alg)
        logit_alg = intercept_alg + np.dot(test_sub_data, slope_alg)
        pr_1_alg = 1 / (1 + np.exp(-logit_alg))
        P_hat_alg = pr_1_alg
        P_HAT_alg[reg_variables] = P_hat_alg

        """1. Assuming Hzc_ we can use only experimental data"""
        rng_key = random.PRNGKey(np.random.randint(100))
        rng_key, rng_key_ = random.split(rng_key)
        kernel = NUTS(binary_logistic_regression)
        num_samples = 1000
        mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
        mcmc.run(
            rng_key_, exp_sub_data, labels_exp
        )

        trace = mcmc.get_samples()
        intercept_exp = trace['alpha'].mean()
        slope_list = []
        for i in range(len(trace['beta'][0, :])):
            slope_list.append(trace['beta'][:, i].mean())

        slope_exp = np.array(slope_list)
        print('intercept and slope from experimental:', intercept_exp, slope_exp)
        logit_exp = intercept_exp + np.dot(test_sub_data, slope_exp)
        pr_1_exp = 1 / (1 + np.exp(-logit_exp))
        pr_0_exp = 1 - pr_1_exp
        P_hat_exp = pr_1_exp
        P_HAT_exp[reg_variables] = P_hat_exp

        """1. Assuming Hzc we can use only observational data"""
        rng_key = random.PRNGKey(np.random.randint(100))
        rng_key, rng_key_ = random.split(rng_key)
        kernel = NUTS(binary_logistic_regression)
        num_samples = 1000
        mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
        mcmc.run(
            rng_key_, sub_data, labels_obs
        )

        trace = mcmc.get_samples()
        intercept_obs = trace['alpha'].mean()
        slope_list = []
        for i in range(len(trace['beta'][0, :])):
            slope_list.append(trace['beta'][:, i].mean())

        slope_obs = np.array(slope_list)
        print('intercept and slope from observational:', intercept_obs, slope_obs)
        logit_obs = intercept_obs + np.dot(test_sub_data, slope_obs)
        pr_1_obs = 1 / (1 + np.exp(-logit_obs))
        pr_0_obs = 1 - pr_1_obs
        # how many 1s in test dataset
        P_1_prior = np.count_nonzero(labels_test == 1) / len(labels_test)
        P_hat_obs = pr_1_obs
        P_HAT_obs[reg_variables] = P_hat_obs

    print('P(De|Do,HZ)', P_Hzc)
    print('P(De|Do,HZc)', P_Hz_not_c)

    # Calculate equation 1
    for set in subset_list:
        P_HZ[set] = math.exp(P_Hzc[set]) / (math.exp(P_Hzc[set]) + math.exp(P_Hz_not_c[set]))
        P_HZC[set] = 1 - P_HZ[set]
        print('PHz for set', set, P_HZ[set])
        print('PHzc for set', set, P_HZC[set])

    # Generate dictionary ith probabilities Hz and Hzc
    columns = []
    for tup in subset_list:
        columns.append(f'P_HZ{tup}')
        columns.append(f'P_HZc{tup}')

    # Create an empty DataFrame with the generated column names
    df = df.reindex(columns=columns)

    # Create the new_row list
    new_row = []
    for tpl in subset_list:
        new_row.append(P_HZ[tpl])  # Add value from P_HZ
        new_row.append(P_HZC[tpl])

    df.loc[len(df)] = new_row

    # Calculate the average post-intervention outcome
    P_Yzc = {}
    P_Yz_not_c = {}

    for key in P_HZC:
        if P_HZ[key]>P_HZC[key]:
            """Can use only De"""
            P_Yz_not_c[key] = P_HZC[key] * P_HAT_exp[key] # ayto einai to p^(y\do(x),Z)
            """Use both Do and De"""
            P_Yzc[key] = P_HZ[key] * P_HAT_alg[key]
        else:
            """Can use only Do* as Z is an adjustment set but not s-admissible"""
            P_Yz_not_c[key] = P_HAT_obs[key]  # ayto einai to P*(y\do(x),Z)
            P_Yzc[key] = 0 * P_HAT_alg[key]


    P_Y_alg = {}
    P_Y_exp = {}
    P_Y_obs = {}

    for key in P_HZ:
        P_Y_alg = P_Yzc[key] + P_Yz_not_c[key]
        Log_loss_alg.append(log_loss(labels_test, P_Y_alg))
        # print('With our algorithm P(Y|do(X), V) ', P_Y_alg)
        print('Log loss function for our algorithm', log_loss(labels_test, P_Y_alg))

        P_Y_exp = P_HAT_exp[key]
        # print('With experimental data P(Y|do(X), V)', P_Y_exp)
        Log_loss_exp.append(log_loss(labels_test, P_Y_exp))
        print('Log loss function for experimental data', log_loss(labels_test, P_Y_exp))

        P_Y_obs = P_HAT_obs[key]
        # print('With observational data P(Y|do(X), V)', P_Y_obs)
        Log_loss_obs.append(log_loss(labels_test, P_Y_obs))
        print('Log loss function for observational data', log_loss(labels_test, P_Y_obs))

print('Log loss from our algorithm', Log_loss_alg)
print('Log loss in experimental', Log_loss_exp)
print('Log loss in observational', Log_loss_obs)

print(df)
