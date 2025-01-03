"""In this example we simulate data Y(binary),T(binary),Z2(continuous) without latent confounding C(continuous)"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import pandas as pd
import math
from collections import Counter
import itertools
import time

# Start the timer
start_time = time.time()

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def Generate_experimetnal_data(sample_size):
    """Adjust the beta coefficient to change the relations between variables:
        Remember that always the treatment, T will be in the first column"""

    [b1, b3, b5] = [1.5, 0.4, 0.4]

    np.random.seed()
    # rng_key = random.PRNGKey(42)
    e = np.random.normal(size=sample_size)

    Z2 = np.random.normal(0, 10, sample_size)
    T = np.random.binomial(1, 0.5, sample_size)
    C = np.random.normal(0, 10, sample_size)
    Z3 = np.random.normal(0, 10, sample_size)
    Z4 = np.random.normal(0, 10, sample_size)
    Z5 = np.random.normal(0, 10, sample_size)

    logit_Y = 1 + b1*T + b3*Z2 + b5*C + 0.6*Z5 + e

    pr = 1 / (1 + np.exp(-logit_Y))  # pass through an inv-logit function
    print('prob of experimental', pr[0:10])
    y = np.random.binomial(n=1, p=pr, size=sample_size)  # bernoulli response variable
    X = np.column_stack((T, Z2, C, Z3, Z4, Z5))

    return X, y

def Generate_observational_data(data_exp, labels_exp):
    """Coefficient b4 will be adjusted to change the confounding relation between T,Y and C"""

    b4 = 0.4
    [b1, b2, b3, b4, b5] = [1.5, 0.4, 0.3, b4, 0.4]

    Z2 = data_exp[:, 1]
    Z5 = data_exp[:, 5]

    """Find ground truth model: P(Y)=f(T,Z,C), C:confounding covariate"""
    np.random.seed()
    rng_key = random.PRNGKey(np.random.randint(100))
    rng_key, rng_key_ = random.split(rng_key)

    def model(data, labels):
        D = data.shape[1]
        alpha = numpyro.sample("alpha", dist.Cauchy(0, 10))
        beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)))
        logits = alpha + jnp.dot(data, beta)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    # Run NUTS.
    kernel = NUTS(model)
    num_samples = 10000
    mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
    mcmc.run(
        rng_key_, data_exp, labels_exp
    )
    mcmc.print_summary()

    trace = mcmc.get_samples()
    intercept = trace['alpha'].mean()
    slope_list = []
    for i in range(len(trace['beta'][0, :])):
        slope_list.append(trace['beta'][:, i].mean())

    slope = np.array(slope_list)
    print('Ground truth model', intercept, slope)
    """End of ground truth model"""

    """Generate observational data, T is dependent on Z2 and C: P(T)=f(Z2, C)"""
    sample_size = data_exp.shape[0]
    e = np.random.normal(size=sample_size)

    logit_T = b2*Z2 + 0.5*Z5 + e
    prob = 1 / (1 + np.exp(-logit_T))
    print('prob of T_new', prob[0:10])

    T_new = np.random.binomial(1, prob.flatten())
    # print('how many 1 in T_new', np.count_nonzero(T_new == 1))

    # Add T_new in the first column
    data_obs = np.concatenate((T_new[:, np.newaxis], data_exp[:, 1:]), axis=1)

    #The fake outcome Y'=f(T_new,Z2,C)
    logit_Y = intercept + np.dot(data_obs, slope)
    pr = 1 / (1 + np.exp(-logit_Y))
    # print('prob of observational', pr[0:10])
    labels_obs = np.random.binomial(n=1, p=pr, size=sample_size)

    return data_obs, labels_obs

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

     prior_samples1 = dist.Cauchy(jnp.zeros(D), 2.5*jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
     prior_samples2 = dist.Cauchy(jnp.zeros(1), 10*jnp.ones(1)).sample(random.PRNGKey(0), (num_samples,))

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

def COMB(elements):
    # Ensure that zero is included in the combinations
    if 0 not in elements:
        raise ValueError("0 must be in the elements")

    # Remove zero from the list to find combinations of the rest
    elements_without_zero = [elem for elem in elements if elem != 0]

    # Generate combinations of the remaining elements of length p-1
    combinations_without_zero = itertools.combinations(elements_without_zero, len(elements) - 2)

    # Add zero to each combination
    combinations_with_zero = [(0,) + combo for combo in combinations_without_zero]
    # Include the full set as a combination at the first position
    full_set_combination = tuple(elements)  # Convert to tuple to maintain consistency with combinations
    combinations_with_zero.insert(0, full_set_combination)  # Insert the full set at the beginning

    return combinations_with_zero

def likelihood_ratio(full_ll, reduced_ll):
    LR = (reduced_ll - full_ll)
    print('LR', LR)
    BF = np.exp(LR)
    print('BF', BF)

    return LR, BF

def find_FUL_SET_after_loop(tuple_list):
    # Count the occurrences of each tuple
    tuple_counts = Counter(tuple_list)

    # Case 1: Find the tuple that appears exactly 2 times
    case_1_result = [tup for tup, count in tuple_counts.items() if count == 2 and tup != (0,)]
    if case_1_result:
        return case_1_result[0]  # Return the tuple that appears 2 times and is not (0,)

    # Case 2: Find the tuple that appears exactly once (unique)
    case_2_result = [tup for tup, count in tuple_counts.items() if count == 1]
    if case_2_result:
        return case_2_result[0]  # Return the tuple with a unique occurrence

    # Case 3: If all elements are the same, return that tuple (e.g., [(0,), (0,), (0,), (0,)])
    if len(tuple_counts) == 1:
        return tuple_list[0]  # All are the same, return the first one

    return None  # If no cases match, return None


Log_loss_alg = []
Log_loss_exp = []
Log_loss_obs = []


ne = 100  #Adjust the number of De
N_D_test = 2 #Number of different datasets that you want to test the algorithm
columns = []
df = pd.DataFrame(columns=columns)

for k in range(N_D_test):
    P_HAT_alg = {}
    P_HAT_exp = {}
    P_HAT_obs = {}
    No = 1000
    Ne = ne
    data_exp, labels_exp = Generate_experimetnal_data(3000)
    data_test, labels_test = data_exp[No + 1:No + 1001, :], labels_exp[No + 1:No + 1001]
    data_obs, labels_obs = Generate_observational_data(data_exp[0:No, :], labels_exp[0:No])
    # print('how many 1 in observational', np.count_nonzero(labels_obs == 1))
    """Continuous columns you want to standardize"""
    num_std_columns = 6
    data_obs[:, -num_std_columns:] = (data_obs[:, -num_std_columns:] - data_obs[:, -num_std_columns:].mean(axis=0)) / data_obs[:, -num_std_columns:].std(axis=0)
    data_test[:, -num_std_columns:] = (data_test[:, -num_std_columns:] - data_test[:, -num_std_columns:].mean(axis=0)) / data_test[:, -num_std_columns:].std(axis=0)
    data_exp[:, -num_std_columns:] = (data_exp[:, -num_std_columns:] - data_exp[:, -num_std_columns:].mean(axis=0)) / data_exp[:, -num_std_columns:].std(axis=0)

    """Let's assume that there is no latent confounder and that MB(Y)=(T,Z2,C)"""
    MB = (0, 1, 2, 3, 4)

    dict_HZ_c = {}
    dict_HZ_not_c = {}
    Scores_HZ_c = {}
    Scores_HZ_not_c = {}
    P_HZ = {}
    P_HZC = {}
    layer_combinations_list = []

    correct_IMB = 0
    num_samples = 1000

    data_exp, labels_exp = data_exp[0:Ne, :], labels_exp[0:Ne]
    print('how many 1 in the final experimental', np.count_nonzero(labels_exp == 1))


    FS = MB
    Layers = len(FS)-1
    list_to_invest = [FS]
    for layer in range(Layers):
        for element in list_to_invest:
            layer_combinations_list += COMB(element)
            layer_combinations_list = list(dict.fromkeys(layer_combinations_list))
        print('layer_combinations_list', layer_combinations_list)

        elements_in_dict = list(dict_HZ_c.keys())
        layer_combinations_list_remains = [item for item in layer_combinations_list if item not in elements_in_dict]
        print('layer_combinations_list_remains',layer_combinations_list_remains)
        for tup in layer_combinations_list_remains:

            reg_variables = tup
            sub_data = data_obs[:, reg_variables]
            exp_sub_data = data_exp[:, reg_variables]

            """P(De|Do, Hzc_)"""
            prior_samples = sample_prior(exp_sub_data, num_samples)
            marginal = calculate_log_marginal(num_samples, prior_samples, exp_sub_data, labels_exp)
            dict_HZ_not_c[reg_variables] = marginal

            """P(De|Do, Hzc)"""
            posterior_samples = sample_posterior(sub_data, labels_obs, num_samples)
            marginal = calculate_log_marginal(num_samples, posterior_samples, exp_sub_data, labels_exp)
            dict_HZ_c[reg_variables] = marginal

        print('dict_HZ_c', dict_HZ_c)
        total_HZ_c = jax.scipy.special.logsumexp(np.array(list(dict_HZ_c.values())))
        total_HZ_not_c = jax.scipy.special.logsumexp(np.array(list(dict_HZ_not_c.values())))
        print('total_HZ_c',total_HZ_c)

        for tup in layer_combinations_list_remains:
            reg_variables = tup
            Scores_HZ_c[reg_variables] = math.exp(dict_HZ_c[reg_variables] - total_HZ_c)
            Scores_HZ_not_c[reg_variables] = math.exp(dict_HZ_not_c[reg_variables] - total_HZ_not_c)


        print('Scores_HZ_c', Scores_HZ_c)
        print('Scores_HZ_not_c',Scores_HZ_not_c)
        # print('dict_HZ_c', dict_HZ_c)
        # print('dict_HZ_not_c', dict_HZ_not_c)

        """If P(Hz_c>0.1) or P(Hz_not_c>0.1) krata to set"""
        remaining_sets = [key for key in Scores_HZ_c if Scores_HZ_c[key] > 0.1] + [key for key in Scores_HZ_not_c if
                                                                                Scores_HZ_not_c[key] > 0.1]

        remaining_sets = list(dict.fromkeys(remaining_sets))
        print('remaining sets', remaining_sets)
        sets_until_the_end = remaining_sets

        if remaining_sets != []:
            if remaining_sets != list_to_invest:
                remaining_sets = [item for item in remaining_sets if item not in list_to_invest]
                list_to_invest = remaining_sets
            else:
                break
        else:
            break

    print('sets in the end', sets_until_the_end)
    visited_elements = list(Scores_HZ_c.keys())
    print('visited elements', visited_elements)

    # Generate dictionary ith probabilities Hz and Hzc
    columns = []
    for tup in visited_elements:
        columns.append(f'P_HZ{tup}')
        columns.append(f'P_HZc{tup}')

    # Create an empty DataFrame with the generated column names
    # df = pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)

    # Calculate the log(p1+p2+p3) when you have log(p1),log(p2),log(p3)
    val_P_Hzc = list(dict_HZ_c.values())
    val_P_Hzc_ = list(dict_HZ_not_c.values())
    # Combine the values of p's into a single list
    combined_values = val_P_Hzc + val_P_Hzc_
    lns = [value for value in combined_values]
    sum_logs = jax.scipy.special.logsumexp(np.array(lns))

    # Calculate equation 1
    for i in range(len(visited_elements)):
        reg_variables = visited_elements[i]
        P_HZ[reg_variables] = math.exp(val_P_Hzc[i] - sum_logs)
        P_HZC[reg_variables] = math.exp(val_P_Hzc_[i] - sum_logs)
        print('PHz for set', reg_variables, P_HZ[reg_variables])
        print('PHzc for set', reg_variables, P_HZC[reg_variables])

    # Create the new_row list
    new_row = []

    for tpl in visited_elements:
        new_row.append(P_HZ[tpl])  # Add value from P_HZ
        new_row.append(P_HZC[tpl])

    df.loc[len(df)] = new_row

    for key, value in dict_HZ_c.items():
        reg_variables = key
        sub_data = data_obs[:, reg_variables]
        exp_sub_data = data_exp[:, reg_variables]

        test_sub_data = data_test[:, reg_variables]

        if P_HZC[key] > 0.001 or P_HZ[key] > 0.001 or key == MB:

            # Calclulate P_hat(Y) without multiply them with P(HZ|De,Do)
            """1. Assuming Hzc we can use both observational and experimental data"""
            combined_X = np.concatenate((sub_data, exp_sub_data), axis=0)  # Concatenate row-wise
            combined_y = np.concatenate((labels_obs, labels_exp), axis=0)  # Concatenate row-wise
            trace = sample_posterior(combined_X, combined_y, num_samples)
            intercept_alg = trace['alpha'].mean()
            slope_list = []
            for i in range(len(trace['beta'][0, :])):
                slope_list.append(trace['beta'][:, i].mean())

            slope_alg = np.array(slope_list)
            print('intercept and slope from observational + experimental:', intercept_alg, slope_alg)
            logt_alg = intercept_alg + np.dot(test_sub_data, slope_alg)
            P_hat_alg = 1 / (1 + np.exp(-logt_alg))
            P_HAT_alg[reg_variables] = P_hat_alg

            """1. Assuming Hzc_ we can use only experimental data"""
            trace = sample_posterior(exp_sub_data, labels_exp, num_samples)
            intercept_exp = trace['alpha'].mean()
            slope_list = []
            for i in range(len(trace['beta'][0, :])):
                slope_list.append(trace['beta'][:, i].mean())

            slope_exp = np.array(slope_list)
            print('intercept and slope from experimental:', intercept_exp, slope_exp)
            logt_exp = intercept_exp + np.dot(test_sub_data, slope_exp)
            P_hat_exp = 1 / (1 + np.exp(-logt_exp))
            P_HAT_exp[reg_variables] = P_hat_exp

            """1. Assuming Hzc we can use only observational data"""
            trace = sample_posterior(sub_data, labels_obs, num_samples)
            intercept_obs = trace['alpha'].mean()
            slope_list = []
            for i in range(len(trace['beta'][0, :])):
                slope_list.append(trace['beta'][:, i].mean())

            slope_obs = np.array(slope_list)
            print('intercept and slope from observational:', intercept_obs, slope_obs)
            logt_obs = intercept_obs + np.dot(test_sub_data, slope_obs)
            P_hat_obs = 1 / (1 + np.exp(-logt_obs))
            P_HAT_obs[reg_variables] = P_hat_obs

        else:
            P_HAT_alg[reg_variables] = np.zeros(num_samples)
            P_HAT_obs[reg_variables] = np.zeros(num_samples)
            P_HAT_exp[reg_variables] = np.zeros(num_samples)

    # Calculate the average post-intervention outcome
    P_Yzc = {}
    P_Yz_not_c = {}

    for key in P_HZC:
        """Can use only De"""
        P_Yz_not_c[key] = P_HZC[key] * P_HAT_exp[key]

    for key in P_HZ:
        """Use both Do and De"""
        P_Yzc[key] = P_HZ[key] * P_HAT_alg[key]

    from sklearn.metrics import log_loss

    P_Y_alg = sum(P_Yzc.values()) + sum(P_Yz_not_c.values())
    Log_loss_alg.append(log_loss(labels_test, P_Y_alg))
    # print('Log loss function for our algorithm', log_loss(labels_test, P_Y_alg))

    P_Y_exp = P_HAT_exp[MB]
    Log_loss_exp.append(log_loss(labels_test, P_Y_exp))
    # print('Log loss function for experimental data', log_loss(labels_test, P_Y_exp))

    P_Y_obs = P_HAT_obs[MB]
    Log_loss_obs.append(log_loss(labels_test, P_Y_obs))
    # print('Log loss function for observational data', log_loss(labels_test, P_Y_obs))

    print('Log loss from our algorithm', Log_loss_alg)
    print('Log loss in experimental', Log_loss_exp)
    print('Log loss in observational', Log_loss_obs)

"""df is the dataframe that contains for each subset of MB the PHz and PHzc. For every subset, by adding PHz and
PHzc you can find the probability of a set Z to be an IMB:"""
print(df)


# End the timer
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds") 
