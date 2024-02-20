import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random, vmap
import numpy
from itertools import combinations
import operator
import pandas as pd
import time

"""
1. Binary outcome
2. Binary independent variables except for continuous Age
3. Process without standardizing the data
For finding the correct MB we may need more than Ne=1000 in the beginning"""

st = time.time()
def Generate_Experimental_Data(sample_size):

     #beta_Y > T*Y + A*Y + Z1*T
     beta_Y = jnp.array([2.5, 1.9, 2.7])
     e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

     Z1 = dist.Bernoulli(probs=0.6).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
     A = 0 + 10 * random.normal(random.PRNGKey((numpy.random.randint(100))), (sample_size, 1))
     Z3 = dist.Bernoulli(probs=0.7).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
     Z4 = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
     Z5 = dist.Bernoulli(probs=0.8).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
     Z6 = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))

     T = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
     data_T_A = jnp.concatenate((T, A, Z1), axis=1)
     logits_Y = jnp.sum(beta_Y * data_T_A + e, axis=-1)
     Y = dist.Bernoulli(logits=logits_Y).sample(random.PRNGKey((numpy.random.randint(100))))

     labels = Y
     # data pane X,Z1,Z2
     data = jnp.concatenate((T, A, Z1, Z3, Z4, Z5), axis=1)

     return data, labels

Ne = 3000
data_exp, labels_exp = Generate_Experimental_Data(Ne)
#Edw ekana allages kai exw valei binary to T kai sta Experimental
T = data_exp[:, 0]
A = data_exp[:, 1]

print("Experimental data", data_exp)

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def model(data, labels):

     D = data.shape[1]
     alpha = numpyro.sample("alpha", dist.Cauchy(0, 2.5))
     beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 10 * jnp.ones(D)))
     logits = alpha + jnp.dot(data, beta)
     return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

# Run NUTS.
kernel = NUTS(model)
num_samples = 1000
mcmc = MCMC(kernel, num_warmup=2000, num_chains=1, num_samples=num_samples)
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

T_new = []
for i in range(len(A)):
    if A[i] < 0:
        T_new.append(0)
    else:
        T_new.append(1)
T_new = np.array(T_new)
# z = np.dot(A, 5.2)
# def logistic(z):
#      return 1 / (1 + np.exp(-z))
#
# prob = logistic(z)
# print(prob)

#Add T_new in the first column
obs_data = np.concatenate((T_new[:, np.newaxis], data_exp[:, 1:]), axis=1)
print("Observational Data", obs_data)

print("mean intercept", intercept)
print("mean slope", slope)
logits = jnp.sum(slope * obs_data + intercept, axis=-1)
obs_labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey((numpy.random.randint(100))))

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
     mcmc = MCMC(kernel, num_warmup=2000, num_samples=num_samples)
     mcmc.run(jax.random.PRNGKey(1244), data, observed_data)


     # Get the posterior samples
     posterior_samples = mcmc.get_samples()

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

     data, observed_data = obs_data, obs_labels

     # Import your necessary dependencies
     from sklearn.feature_selection import RFE
     from sklearn.linear_model import LogisticRegression

     # Feature extraction
     model = LogisticRegression()
     rfe = RFE(model, n_features_to_select=1, step=1)
     fit = rfe.fit(data, observed_data)
     print("Num Features: %s" % (fit.n_features_))
     print("Selected Features: %s" % (fit.support_))
     print("Feature Ranking: %s" % (fit.ranking_))

     from sklearn.feature_selection import SequentialFeatureSelector
     from sklearn.linear_model import RidgeCV

     feature_names = np.array(["T", "A", "Z1", "Z3", "Z4", "Z5"])
     ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(data, observed_data)

     sfs_forward = SequentialFeatureSelector(
         ridge, n_features_to_select=3, direction="forward"
     ).fit(data, observed_data)

     sfs_backward = SequentialFeatureSelector(
         ridge, n_features_to_select=3, direction="backward"
     ).fit(data, observed_data)

     print(
         "Features selected by forward sequential selection: "
         f"{feature_names[sfs_forward.get_support()]}"
     )
     print(
         "Features selected by backward sequential selection: "
         f"{feature_names[sfs_backward.get_support()]}"
     )

     from sklearn.feature_selection import SequentialFeatureSelector
     from sklearn.neighbors import KNeighborsClassifier

     knn = KNeighborsClassifier(n_neighbors=3)
     sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
     sfs.fit(data, observed_data)
     SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                               n_features_to_select=3)
     print('SFS with knn', sfs.get_support())

     import statsmodels.api as sm

     # Create DataFrame
     df = sm.add_constant(
          pd.DataFrame({'T': data[:, 0], 'A': data[:, 1], 'Z1': data[:, 2], 'Z3': data[:, 3], 'Z4': data[:, 4], 'Z5': data[:, 5] }))
     df['y'] = observed_data

     # Fit logistic regression model
     model = sm.GLM(df['y'], df[['T', 'A', 'Z1', 'Z3', 'Z4', 'Z5']], family=sm.families.Binomial())
     result = model.fit()
     print('GLM ', result.summary())

     num_samples = 1000
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

     exp_data, exp_observed_data = data_exp[1:300, :], labels_exp[1:300]

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

#9/10 me Ne = 300

