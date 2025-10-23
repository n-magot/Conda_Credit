from scipy.special import gammaln
from sklearn.metrics import log_loss
from scipy.special import softmax
from scipy.special import logsumexp
import itertools
import numpy as np
import pandas as pd

#Standard seed for reproducibility
np.random.seed(42)

def simulate_multinomial_logit_exp(n_samples, coefs, intercepts):
    """
    Simulates data for a 3-class multinomial logistic regression model.

    Parameters:
        n_samples (int): Number of samples to simulate.
        coefs (np.ndarray): Coefficients for each class (shape: [3, num_features]).
        intercepts (np.ndarray): Intercepts for each class (shape: [3]).

    Returns:
        df: DataFrame with features X1, X2,... and label Y.
    """
    np.random.seed(42)

    # Simulate features: let's assume 3 binary features
    T = np.random.randint(0, 2, size=(n_samples, 1))
    Z = np.random.randint(0, 2, size=(n_samples, 1))
    C = np.random.randint(0, 2, size=(n_samples, 1))
    X = np.hstack((T, Z, C))

    # Compute logits (shape: [n_samples, 3])
    logits = X @ coefs.T + intercepts
    probs = softmax(logits, axis=1)

    # Sample Y from the categorical distribution
    Y = np.array([np.random.choice(3, p=probs[i]) for i in range(n_samples)])

    # Combine into a DataFrame
    df = pd.DataFrame(X, columns=['T', 'Z', 'C'])
    df['Y'] = Y

    return df

def simulate_multinomial_logit_obs(n_samples, coefs, intercepts):
    np.random.seed(42)

    # Simulate 2 binary features Z1, Z2
    Z = np.random.randint(0, 2, size=(n_samples, 1))
    C = np.random.randint(0, 2, size=(n_samples, 1))
    X = np.hstack((Z, C))

    logit_T = 3.8 * X[:, 1]
    prob_T = 1 / (1 + np.exp(-logit_T))
    T = np.random.binomial(1, prob_T.flatten())  # shape (n_samples,)

    # Combine T, Z1, Z2, and C into a feature matrix
    T_col = T.reshape(-1, 1)
    X_new = np.hstack((T_col, X))

    # Compute logits and probabilities
    logits = X_new @ coefs.T + intercepts
    probs = softmax(logits, axis=1)

    # Sample Y from the multinomial distribution
    Y = np.array([np.random.choice(3, p=probs[i]) for i in range(n_samples)])

    # Create DataFrame
    df = pd.DataFrame(X_new, columns=['T', 'Z', 'C'])  # X = T
    df['Y'] = Y

    return df

def log_loss_GT(X_test,y_test, coefs, intercepts):

    # Compute logits for all samples and classes
    logits = X_test @ coefs.T + intercepts  # shape (N, 3)

    # Compute predicted probabilities
    probs = softmax(logits, axis=1)  # shape (N, 3)

    # Calculate log loss using sklearn (expects probs and integer labels)
    loss = log_loss(y_test, probs)

    return loss

def get_counts_multiZ(df, Z_cols, Y_col):
    df = df.copy()

    # Force Z columns to categorical with categories [0, 1] (default assumption)
    for col in Z_cols:
        if df[col].dtype.name != 'category':
            df[col] = pd.Categorical(df[col], categories=[0, 1])

    # Y column categorical as usual
    df[Y_col] = df[Y_col].astype('category')
    Y_values = df[Y_col].cat.categories.tolist()

    # Cartesian product of Z categories (all possible 0/1 combos)
    Z_categories = [df[col].cat.categories.tolist() for col in Z_cols]
    Z_values = list(itertools.product(*Z_categories))

    index_map = {z: i for i, z in enumerate(Z_values)}
    y_map = {y: i for i, y in enumerate(Y_values)}

    N_j = np.zeros(len(Z_values), dtype=int)
    N_jk = np.zeros((len(Z_values), len(Y_values)), dtype=int)

    for _, row in df.iterrows():
        z_tuple = tuple(row[Z_cols])
        y_val = row[Y_col]
        if any(pd.isnull(v) for v in z_tuple) or pd.isnull(y_val):
            continue
        i = index_map[z_tuple]
        k = y_map[y_val]
        N_j[i] += 1
        N_jk[i, k] += 1

    df_counts = pd.DataFrame(N_jk, index=Z_values, columns=Y_values)
    df_counts.index.name = "Z_configuration"
    df_counts["Total"] = N_j

    return Z_values, Y_values, N_j, N_jk, df_counts


def dirichlet_bayesian_score(counts, priors=None):
    counts = np.asarray(counts)
    if priors is None:
        priors = np.zeros_like(counts)
    N = np.sum(counts, axis=-1)
    sum_priors = np.sum(priors + 1, axis=-1)
    score_in = (
            gammaln(sum_priors)
            - np.sum(gammaln(priors + 1), axis=-1)
            + np.sum(gammaln(counts + priors + 1), axis=-1)
            - gammaln(N + sum_priors)
    )
    if score_in.ndim > 0:
        return np.sum(score_in)
    else:
        return score_in


def P_De_given_HZc_log(N_o_jk, N_e_jk, priors):
    log_prob_marginal_joint = dirichlet_bayesian_score(N_o_jk + N_e_jk, priors)
    log_prob_marginal_obs = dirichlet_bayesian_score(N_o_jk, priors)
    log_prob_conditional_diff = log_prob_marginal_joint - log_prob_marginal_obs

    return log_prob_conditional_diff


def P_De_given_HZc_bar_log(N_e_jk, priors):
    return dirichlet_bayesian_score(N_e_jk, priors)


def run_subset_pipeline(Do, De, X_col, Z_cols, Y_col, priors_val=1):
    results = []

    # Always include X, add subsets of Z_cols (empty to full)
    for r in range(len(Z_cols) + 1):
        for subset in itertools.combinations(Z_cols, r):
            # Current variables: X plus subset of Z
            current_vars = [X_col] + list(subset)

            # Get counts for Do and De
            _, _, N_o_j, N_o_jk, df_counts_o = get_counts_multiZ(Do, current_vars, Y_col)
            _, _, N_e_j, N_e_jk, df_counts_e = get_counts_multiZ(De, current_vars, Y_col)

            # Priors: alpha_jk = 1 uniformly
            alpha_jk = np.ones_like(N_o_jk) * priors_val
            priors = alpha_jk - 1

            # Calculate probabilities
            p_hzc_log = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
            p_hzcbar_log = P_De_given_HZc_bar_log(N_e_jk, priors)

            # Store in dict
            # Store values (no exp yet!)
            results.append({
                'Variables': tuple(current_vars),
                'P(De|Do,HcZ) log': p_hzc_log,
                'P(De|Do,HcZ_bar) log': p_hzcbar_log,
            })

        # --- Compute total log-sum to normalize safely ---
        log_total = logsumexp([
            np.logaddexp(res['P(De|Do,HcZ) log'], res['P(De|Do,HcZ_bar) log'])
            for res in results
        ])

        # --- Normalize probabilities in log-space and store them ---
        for res in results:
            p_hzc_log = res['P(De|Do,HcZ) log']
            p_hzcbar_log = res['P(De|Do,HcZ_bar) log']

            # Normalized probabilities in real space using log-sum-exp trick
            res['P_HZ_c'] = np.exp(p_hzc_log - log_total)
            res['P_HZ_c_bar'] = np.exp(p_hzcbar_log - log_total)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    return df_results


def compute_posterior_predictive_both_hypotheses(N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j):
    """
    Returns:
    - probs_HZc: P(Y|do(X), Z, HZc) predictive probs under H_Z^c
    - probs_HZc_bar: P(Y|do(X), Z, HZc_bar) predictive probs under H̄_Z^c
    """
    numerator_HZc = N_o_jk + N_e_jk + alpha_jk
    denominator_HZc = (N_o_j + N_e_j + alpha_j)[:, np.newaxis]
    probs_Y_HZc = numerator_HZc / denominator_HZc

    numerator_HZc_bar = N_e_jk + alpha_jk
    denominator_HZc_bar = (N_e_j + alpha_j)[:, np.newaxis]
    probs_Y_HZc_bar = numerator_HZc_bar / denominator_HZc_bar

    numerator_obs = N_o_jk + alpha_jk
    denominator_obs = (N_o_j + alpha_j)[:, np.newaxis]
    probs_Y_obs = numerator_obs / denominator_obs

    return probs_Y_HZc, probs_Y_HZc_bar, probs_Y_obs

"""Inputs for simulate the data"""
coefs = np.array([
    [3.0, -3.0, 2.5],
    [4.5, 4.5, -3.0],
    [3.6, -1.5, 3.0]
])

intercepts = np.array([0.2, -0.1, 0.3])  # shape (3,)

coefs_GT = np.array([
    [3.0, -3.0, 2.5],
    [4.5, 4.5, -3.0],
    [3.6, -1.5, 3.0]
])

intercepts_GT = np.array([0.2, -0.1, 0.3])

No = 1000
Ne = 50
Ntest = 1000

Do = simulate_multinomial_logit_obs(n_samples=No, coefs=coefs, intercepts=intercepts)
De = simulate_multinomial_logit_exp(n_samples=Ne, coefs=coefs, intercepts=intercepts)
De_test = simulate_multinomial_logit_exp(n_samples=Ntest, coefs=coefs, intercepts=intercepts)

treatment = 'T'
outcome = 'Y'
covariate_with_T = [col for col in Do.columns if col != outcome]
covariates_without_T = [col for col in Do.columns if col not in [outcome, treatment]]

X_test = De_test[covariate_with_T]
y_test = De_test[outcome]

GT_log_loss = log_loss_GT(X_test, y_test, coefs_GT, intercepts_GT)


Z_cols = covariate_with_T
FS = Z_cols
Y_col = outcome

# Run it
df_scores = run_subset_pipeline(Do, De, treatment, covariates_without_T, outcome)

print(df_scores.round(4))

tuple_subsets = df_scores.iloc[:, 0].tolist()
subsets = [list(t) for t in tuple_subsets]

for Z_cols in subsets:
    # Get counts for each Z config
    Z_vals_Do, _, N_o_j, N_o_jk, df_counts_o = get_counts_multiZ(Do, Z_cols, outcome)
    Z_vals_De, _, N_e_j, N_e_jk, df_counts_e = get_counts_multiZ(De, Z_cols, outcome)

    # Priors
    alpha_jk = np.ones_like(N_o_jk)
    alpha_j = np.sum(alpha_jk, axis=1)

    # Predictive probs under both hypotheses
    probs_Y_HZc, probs_Y_HZc_bar, probs_Y_obs = compute_posterior_predictive_both_hypotheses(
        N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j
    )

    # Map Z configuration to index
    Z_config_to_index = {tuple(row): i for i, row in enumerate(Z_vals_De)}

    probs_rows_HZc = []
    probs_rows_HZc_bar = []
    probs_rows_obs = []

    for _, row in De_test.iterrows():
        z_tuple = tuple(row[col] for col in Z_cols)
        idx = Z_config_to_index.get(z_tuple)

        if idx is not None:
            probs1 = probs_Y_HZc[idx]
            probs2 = probs_Y_HZc_bar[idx]
            probs_obs = probs_Y_obs[idx]
        else:
            K = N_o_jk.shape[1]
            probs1 = probs2 = probs_obs = np.ones(K) / K  # fallback to uniform

        probs_rows_HZc.append(probs1)
        probs_rows_HZc_bar.append(probs2)
        probs_rows_obs.append(probs_obs)

    # Build dataframe
    df_Prob_Y_Hzc = pd.DataFrame(probs_rows_HZc, columns=[f"P_HZc(Y={k})" for k in range(probs_Y_HZc.shape[1])])
    df_Prob_Y_Hzc_bar = pd.DataFrame(probs_rows_HZc_bar,
                                     columns=[f"P_HZc_bar(Y={k})" for k in range(probs_Y_HZc_bar.shape[1])])
    df_Prob_Y_obs = pd.DataFrame(probs_rows_obs, columns=[f"P_obs(Y={k})" for k in range(probs_Y_obs.shape[1])])

    # posterior predictive probabilities* (P_Hzc/P_Hzc_bar)
    # print('df_Prob_Hzc', df_Prob_Y_Hzc)
    # print('df_Prob_Hzc_bar', df_Prob_Y_Hzc_bar)

    P_Hz_c = df_scores.loc[df_scores['Variables'] == tuple(Z_cols), 'P_HZ_c'].values
    P_Hz_c_bar = df_scores.loc[df_scores['Variables'] == tuple(Z_cols), 'P_HZ_c_bar'].values

    P_Y_alg = (df_Prob_Y_Hzc * P_Hz_c[0]).to_numpy() + (df_Prob_Y_Hzc_bar * P_Hz_c_bar[0]).to_numpy()

    if Z_cols == FS:
        P_Y_exp = df_Prob_Y_Hzc_bar.to_numpy()
        P_Y_obs = df_Prob_Y_obs.to_numpy()

y_true = De_test[outcome]

alg_loss = log_loss(y_true, P_Y_alg)
print(f"Log loss algorithm: {alg_loss}")
exp_loss = log_loss(y_true, P_Y_exp)
print(f"Log loss experimental: {exp_loss}")
obs_loss = log_loss(y_true, P_Y_obs)
print(f"Log loss observational: {obs_loss}")

print(f"Log loss GT: {GT_log_loss}")
