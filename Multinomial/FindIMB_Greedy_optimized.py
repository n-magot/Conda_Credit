"""Try to optimize the running time."""
import itertools
import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def get_counts_multiZ(df, Z_cols, Y_col, Z_reference=None):
    df = df.copy()

    for col in Z_cols:
        df[col] = df[col].astype('category')

    df[Y_col] = df[Y_col].astype('category')
    Y_values = df[Y_col].cat.categories.tolist()

    # If no external reference provided, use all combos from this df
    if Z_reference is None:
        Z_categories = [sorted(df[col].cat.categories) for col in Z_cols]
        Z_values = list(itertools.product(*Z_categories))
    else:
        # Use the pre-defined reference (same across datasets)
        Z_values = Z_reference

    index_map = {z: i for i, z in enumerate(Z_values)}
    y_map = {y: i for i, y in enumerate(Y_values)}

    N_j = np.zeros(len(Z_values), dtype=int)
    N_jk = np.zeros((len(Z_values), len(Y_values)), dtype=int)

    for _, row in df.iterrows():
        z_tuple = tuple(row[Z_cols])
        y_val = row[Y_col]
        if any(pd.isnull(v) for v in z_tuple) or pd.isnull(y_val):
            continue
        if z_tuple in index_map:
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


def compute_posterior_predictive_single(N_o_jk, N_o_j, N_e_jk, N_e_j, N_o_e_jk, N_o_e_j, alpha_jk, alpha_j):
    """
    Returns:
    - probs_HZc: P(Y|do(X), Z, HZc) predictive probs under H_Z^c
    - probs_HZc_bar: P(Y|do(X), Z, HZc_bar) predictive probs under H̄_Z^c
    """

    numerator_HZc_bar = N_e_jk + alpha_jk
    denominator_HZc_bar = (N_e_j + alpha_j)[:, np.newaxis]
    probs_Y_HZc_bar = numerator_HZc_bar / denominator_HZc_bar

    numerator_obs = N_o_jk + alpha_jk
    denominator_obs = (N_o_j + alpha_j)[:, np.newaxis]
    probs_Y_obs = numerator_obs / denominator_obs
    
    numerator_all = N_o_e_jk + alpha_jk
    denominator_all = (N_o_e_j + alpha_j)[:, np.newaxis]
    probs_Y_all = numerator_all / denominator_all

    return probs_Y_HZc_bar, probs_Y_obs, probs_Y_all

# ============================================================
# Dirichlet-Multinomial provider
# ============================================================

def dirichlet_prob_provider(Test_data, probs_matrix):
    """
    probs_matrix: shape (N, 2), row-aligned with Test_data
    """
    probs_T0 = probs_matrix[:, 0]
    probs_T1 = probs_matrix[:, 1]
    return probs_T0, probs_T1

def evaluate_expected_outcome(De_test, Do, De, df_scores, subsets, treatment, outcome, best_set_De, best_set_Do, best_set_Do_De):
    """
    Vectorized evaluation of expected outcome under optimal treatment for each row.
    
    Returns:
      avgY1_obs, avgY1_exp, avgY1_alg
    """
    Do_De = pd.concat([Do, De], ignore_index=True)

    N = len(De_test)
    K = 2 # For binary outcomes

    # Prepare T0 and T1 datasets
    Dtest_T0 = De_test.copy()
    Dtest_T0[treatment] = 0
    Dtest_T1 = De_test.copy()
    Dtest_T1[treatment] = 1

    # Storage for probabilities
    def compute_BMA_probs(De_input):
        P_Y_alg_accum = np.zeros((N, K))
        P_Y_exp = None
        P_Y_obs = None

        for Z_cols in subsets:
            Z_cols_list = list(Z_cols)
            w_c = float(df_scores.loc[df_scores['Variables'] == tuple(Z_cols), 'P_HZ_c'])
            w_cb = float(df_scores.loc[df_scores['Variables'] == tuple(Z_cols), 'P_HZ_c_bar'])
            
                
            # Build a universal Z_reference across both datasets
            Z_categories = []
            for col in Z_cols:
                cats_Do = Do[col].astype('category').cat.categories
                cats_De = De[col].astype('category').cat.categories
                all_cats = sorted(set(cats_Do) | set(cats_De))  # union of both
                Z_categories.append(all_cats)
            
            Z_reference = list(itertools.product(*Z_categories))

            Z_vals_Do, _, N_o_j, N_o_jk, _ = get_counts_multiZ(Do, Z_cols, outcome, Z_reference=Z_reference)
            Z_vals_De, _, N_e_j, N_e_jk, _ = get_counts_multiZ(De, Z_cols, outcome, Z_reference=Z_reference)

            alpha_jk = np.ones_like(N_o_jk)
            alpha_j = np.sum(alpha_jk, axis=1)

            probs_Y_HZc, probs_Y_HZc_bar, probs_Y_obs_tmp = compute_posterior_predictive_both_hypotheses(
                N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j
            )

            Z_config_to_index = {tuple(z): i for i, z in enumerate(Z_vals_De)}
            probs_rows_HZc, probs_rows_HZc_bar, probs_rows_obs_tmp = [], [], []

            for _, row in De_input.iterrows():
                z_tuple = tuple(row[col] for col in Z_cols_list)
                idx = Z_config_to_index.get(z_tuple)
                if idx is not None:
                    probs_rows_HZc.append(probs_Y_HZc[idx])
                    probs_rows_HZc_bar.append(probs_Y_HZc_bar[idx])
                    probs_rows_obs_tmp.append(probs_Y_obs_tmp[idx])
                else:
                    probs_rows_HZc.append(np.ones(K) / K)
                    probs_rows_HZc_bar.append(np.ones(K) / K)
                    probs_rows_obs_tmp.append(np.ones(K) / K)

            probs_rows_HZc = np.asarray(probs_rows_HZc)
            probs_rows_HZc_bar = np.asarray(probs_rows_HZc_bar)
            probs_rows_obs_tmp = np.asarray(probs_rows_obs_tmp)

            # BMA accumulation
            P_Y_alg_accum += w_c * probs_rows_HZc + w_cb * probs_rows_HZc_bar
            


        P_Y_alg = P_Y_alg_accum / np.clip(P_Y_alg_accum.sum(axis=1, keepdims=True), 1e-12, None)
        
        Z_vals_Do, _, N_o_j, N_o_jk, _ = get_counts_multiZ(Do, list(best_set_Do), outcome, Z_reference=Z_reference)
        Z_vals_De, _, N_e_j, N_e_jk, _ = get_counts_multiZ(De, list(best_set_De), outcome, Z_reference=Z_reference)
        Z_vals_Do_De, _, N_o_e_j, N_o_e_jk, _ = get_counts_multiZ(Do_De, list(best_set_Do_De), outcome, Z_reference=Z_reference)

        alpha_jk = np.ones_like(N_o_jk)
        alpha_j = np.sum(alpha_jk, axis=1)

        probs_Y_HZc_bar, probs_Y_obs_tmp, probs_Y_all = compute_posterior_predictive_single(
            N_o_jk, N_o_j, N_e_jk, N_e_j, N_o_e_jk, N_o_e_j, alpha_jk, alpha_j
        )
        
        probs_rows_HZc_bar = np.asarray(probs_rows_HZc_bar)
        probs_rows_obs_tmp = np.asarray(probs_rows_obs_tmp)
        probs_Y_all = np.asarray(probs_Y_all)
        
        P_Y_exp = probs_rows_HZc_bar        
        P_Y_obs = probs_rows_obs_tmp
        P_Y_all = probs_Y_all
        
        
        return P_Y_alg, P_Y_exp, P_Y_obs, P_Y_all

    # Compute BMA probabilities for T0 and T1
    P_Y_alg_T0, P_Y_exp_T0, P_Y_obs_T0, P_Y_all_T0 = compute_BMA_probs(Dtest_T0)

    P_Y_alg_T1, P_Y_exp_T1, P_Y_obs_T1, P_Y_all_T1 = compute_BMA_probs(Dtest_T1)

    # Determine optimal treatment for each row (vectorized), considering good outcome Y=0
    def best_treatment_probs(P_T0, P_T1):
        best_T = (P_T1[:, 0] > P_T0[:, 0]).astype(int)
        
        """If best_T is always 1"""
        # best_T = np.ones(P_T1.shape[0], dtype=int)
        
        best_probs = np.where(best_T[:, None] == 1, P_T1, P_T0)
        avgY0 = best_probs[:, 0].mean()
        T0Count = int(np.sum(best_T == 0))
        T1Count = int(np.sum(best_T == 1))
        return T0Count, T1Count, avgY0

    T0_alg, T1_alg, avgY1_alg = best_treatment_probs(P_Y_alg_T0, P_Y_alg_T1)
    T0_exp, T1_exp, avgY1_exp = best_treatment_probs(P_Y_exp_T0, P_Y_exp_T1)
    T0_obs, T1_obs, avgY1_obs = best_treatment_probs(P_Y_obs_T0, P_Y_obs_T1)
    T0_all, T1_all, avgY1_all = best_treatment_probs(P_Y_all_T0, P_Y_all_T1)


    return T0_obs, T1_obs, avgY1_obs, T0_exp, T1_exp, avgY1_exp,  T0_alg, T1_alg, avgY1_alg, T0_all, T1_all, avgY1_all


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        bin_size = np.sum(mask)
        if bin_size > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (bin_size / N) * np.abs(acc - conf)

    return ece


def COMB_names(elements, k):
    if k >= len(elements):
        return [tuple(elements)]
    treatment = elements[0]
    rest = elements[1:]
    combos = list(itertools.combinations(rest, k - 1))
    return [(treatment,) + c for c in combos]


def run_layered_pipeline_names(Do, De, X_col, Z_cols, Y_col, threshold=0.1, priors_val=1):
    """
    Layered subset search with pruning, using log-space normalization.
    Returns df_results with the same schema as the exhaustive pipeline.
    """
    FS_vars = [X_col] + Z_cols
    N = len(FS_vars) - 1
    list_to_invest = [tuple(FS_vars)]

    log_P_HZc = {}
    log_P_HZc_bar = {}
    
    # Build a universal Z_reference across both datasets
    Z_categories = []
    for col in FS_vars:
        cats_Do = Do[col].astype('category').cat.categories
        cats_De = De[col].astype('category').cat.categories
        all_cats = sorted(set(cats_Do) | set(cats_De))  # union of both
        Z_categories.append(all_cats)
    
    Z_reference = list(itertools.product(*Z_categories))

    
    _, _, N_o_j, N_o_jk, _ = get_counts_multiZ(Do, FS_vars, Y_col, Z_reference=Z_reference)
    _, _, N_e_j, N_e_jk, _ = get_counts_multiZ(De, FS_vars, Y_col, Z_reference=Z_reference)
    alpha_jk = np.ones_like(N_o_jk) * priors_val
    priors = alpha_jk - 1

    log_P_HZc[tuple(FS_vars)] = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
    log_P_HZc_bar[tuple(FS_vars)] = P_De_given_HZc_bar_log(N_e_jk, priors)

    # Layered search
    for k in range(N, 0, -1):
        next_layer = []
        for s in list_to_invest:
            next_layer.extend(COMB_names(s, k))
        list_to_invest = []

        for Z_subset in next_layer:
            if Z_subset in log_P_HZc:
                continue
            
            # Build a universal Z_reference across both datasets
            Z_categories = []
            for col in list(Z_subset):
                cats_Do = Do[col].astype('category').cat.categories
                cats_De = De[col].astype('category').cat.categories
                all_cats = sorted(set(cats_Do) | set(cats_De))  # union of both
                Z_categories.append(all_cats)
            
            Z_reference = list(itertools.product(*Z_categories))

            _, _, N_o_j, N_o_jk, _ = get_counts_multiZ(Do, list(Z_subset), Y_col, Z_reference=Z_reference)
            _, _, N_e_j, N_e_jk, _ = get_counts_multiZ(De, list(Z_subset), Y_col, Z_reference=Z_reference)
            alpha_jk = np.ones_like(N_o_jk) * priors_val
            priors = alpha_jk - 1

            log_P_HZc[Z_subset] = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
            log_P_HZc_bar[Z_subset] = P_De_given_HZc_bar_log(N_e_jk, priors)
            list_to_invest.append(Z_subset)

        # Normalize in log space for pruning
        log_total = logsumexp([np.logaddexp(log_P_HZc[z], log_P_HZc_bar[z]) for z in log_P_HZc])
        Scores_HZc = {z: np.exp(log_P_HZc[z] - log_total) for z in log_P_HZc}
        Scores_HZc_bar = {z: np.exp(log_P_HZc_bar[z] - log_total) for z in log_P_HZc_bar}

        # Prune subsets with very low probability
        list_to_invest = [z for z in list_to_invest
                          if Scores_HZc[z] > threshold or Scores_HZc_bar[z] > threshold]

        if not list_to_invest:
            break

    # Final df_results (same schema as exhaustive pipeline)
    df_results = pd.DataFrame({
        'Variables': list(log_P_HZc.keys()),
        'P(De|Do,HcZ) log': list(log_P_HZc.values()),
        'P(De|Do,HcZ_bar) log': list(log_P_HZc_bar.values())
    })

    # Final normalization in log space (over all visited subsets)
    log_total = logsumexp([np.logaddexp(log_P_HZc[z], log_P_HZc_bar[z]) for z in log_P_HZc])
    df_results['P_HZ_c'] = df_results['P(De|Do,HcZ) log'].apply(lambda v: float(np.exp(v - log_total)))
    df_results['P_HZ_c_bar'] = df_results['P(De|Do,HcZ_bar) log'].apply(lambda v: float(np.exp(v - log_total)))

    return df_results


def run_layered_pipeline_marginal(data, X_col, Z_cols, Y_col, threshold=0.1, priors_val=1):
    """
    Layered subset search to compute P(data | Z) for any dataset.
    Works for either experimental (De) or observational (Do) data.
    """

    FS_vars = [X_col] + Z_cols
    N = len(FS_vars) - 1
    list_to_invest = [tuple(FS_vars)]  # Start with full set
    log_P_data = {}  # store log probabilities per subset

    # --- First full subset ---
    Z_categories = []
    for col in FS_vars:
        cats = data[col].astype('category').cat.categories
        Z_categories.append(cats)
    Z_reference = list(itertools.product(*Z_categories))

    _, _, N_j, N_jk, _ = get_counts_multiZ(data, FS_vars, Y_col, Z_reference=Z_reference)
    alpha_jk = np.ones_like(N_jk) * priors_val
    priors = alpha_jk - 1
    log_P_data[tuple(FS_vars)] = P_De_given_HZc_bar_log(N_jk, priors)

    # --- Layered search ---
    for k in range(N, 0, -1):
        next_layer = []
        for s in list_to_invest:
            next_layer.extend(COMB_names(s, k))
        list_to_invest = []

        for Z_subset in next_layer:
            if Z_subset in log_P_data:
                continue

            # Build consistent category reference
            Z_categories = []
            for col in list(Z_subset):
                cats = data[col].astype('category').cat.categories
                Z_categories.append(cats)
            Z_reference = list(itertools.product(*Z_categories))

            _, _, N_j, N_jk, _ = get_counts_multiZ(data, list(Z_subset), Y_col, Z_reference=Z_reference)
            alpha_jk = np.ones_like(N_jk) * priors_val
            priors = alpha_jk - 1
            log_P_data[Z_subset] = P_De_given_HZc_bar_log(N_jk, priors)

            list_to_invest.append(Z_subset)

        # --- Normalize and prune ---
        log_total = logsumexp(list(log_P_data.values()))
        Scores = {z: np.exp(log_P_data[z] - log_total) for z in log_P_data}

        # Keep only high-scoring subsets
        list_to_invest = [z for z in list_to_invest if Scores[z] > threshold]

        if not list_to_invest:
            break

    # --- Build final results ---
    df_results = pd.DataFrame({
        'Variables': list(log_P_data.keys()),
        'logP(data|Z)': list(log_P_data.values())
    })

    # Normalize in log-space
    log_total = logsumexp(list(log_P_data.values()))
    df_results['P(data|Z)'] = df_results['logP(data|Z)'].apply(lambda v: float(np.exp(v - log_total)))

    return df_results



def bma_predict_and_evaluate(Do, De, De_test, df_scores, treatment, outcome, best_set_De, best_set_Do, best_set_Do_De):
    """
    Perform BMA over subsets and evaluate the metrics using the new expected utility function.
    The evaluation metrics are binary cross-entropy, expected calibration error, and expected utility.
    """
    Do_De = pd.concat([Do, De], ignore_index=True)

    Ntest = len(De_test)
    K = 2
    P_Y_alg_accum = np.zeros((Ntest, K))
    P_Y_exp = None
    P_Y_obs = None

    # Keep track of subsets for expected outcome evaluation
    subsets = [list(t) for t in df_scores['Variables']]

    # Iterate over subsets to compute BMA
    for _, row in df_scores.iterrows():
        Z_cols = list(row['Variables'])
        w_c = float(row['P_HZ_c'])
        w_cb = float(row['P_HZ_c_bar'])
        
            
        # Build a universal Z_reference across both datasets
        Z_categories = []
        for col in Z_cols:
            cats_Do = Do[col].astype('category').cat.categories
            cats_De = De[col].astype('category').cat.categories
            all_cats = sorted(set(cats_Do) | set(cats_De))  # union of both
            Z_categories.append(all_cats)
        
        Z_reference = list(itertools.product(*Z_categories))
        

        Z_vals_Do, _, N_o_j, N_o_jk, _ = get_counts_multiZ(Do, Z_cols, outcome, Z_reference=Z_reference)
        Z_vals_De, _, N_e_j, N_e_jk, _ = get_counts_multiZ(De, Z_cols, outcome, Z_reference=Z_reference)

        alpha_jk = np.ones_like(N_o_jk)
        alpha_j = np.sum(alpha_jk, axis=1)

        probs_Y_HZc, probs_Y_HZc_bar, probs_Y_obs_tmp = compute_posterior_predictive_both_hypotheses(
            N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j
        )

        Z_config_to_index = {tuple(z): i for i, z in enumerate(Z_vals_De)}
        probs_rows_HZc, probs_rows_HZc_bar, probs_rows_obs_tmp = [], [], []

        for _, rtest in De_test.iterrows():
            z_tuple = tuple(rtest[col] for col in Z_cols)
            idx = Z_config_to_index.get(z_tuple)
            if idx is not None:
                probs_rows_HZc.append(probs_Y_HZc[idx])
                probs_rows_HZc_bar.append(probs_Y_HZc_bar[idx])
                probs_rows_obs_tmp.append(probs_Y_obs_tmp[idx])
            else:
                probs_rows_HZc.append(np.ones(K) / K)
                probs_rows_HZc_bar.append(np.ones(K) / K)
                probs_rows_obs_tmp.append(np.ones(K) / K)

        probs_rows_HZc = np.asarray(probs_rows_HZc)
        probs_rows_HZc_bar = np.asarray(probs_rows_HZc_bar)
        probs_rows_obs_tmp = np.asarray(probs_rows_obs_tmp)

        P_Y_alg_accum += w_c * probs_rows_HZc + w_cb * probs_rows_HZc_bar

    # Normalize BMA
    P_Y_alg = P_Y_alg_accum / np.clip(P_Y_alg_accum.sum(axis=1, keepdims=True), 1e-12, None)
    
    Z_vals_Do, _, N_o_j, N_o_jk, _ = get_counts_multiZ(Do, list(best_set_Do), outcome, Z_reference=Z_reference)
    Z_vals_De, _, N_e_j, N_e_jk, _ = get_counts_multiZ(De, list(best_set_De), outcome, Z_reference=Z_reference)
    Z_vals_Do_De, _, N_o_e_j, N_o_e_jk, _ = get_counts_multiZ(Do_De, list(best_set_Do_De), outcome, Z_reference=Z_reference)


    alpha_jk = np.ones_like(N_o_jk)
    alpha_j = np.sum(alpha_jk, axis=1)

    probs_Y_HZc_bar, probs_Y_obs_tmp, probs_Y_all = compute_posterior_predictive_single(
        N_o_jk, N_o_j, N_e_jk, N_e_j, N_o_e_jk, N_o_e_j, alpha_jk, alpha_j
    )
    
    probs_rows_HZc_bar = np.asarray(probs_rows_HZc_bar)
    probs_rows_obs_tmp = np.asarray(probs_rows_obs_tmp)
    probs_Y_all = np.asarray(probs_Y_all)
    
    P_Y_exp = probs_rows_HZc_bar    
    P_Y_obs = probs_rows_obs_tmp   
    P_Y_all = probs_Y_all

    # Evaluate expected outcomes using the new function
    T0_obs, T1_obs, avgY1_obs, T0_exp, T1_exp, avgY1_exp,  T0_alg, T1_alg, avgY1_alg, T0_all, T1_all, avgY1_all = evaluate_expected_outcome(
        De_test=De_test,
        Do=Do,
        De=De,
        df_scores=df_scores,
        subsets=subsets,
        treatment=treatment,
        outcome=outcome,
        best_set_De = best_set_De,
        best_set_Do = best_set_Do,
        best_set_Do_De = best_set_Do_De
    )

    # Log-loss
    y_true = De_test[outcome].to_numpy()
    alg_loss = log_loss(y_true, P_Y_alg)
    exp_loss = log_loss(y_true, P_Y_exp)
    obs_loss = log_loss(y_true, P_Y_obs)
    all_loss = log_loss(y_true, P_Y_all)

    # ECE
    ece_alg = expected_calibration_error(y_true, P_Y_alg[:, 1], n_bins=10)
    ece_exp = expected_calibration_error(y_true, P_Y_exp[:, 1], n_bins=10)
    ece_obs = expected_calibration_error(y_true, P_Y_obs[:, 1], n_bins=10)
    ece_all = expected_calibration_error(y_true, P_Y_all[:, 1], n_bins=10)

    # Pack results
    metrics = {
        "algorithm": dict(T0=T0_alg, T1=T1_alg, avgY1=avgY1_alg, logloss=alg_loss, ece=ece_alg),
        "experimental": dict(T0=T0_exp, T1=T1_exp, avgY1=avgY1_exp, logloss=exp_loss, ece=ece_exp),
        "observational": dict(T0=T0_obs, T1=T1_obs, avgY1=avgY1_obs, logloss=obs_loss, ece=ece_obs),
        "all": dict(T0=T0_all, T1=T1_all, avgY1=avgY1_all, logloss=all_loss, ece=ece_all),
        "P_Y": dict(alg=P_Y_alg, exp=P_Y_exp, obs=P_Y_obs)
    }

    return metrics

n_runs = 2

list_Expected_alg = []
list_Expected_exp = []
list_Expected_obs = []
list_Expected_all = []

list_T0_alg = []
list_T0_exp = []
list_T0_obs = []
list_T0_all = []

list_T1_alg = []
list_T1_exp = []
list_T1_obs = []
list_T1_all = []

list_log_alg = []
list_log_exp = []
list_log_obs = []
list_log_all = []

list_ece_alg = []
list_ece_exp = []
list_ece_obs = []
list_ece_all = []

Do = pre_pross_data(obs_data)
De_all = pre_pross_data(exp_data)

treatment = 'T'
outcome = 'Y'
covariate_with_T = [col for col in Do.columns if col != outcome]
covariates_without_T = [col for col in Do.columns if col not in [outcome, treatment]]
# Z_cols = covariate_with_T


df_Score_Do = run_layered_pipeline_marginal(Do, treatment, covariates_without_T, outcome, threshold=0.1)
print(df_Score_Do)

# df_results is the output of your function
max_row_Do = df_Score_Do.loc[df_Score_Do['P(data|Z)'].idxmax()]
# Extract the subset and its probability
best_set_Do = max_row_Do['Variables']
best_prob = max_row_Do['P(data|Z)']

print("Best subset in Do:", best_set_Do)
print("Probability for the best subset in Do:", best_prob)

for k in range(n_runs):

    De, De_test = train_test_split(De_all, test_size=0.3, random_state=k)
    
    Do_De = pd.concat([Do, De], ignore_index=True)
    
    X_test = De_test[covariate_with_T]
    y_test = De_test[outcome]
    
    df_Score_De = run_layered_pipeline_marginal(De, treatment, covariates_without_T, outcome, threshold=0.1)
    print(df_Score_De)
    
    max_row_De = df_Score_De.loc[df_Score_De['P(data|Z)'].idxmax()]
    best_set_De = max_row_De['Variables']
    best_prob = max_row_De['P(data|Z)']
    
    print("Best subset in De:", best_set_De)
    print("Probability for the best subset in De:", best_prob)
    
    df_Score_Do_De = run_layered_pipeline_marginal(Do_De, treatment, covariates_without_T, outcome, threshold=0.1)
    print(df_Score_Do_De)
    
    max_row_Do_De = df_Score_Do_De.loc[df_Score_Do_De['P(data|Z)'].idxmax()]
    best_set_Do_De = max_row_Do_De['Variables']
    best_prob = max_row_Do_De['P(data|Z)']
    
    print("Best subset in (Do + De):", best_set_Do_De)
    print("Probability for the best subset in (Do + De):", best_prob)
    
    
    # 1) Run layered search (or exhaustive in the older FindIMB version)
    df_scores = run_layered_pipeline_names(Do, De, treatment, covariates_without_T, outcome, threshold=0.1)
    print(df_scores)
    
    
    # 2) Do Bayesian model averaging across subsets and evaluate
    metrics = bma_predict_and_evaluate(Do, De, De_test, df_scores, treatment, outcome, best_set_De, best_set_Do, best_set_Do_De)

    list_Expected_alg.append(metrics['algorithm']['avgY1'])
    list_Expected_exp.append(metrics['experimental']['avgY1'])
    list_Expected_obs.append(metrics['observational']['avgY1'])
    list_Expected_all.append(metrics['all']['avgY1'])

    list_T0_alg.append(metrics['algorithm']['T0'])
    list_T0_exp.append(metrics['experimental']['T0'])
    list_T0_obs.append(metrics['observational']['T0'])
    list_T0_all.append(metrics['all']['T0'])

    list_T1_alg.append(metrics['algorithm']['T1'])
    list_T1_exp.append(metrics['experimental']['T1'])
    list_T1_obs.append(metrics['observational']['T1'])
    list_T1_all.append(metrics['all']['T1'])
    
    list_log_alg.append(metrics['algorithm']['logloss'])
    list_log_exp.append(metrics['experimental']['logloss'])
    list_log_obs.append(metrics['observational']['logloss'])
    list_log_all.append(metrics['all']['logloss'])

    list_ece_alg.append(metrics['algorithm']['ece'])
    list_ece_exp.append(metrics['experimental']['ece'])
    list_ece_obs.append(metrics['observational']['ece'])
    list_ece_all.append(metrics['all']['ece'])


print('Expected Utility algorithm:', list_Expected_alg)
print('Expected Utility experimental:',list_Expected_exp)
print('Expected Utility observational:', list_Expected_obs)
print('Expected Utility all:', list_Expected_all)

print('T0 algorithm:', list_T0_alg)
print("T0 experimental", list_T0_exp)
print("T0 observational", list_T0_obs)
print("T0 all", list_T0_all)

print('T1 algorithm:', list_T1_alg)
print("T1 experimental", list_T1_exp)
print("T1 observational", list_T1_obs)
print("T1 all", list_T1_all)

print('Log loss algorithm:', list_log_alg)
print("Log loss experimental", list_log_exp)
print("Log loss observational", list_log_alg)
print("Log loss all", list_log_all)

print("ECE Algorithm:", list_ece_alg)
print("ECE Experimental", list_ece_exp)
print("ECE Observational", list_ece_obs)
print("ECE all", list_ece_all)

