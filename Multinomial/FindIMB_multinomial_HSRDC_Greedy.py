"""
The code uses multinomial-Dirichlet distributions for binary and categorical data with more than two or three categories, offering closed-form 
marginal solutions. Instead of an exhaustive search, we use a greedy search method. This approach is applied to find the sets in Dₒ, Dₑ, and (Dₒ + Dₑ) 
that produce the best Dₒ-based, Dₑ-based, and (Dₒ + Dₑ)-based estimators. For FindIMB, we score the best set based on P(De|Do, HZc) (Eq.5 for HZc from 
the UAI paper), and for the other estimators, we score based on the  marginal probability of Do, De, and (Do+De) for each set: P(Do|HZ_not_c), P(De|HZ_not_c)
and P(Do, De|HZ_not_c) (Eq.5 for HZ_not_c from UAI paper, second row of the table S1 adjusted)

For each of the 20 runs, we also report the three sets that maximize either P(HZc|Do, De) or P(HZ_not_c|Do, De).
For the De_test cases, we calculate the fraction of patients who received A1 and had a good outcome (PONV = 0) and report the average of this value. 
We do the same for patients who were treated with A2 in the De_test cases.
We evaluate the estimators using Expected Utility, Binary-Cross entropy, and Expected Calibration Error (ECCE).
"""
