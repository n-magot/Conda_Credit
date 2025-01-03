# Conda_Credit
A 2024 project on "Individualized Prediction of Treatment Effects Using Data from Both Embedded Clinical Trials and Electronic Health Records".
## Implementation
A repository that contains folders for tasks: 

1. Calculate the conditional probabilities for the outcome with and without controlling the confounding.
   
3. Create a random dag and simulate mixed data (binary and continuous) from a DAG.
   
5. Python implementation for the extension of the FindIMB algorithm which is limited to categorical data proposed by [Triantafillou et al. (2021)](https://proceedings.mlr.press/v161/triantafillou21a.html), to ordinal and binary outcomes, binary treatments, and mixed covariates. FindIMB extended algorithm, uses Bayesian regression models and approximate inference for combining observational and experimental data to learn causal and interventional Markov boundaries and improve causal estimation, [Lelova et al. (2024) ](https://proceedings.mlr.press/v246/lelova24a.html). This implementation approximates MB using Bayesian Regression models for low-dimensional datasets.
   
7. An optimized version of FindIMB for mixed data types using Brute-force search with pruning for scaling up the method to allow for more conditioning covariates.
   
9. A way to generate synthetic data (both experimental and observational data) for binary outcomes in the case of (a) binary or (b) both binary and ordinal covariates using a logistic regression model. Based on the ground truth model that the experimental data entails we synthetically introduce a confounding variable that affects both the treatment and the outcome and we create observational data.
    
11. A new algorithm (TranSampler) has been proposed for scoring sets that can be used to transport causal knowledge from the source to a target population when we have available only experimental data from the source population (De) and observational data from the target population (Do*). These sets are s-admissible and we can identify them only if they are also adjustment sets for the target population.

    ## Contact
    If you have any questions, email me:  [konlelova@gmail.com](mailto:konlelova@gmail.com) 

