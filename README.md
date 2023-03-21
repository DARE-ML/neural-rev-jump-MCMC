# neural-rev-jump-MCMC
Neural architecture uncertainty quantification with reversible jump MCMC

## Required Packages

Requires Python 3.7 or later with the following libraries: 

`matplotlib==3.5.3`
`scikit-learn==1.1.2`
`sklearn`

## Running Files

Dependent on the type of problems to be run, run the respective files. 

Files named with `_classification.py` is for classification problems, files named with `_hid.py` is for Jump-H, files named with `_langevin.py` is for RJMCMC with Langevin gradient. 

```{bash}
neural-rev-jump-MCMC (master) python mt_bnn_dts_classification_langevin.py 

Arguments to adjust:
--problem PROBLEM         Name of the problem to work with
--min_perf MIN_PERF       Stop when RMSE reches this point
--subtasks SUBTASKS       The number of sub-models to jump 
--num_samples NUM_SAMPLES The number of samples needed to train the model
--mtaskNet MTASKNET       The sub-model's network architecture
--prob_type PROB_TYPE     Type of problem (regression or classification)
```

## Data

The data used in this project is available [here](https://github.com/DARE-ML/neural-rev-jump-MCMC/tree/master/Data_OneStepAhead) for regression and [here](https://github.com/DARE-ML/neural-rev-jump-MCMC/tree/master/DATA) for classification
