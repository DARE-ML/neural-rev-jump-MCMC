# Sequential reversible jump MCMC for dynamic Bayesian neural networks

The aim of this research is neural architecture uncertainty quantification with reversible jump MCMC. We provide the code in Python with data and instructions that enable their use and extension. We provide results in the paper for some benchmark problems showing the strengths and weaknesses of the method.

## Method

We presented a dynamic Bayesian neural network framework that utilizes S-RJMCMC to effectively train model parameters in a dynamic envirionment using cascaded neural networks with shared knowledge representation among sub-models.

![image](https://github.com/DARE-ML/neural-rev-jump-MCMC/assets/54335413/bf294034-8852-494b-bae2-28489d7b6cca)


## Code

We provide complete code `mt_bnn_dts_hid_langevin_classification.py` for regression and classification problems using dynamic hidden neurons (Jump-H) and Langevin-gradient proposal distribution. Other variations of the method, presented in the paper, can be found in `misc` folder.

Below are important hyperparameters that users can adjust to replicate the experiments. 
```{bash}
neural-rev-jump-MCMC (main) python mt_bnn_dts_hid_langevin_classification.py 

Arguments to adjust:
--problem     PROBLEM         Name of the problem to work with
--min_perf    MIN_PERF        Stop when RMSE reches this point
--subtasks    SUBTASKS        The number of sub-models to jump 
--num_samples NUM_SAMPLES     The number of samples needed to train the model
--mtaskNet    MTASKNET        The sub-model's network architecture
--prob_type   PROB_TYPE       Type of problem (regression or classification)
```
### Required Packages

Requires Python 3.7 or later with the following libraries: 

`matplotlib==3.5.3`
`scikit-learn==1.1.2`
`scipy`
`numpy`
`pandas`

## Data

The data used in this project is available [here](https://github.com/DARE-ML/neural-rev-jump-MCMC/tree/main/Data_OneStepAhead) for regression and [here](https://github.com/DARE-ML/neural-rev-jump-MCMC/tree/main/DATA) for classification

## Results

Our results demonstrate that the methodology excels not only in training and achieving good accuracy but also in providing uncertainty quantification, especially when dealing with dynamic input and dynamic hidden neuron settings. We further observe that the predictive performance goes up as the number of sub-models increases. This suggests that the jumps effectively facilitate knowledge transfer from one sub-model to another, leading to an overall enhancement in the performance of all sub-models.

![image](https://github.com/DARE-ML/neural-rev-jump-MCMC/assets/54335413/200bbf2f-29c9-43a2-9425-8170fc5b44f3)

## Publication

Nguyen, Nhat Minh & Tran, Minh-Ngoc & Chandra, Rohitash. (2024). Sequential reversible jump MCMC for dynamic Bayesian neural networks. Neurocomputing. 564. 126960. 10.1016/j.neucom.2023.126960.

## Invited Talk

The methodology was demonstrate by Nhat Minh Nguyen during the seminar hosted by Transitional Artificial Intelligence Research Group, School of Mathematics and Statistics, UNSW Sydney. 
The talk can be found here: https://www.youtube.com/watch?v=-13hc-vzq-Q

