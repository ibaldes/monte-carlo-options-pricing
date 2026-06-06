*****************************************************

Code for Monte-Carlo pricing of options assuming the Black-Scholes model and risk-neutral probability. Comparison of prices and greeks to known analytic formulas. 

*****************************************************
# Libraries Used 

- sys
- pathlib
- pandas
- numpy
- matplotlib 
- scipy

*****************************************************
# SUMMARY 

## options_benchmark folder
Contains standard scripts for calculating options pricing using analytic formulas and monte-carlo. Monte-Carlo greeks calculated using finite difference approach.

* EuropeanVanilla.py contains functions for the following:
1. Analytic results of Black Scholes Vanilla European options with or without Greeks (delta, gamma, vega, theta, rho).
2. Monte-Carlo implementation Vanilla European options pricing with or without Greeks. These also return standard errors of the prices and greeks.

* EuropeanBarrier.py contains functions for the following:
1. Analytic pricing for Knock-in/Knock-out European puts and calls.
2. Monte-Carlo pricing for Knock-in/Knock-out European puts and calls with Greeks. These also return standard errors of the prices and greeks. (To do: improve estimates of the greeks using more advanced techniques)

* LookBack.py contains functions for the following:
1. Analytic pricing for Look Back Puts and Calls with Floating or Fixed Strikes.
2. Monte-Carlo pricing for Look Back Puts and Calls with Floating or Fixed Strikes. These also return standard errors of the prices and greeks.

* Asian.py contains functions for the following:
1. Approximate analytic pricing for Asian options with arithmetic averaging applied to the price or strike.
2. Monte-Carlo pricing for Asian options with arithmetic averaging applied to the price or strike. These also return standard errors of the prices and greeks.


* PricingExample.py calls the functions from EuropeanVanilla.py and EuropeanBarrier.py showing some example prices.

## options_antithetic folder
Similar to options_benchmark, but uses antithetic variates. The first-half batch of simulations draws from the normal distribution as before. The second-half batch of simulations uses the antithetic variates (i.e. reverses the sign of the brownian motion of the first-half batch).

The scripts are:

* EuropeanVanillaAntithetic.py is as the above EuropeanVanilla.py, but with Antithetic Variates for the Monte-Carlo.

* EuropeanBarrierAntithetic.py is as the above EuropeanBarrier.py, but with Antithetic Variates for the Monte-Carlo.

* LookBackAntithetic.py is as the above LookBack.py, but with Antithetic Variates for the Monte-Carlo.

* AsianAntithetic.py is as the above Asian.py, but with Antithetic Variates for the Monte-Carlo.

## options_pathwise folder
Similar to options_benchmark, but uses antithetic variates, and pathwise derivatives for evaluating the greeks Delta and Gamma.

The scripts are:

* EuropeanVanillaPathwise.py is as the above EuropeanVanillaAntithetic.py, but with pathwise Delta and Gamma.

* BarrierPathwise.py is as the above EuropeanBarrierAntithetic.py.py, but with pathwise Delta and Gamma. 

* LookBackPathwise.py is as the above LookBackAntithetic.py, but with pathwise Delta and Gamma.

* AsianPathwise.py is as the above AsianAntithetic.py, but with pathwise Delta and Gamma.



## Nsims_scaling folder

Contains scripts for calculating options prices and greeks using monte-carlo and comparing to the analytic formulas. Shows the convergence as N_simulations is increased. Contains separate folders for vanilla, barrier, lookback, and asian options. The resulting plots are saved in the plots folder.

## visualisations folder

Contains a script generating some geometric brownian motion paths for stock prices. Generates plots comparing the evolution under the objective and risk-neutral measures. 

## plots folder

Stores the generated plots. Contains separate folders for vanilla, barrier, lookback, and asian options.

*****************************************************
# Eventual Utility

These scripts can eventually be used for validating methods to be applied to exotics without analytic results. They could also be used to generate training data for neural network pricing of exotics. Similarly, they can be used to validate output of PDE based approaches for exotic pricing.

