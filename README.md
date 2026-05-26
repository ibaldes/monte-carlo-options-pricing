*****************************************************

Code for Monte-Carlo pricing of options assuming the Black-Scholes model and risk-neutral probability. Undergoing further improvements and development (American, Bermudan, Parisian, Asian, ... Options.)

*****************************************************
# Libraries Used 

- sys
- os
- pandas
- numpy
- matplotlib 
- multiprocessing (to allow for quicker calculation over large n_simulations)
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

* PricingExample.py calls the functions from EuropeanVanilla.py and EuropeanBarrier.py showing some example prices.

## options_antithetic folder
Similar to options_benchmark, but uses antithetic variates. The first-half batch of simulations draws from the normal distribution as before. The second-half batch of simulations uses the antithetic variates (i.e. reverses the sign of the brownian motion of the first-half batch).

The scripts are:

* EuropeanVanillaAntithetic.py is as the above EuropeanVanilla, but with Antithetic Variates for the Monte-Carlo.

* EuropeanBarrierAntithetic.py is as the above EuropeanBarrier, but with Antithetic Variates for the Monte-Carlo.

* LookBackAntithetic.py is as the above LookBack, but with Antithetic Variates for the Monte-Carlo.


## Nsims_scaling folder

* ScalingWithNsimulations_CallOption.py calls functions from EuropeanVanilla.py and EuropeanVanillaAntithetic.py and shows the convergence of the Monte-Carlo calculation of the VANILLA call option price and greeks with the analytic result for large n_simulations. 

* ScalingWithNsimulations_PutOption.py calls functions from EuropeanVanilla.py and EuropeanVanillaAntithetic.py  and shows the convergence of the Monte-Carlo calculation of the VANILLA put option price and greeks with the analytic result for large n_simulations.

* ScalingWithNsimulations_CallOption_KnockInBarrier.py calls functions from EuropeanBarrier.py and EuropeanBarrierAntithetic.py and shows the convergence of the Monte-Carlo calculation of the KNOCK IN BARRIER call option price and greeks with the analytic result for large n_simulations. (Note estimates of the Greeks, particularly Gamma, are still to be improved).

* ScalingWithNsimulations_CallOption_KnockOutBarrier.py calls functions from EuropeanBarrier.py and EuropeanBarrierAntithetic.py and shows the convergence of the Monte-Carlo calculation of the KNOCK OUT BARRIER call option price and greeks with the analytic result for large n_simulations.

* ScalingWithNsimulations_PutOption_KnockInBarrier.py calls functions from EuropeanBarrier.py and EuropeanBarrierAntithetic.py and shows the convergence of the Monte-Carlo calculation of the KNOCK IN BARRIER put option price and greeks with the analytic result for large n_simulations. 

* ScalingWithNsimulations_PutOption_KnockOutBarrier.py calls functions from EuropeanBarrier.py and EuropeanBarrierAntithetic.py and shows the convergence of the Monte-Carlo calculation of the KNOCK OUT BARRIER put option price and greeks with the analytic result for large n_simulations. 

* ScalingWithNsimulations_LookBackCall_FloatingStrike.py calls functions from LookBack.py and LookBackAntithetic.py  and shows the convergence of the Monte-Carlo calculation of the Look Back Floating strike call option price with the analytic result. 

* ScalingWithNsimulations_LookBackPut_FloatingStrike.py calls functions from LookBack.py and LookBackAntithetic.py and shows the convergence of the Monte-Carlo calculation of the Look Back Floating strike put option price with the analytic result. 

* ScalingWithNsimulations_LookBackCall_FixedStrike.py calls functions from LookBack.py and LookBackAntithetic.py and shows the convergence of the Monte-Carlo calculation of the Look Back fixed strike call option price with the analytic result. 

* ScalingWithNsimulations_LookBackCall_FixedStrike.py calls functions from LookBack.py and LookBackAntithetic.py  and shows the convergence of the Monte-Carlo calculation of the Look Back fixed strike call option price with the analytic result. 

## visualisations folder

Contains a script generating some geometric brownian motion paths for stock prices. Generates plots comparing the evolution under the objective and risk-neutral measures. 

## plots folder

Stores the generated plots.

*****************************************************
# Eventual Utility

These scripts can eventually be used for generating training data for neural network pricing of exotics. Similarly, they can be used to validate output of PDE based approaches for exotic pricing.

