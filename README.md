Code for Monte-Carlo pricing of options assuming the Black-Scholes model and risk-neutral probability. Undergoing further improvements and development (American, Bermudan, Parisian, Asian, ... Options.)

******************* Libraries Used ******************

- pandas
- numpy
- matplotlib 
- multiprocessing (to allow for quicker calculation over large n_simulations)
- scipy

*****************************************************
********************** SUMMARY **********************

EuropeanVanilla.py contains functions for the following:
- Analytic results of Black Scholes Vanilla European options with or without Greeks (delta, gamma, vega, theta, rho).
- Monte-Carlo implementation Vanilla European options pricing with or without Greeks. These also return standard errors of the prices and greeks.

EuropeanBarrier.py contains functions for the following:
- Monte-Carlo pricing for Knock-in/Knock-out European put and calls with Greeks. These also return standard errors of the prices and greeks.

PricingExample.py calls the functions from EuropeanVanilla.py and EuropeanBarrier.py showing some example prices.

PricingExample.py calls the functions from EuropeanVanilla.py and EuropeanBarrier.py showing some example prices.

ScalingWithNsimulations.py calls functions from EuropeanVanilla.py and shows the convergence of the Monte-Carlo calculation with the analytic result with large n_simulations.

The plots folder is where the generated plots are stored.

***************** Eventual Utility ****************

These scripts can eventually be used for generating training data for neural network pricing of exotics. Similarly, they can be used to validate output of PDE based approaches for exotic pricing.

