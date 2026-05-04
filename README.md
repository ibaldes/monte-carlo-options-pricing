*****************************************************

Code for Monte-Carlo pricing of options assuming the Black-Scholes model and risk-neutral probability. Undergoing further improvements and development (American, Bermudan, Parisian, Asian, ... Options.)

*****************************************************
******************* Libraries Used ******************

- pandas
- numpy
- matplotlib 
- multiprocessing (to allow for quicker calculation over large n_simulations)
- scipy

*****************************************************
********************** SUMMARY **********************

* EuropeanVanilla.py contains functions for the following:
- Analytic results of Black Scholes Vanilla European options with or without Greeks (delta, gamma, vega, theta, rho).
- Monte-Carlo implementation Vanilla European options pricing with or without Greeks. These also return standard errors of the prices and greeks.

* EuropeanBarrier.py contains functions for the following:
- Analytic pricing for Knock-in/Knock-out European puts and calls (greeks: calculated using finite difference).
- Monte-Carlo pricing for Knock-in/Knock-out European puts and calls with Greeks. These also return standard errors of the prices and greeks. (To do: improve estimates of the greeks using more advanced techniques)

* PricingExample.py calls the functions from EuropeanVanilla.py and EuropeanBarrier.py showing some example prices.

* ScalingWithNsimulations_CallOption.py calls functions from EuropeanVanilla.py and shows the convergence of the Monte-Carlo calculation of the VANILLA call option price and greeks with the analytic result for large n_simulations. (To do: investigate behaviour of Gamma at small values of n_simulations.)

* ScalingWithNsimulations_PutOption.py calls functions from EuropeanVanilla.py and shows the convergence of the Monte-Carlo calculation of the VANILLA put option price and greeks with the analytic result for large n_simulations. (To do: investigate behaviour of Gamma at small values of n_simulations.)

* ScalingWithNsimulations_CallOption_Barrier.py calls functions from EuropeanBarrier.py and shows the convergence of the Monte-Carlo calculation of the BARRIER call option price and greeks with the analytic result for large n_simulations. (Note estimates of the Greeks are still to be improved).

* ScalingWithNsimulations_PutOption_Barrier.py calls functions from EuropeanBarrier.py and shows the convergence of the Monte-Carlo calculation of the BARRIER put option price and greeks with the analytic result for large n_simulations. (Note estimates of the Greeks are still to be improved).

The plots folder is where the generated plots are stored.

*****************************************************
***************** Eventual Utility ****************

These scripts can eventually be used for generating training data for neural network pricing of exotics. Similarly, they can be used to validate output of PDE based approaches for exotic pricing.

