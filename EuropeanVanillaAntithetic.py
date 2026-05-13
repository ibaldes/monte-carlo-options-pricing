import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from scipy.stats import norm
from multiprocessing import Pool

'''
PRICING OF EUROPEAN VANILLA OPTIONS USING MONTE-CARLO AS A CHECK OF OUR NUMERICAL METHODS
USES ANTITHETIC VARIATES FOR VARIANCE REDUCTION
'''

StandardBaseSeed = 0

#####################

def BlackScholesVanillaEuropeanCallWithGreeks(S,K,r,sigma,t,T):

	'''
	Calculates the Black Scholes Vanilla European Call price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formulas

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	Output is the call price in dollars, Delta, Gamma, Vega, Theta, and Rho.
	
	Delta = dV/dS
	Gamma = d^2V/dS^2
	Vega = dV/dsigma
	Theta = -dV/dt
	Rho = dV/dr
	
	(All are understood to be partial derivatives)
	
	'''
		
	d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	
	Callprice = S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2)
	
	Delta = norm.cdf(d1)
	
	Gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T-t))
	
	Vega = S*norm.pdf(d1)*np.sqrt(T-t)
	
	# Theta is defined as -dV/dt #
	Theta = -1*(-S*norm.pdf(d1)*sigma/(2*np.sqrt(T-t)) - r*K*np.exp(-r*(T-t))*norm.cdf(d2))
	
	Rho = K*(T-t)*np.exp(-r*(T-t))*norm.cdf(d2)
	
	return(Callprice, Delta, Gamma, Vega, Theta, Rho)


def BlackScholesVanillaEuropeanPutWithGreeks(S,K,r,sigma,t,T):

	'''
	Calculates the Black Scholes Vanilla European Put price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formulas

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	Output is the put price in dollars, Delta, Gamma, Vega, Theta, and Rho.
	
	Delta = dV/dS
	Gamma = d^2V/dS^2
	Vega = dV/dsigma
	Theta = -dV/dt
	Rho = dV/dr
	
	(All are understood to be partial derivatives)
	
	'''
		
	d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	
	Putprice = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1) 
	
	Delta = norm.cdf(d1) - 1
	
	Gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T-t))
	
	Vega = S*norm.pdf(d1)*np.sqrt(T-t)
	
	# Theta is defined as -dV/dt #
	Theta = -1*(-S*norm.pdf(d1)*sigma/(2*np.sqrt(T-t)) + r*K*np.exp(-r*(T-t))*norm.cdf(-d2))
	
	Rho = -K*(T-t)*np.exp(-r*(T-t))*norm.cdf(-d2)
	
	return(Putprice, Delta, Gamma, Vega, Theta, Rho)

#######

##########################
##########################

def generate_terminal_price_ForGreeks_Antithetic(S, r, sigma, t, T, n_steps=100, seed=0):

	'''
	Generates terminal stock price, given evolution under the risk-neutral measure for the discounted stock price 
	Also generates finite differences to the terminal stock prices for changes to S, r, sigma, and t, so we can calculate the Greeks	
	
	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	n_steps is the number of steps to use (defaults to 100)
	seed is for the RNG
	
	Output is the terminal price and the terminal prices for the perturbed quantities, which will be used for the finite difference estimates of the partial derivatives.
	Output is doubled, with entries in the second half the antithetic pairs of the first.
	'''
	
	### force n_steps integer value ####
	n_steps = int(n_steps)
	
	### set the random seed #######
	rng = np.random.default_rng(seed)

	### calculate the required time step, unit is in years #####
	time_step = (T-t)/n_steps
	
	#################################################
	################################################
	
	### generate an array of brownian motions, one for each time step, given the time step and volatility	
	brownian_array = np.sqrt(time_step)*sigma*rng.normal(0, 1, size=n_steps)
	
	#### initialize the stock price array, the first entry is the price at t, the rest are initialized to zero ####
	stock_price_array = np.zeros(n_steps+1)
	stock_price_array[0] = S	
	
	#### fill in the stock price array, given the stock price on the previous day, and the brownian motion #####	
	for j in range(1,n_steps+1):
		stock_price_array[j] = stock_price_array[j-1]*(1+brownian_array[j-1]+r*time_step)
	
	### return the terminal stock price	
	terminal_price = stock_price_array[-1]
	
	### Finite differences for Delta and Gamma: start with 1 cent greater or smaller price ###
	terminal_price_smaller_S = terminal_price*(S-0.01)/S
	terminal_price_larger_S = terminal_price*(S+0.01)/S 

	### Finite differences for sigma: start with 1 per cent greater or smaller volatility ###
	brownian_array_smaller_sigma = brownian_array*(sigma-0.01)/sigma
	brownian_array_larger_sigma = brownian_array*(sigma+0.01)/sigma
	
	stock_price_array_smaller_sigma = np.zeros(n_steps+1)
	stock_price_array_larger_sigma = np.zeros(n_steps+1)
	stock_price_array_smaller_sigma[0] = S
	stock_price_array_larger_sigma[0] = S
	
	for j in range(1,n_steps+1):
		stock_price_array_smaller_sigma[j] = stock_price_array_smaller_sigma[j-1]*(1+brownian_array_smaller_sigma[j-1]+r*time_step)
		stock_price_array_larger_sigma[j] = stock_price_array_larger_sigma[j-1]*(1+brownian_array_larger_sigma[j-1]+r*time_step)
	
	terminal_price_smaller_sigma = stock_price_array_smaller_sigma[-1]
	terminal_price_larger_sigma = stock_price_array_larger_sigma[-1]
	
	### Finite differences for t: finish with plus/minus 1 time step differences #############
	
	addedjump_smaller_t = np.sqrt(time_step)*sigma*rng.normal(0, 1)
	
	terminal_price_smaller_t = stock_price_array[-1]*(1+addedjump_smaller_t+r*time_step)
	terminal_price_larger_t = stock_price_array[-2]

	### Finite differences for r: start with plus/minus 1 bps (1e-4 in r) difference ########
	
	stock_price_array_smaller_r = np.zeros(n_steps+1)
	stock_price_array_larger_r = np.zeros(n_steps+1)
	stock_price_array_smaller_r[0] = S
	stock_price_array_larger_r[0] = S
	
	for j in range(1,n_steps+1):
		stock_price_array_smaller_r[j] = stock_price_array_smaller_r[j-1]*(1+brownian_array[j-1]+(r-1e-4)*time_step)
		stock_price_array_larger_r[j] = stock_price_array_larger_r[j-1]*(1+brownian_array[j-1]+(r+1e-4)*time_step)
	
	terminal_price_smaller_r = stock_price_array_smaller_r[-1]
	terminal_price_larger_r = stock_price_array_larger_r[-1]
	
	#################################################
	######	Antithetic Pairs of the above ###########
	
	brownian_array_AT = -brownian_array

	stock_price_array_AT = np.zeros(n_steps+1)
	stock_price_array_AT[0] = S	
	
	for j in range(1,n_steps+1):
		stock_price_array_AT[j] = stock_price_array_AT[j-1]*(1+brownian_array_AT[j-1]+r*time_step)
	
	### return the terminal stock price	
	terminal_price_AT = stock_price_array_AT[-1]
	
	### Finite differences for Delta and Gamma: start with 1 cent greater or smaller price ###
	terminal_price_smaller_S_AT = terminal_price_AT*(S-0.01)/S
	terminal_price_larger_S_AT = terminal_price_AT*(S+0.01)/S 

	### Finite differences for sigma: start with 1 per cent greater or smaller volatility ###
	brownian_array_smaller_sigma_AT = brownian_array_AT*(sigma-0.01)/sigma
	brownian_array_larger_sigma_AT = brownian_array_AT*(sigma+0.01)/sigma
	
	stock_price_array_smaller_sigma_AT = np.zeros(n_steps+1)
	stock_price_array_larger_sigma_AT = np.zeros(n_steps+1)
	stock_price_array_smaller_sigma_AT[0] = S
	stock_price_array_larger_sigma_AT[0] = S
	
	for j in range(1,n_steps+1):
		stock_price_array_smaller_sigma_AT[j] = stock_price_array_smaller_sigma_AT[j-1]*(1+brownian_array_smaller_sigma_AT[j-1]+r*time_step)
		stock_price_array_larger_sigma_AT[j] = stock_price_array_larger_sigma_AT[j-1]*(1+brownian_array_larger_sigma_AT[j-1]+r*time_step)
	
	terminal_price_smaller_sigma_AT = stock_price_array_smaller_sigma_AT[-1]
	terminal_price_larger_sigma_AT = stock_price_array_larger_sigma_AT[-1]
	
	### Finite differences for t: finish with plus/minus 1 time step differences #############
	
	terminal_price_smaller_t_AT = stock_price_array_AT[-1]*(1-addedjump_smaller_t+r*time_step)
	terminal_price_larger_t_AT = stock_price_array_AT[-2]

	### Finite differences for r: start with plus/minus 1 bps (1e-4 in r) difference ########
	
	stock_price_array_smaller_r_AT = np.zeros(n_steps+1)
	stock_price_array_larger_r_AT = np.zeros(n_steps+1)
	stock_price_array_smaller_r_AT[0] = S
	stock_price_array_larger_r_AT[0] = S
	
	for j in range(1,n_steps+1):
		stock_price_array_smaller_r_AT[j] = stock_price_array_smaller_r_AT[j-1]*(1+brownian_array_AT[j-1]+(r-1e-4)*time_step)
		stock_price_array_larger_r_AT[j] = stock_price_array_larger_r_AT[j-1]*(1+brownian_array_AT[j-1]+(r+1e-4)*time_step)
	
	terminal_price_smaller_r_AT = stock_price_array_smaller_r_AT[-1]
	terminal_price_larger_r_AT = stock_price_array_larger_r_AT[-1]
	
	#######################################################
	#######################################################
	
	return(terminal_price, terminal_price_smaller_S, terminal_price_larger_S, terminal_price_smaller_sigma, terminal_price_larger_sigma, terminal_price_smaller_t, terminal_price_larger_t, terminal_price_smaller_r, terminal_price_larger_r, terminal_price_AT, terminal_price_smaller_S_AT, terminal_price_larger_S_AT, terminal_price_smaller_sigma_AT, terminal_price_larger_sigma_AT, terminal_price_smaller_t_AT, terminal_price_larger_t_AT, terminal_price_smaller_r_AT, terminal_price_larger_r_AT)


###########################################################################

#### define a function to find the Black-Scholes European call price and using the Monte-Carlo simulation ########################
#### uses generate_terminal_price_ForGreeks_Antithetic to find the final stock price and the payoff for the given number of simulations #####

def MonteCarloVanillaEuropeanCallWithGreeks(S, K, r, sigma, t, T, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculate the Black Scholes Vanilla European Call using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	
	OUTPUT:
	option value
	delta
	gamma
	vega
	theta 
	rho
	standard error of the option value
	standard error of delta
	standard error of gamma
	standard error of vega
	standard error of theta
	standard error of rho
	'''
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]

	
	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_Antithetic, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'terminal_price_AT', 'terminal_price_smaller_S_AT', 'terminal_price_larger_S_AT', 'terminal_price_smaller_sigma_AT', 'terminal_price_larger_sigma_AT', 'terminal_price_smaller_t_AT', 'terminal_price_larger_t_AT', 'terminal_price_smaller_r_AT', 'terminal_price_larger_r_AT'] )
	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = np.concatenate( ( full_terminal_price_df['terminal_price'].to_numpy(), full_terminal_price_df['terminal_price_AT'].to_numpy() ) )
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(2*n_simulations)
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(terminal_price_array-K, 0)
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = np.concatenate( ( full_terminal_price_df['terminal_price_smaller_S'].to_numpy(), full_terminal_price_df['terminal_price_smaller_S_AT'].to_numpy() ) )
	terminal_price_array_larger_S = np.concatenate( ( full_terminal_price_df['terminal_price_larger_S'].to_numpy(), full_terminal_price_df['terminal_price_larger_S_AT'].to_numpy() ) )
	
	payoff_array_smaller_S = np.maximum(terminal_price_array_smaller_S-K, 0)
	payoff_array_larger_S = np.maximum(terminal_price_array_larger_S-K, 0)
	
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	
	delta_value = np.mean(delta_array)
	
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(0.01**2)
	
	gamma_value = np.mean(gamma_array)
	
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = np.concatenate( ( full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy(), full_terminal_price_df['terminal_price_smaller_sigma_AT'].to_numpy() ))
	terminal_price_array_larger_sigma = np.concatenate( ( full_terminal_price_df['terminal_price_larger_sigma'].to_numpy(), full_terminal_price_df['terminal_price_larger_sigma_AT'].to_numpy() ))
	
	payoff_array_smaller_sigma = np.maximum(terminal_price_array_smaller_sigma-K, 0)
	payoff_array_larger_sigma = np.maximum(terminal_price_array_larger_sigma-K, 0)
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	
	vega_value = np.mean(vega_array)
	
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = np.concatenate( ( full_terminal_price_df['terminal_price_smaller_t'].to_numpy(), full_terminal_price_df['terminal_price_smaller_t_AT'].to_numpy() ))
	terminal_price_array_larger_t = np.concatenate( ( full_terminal_price_df['terminal_price_larger_t'].to_numpy(), full_terminal_price_df['terminal_price_larger_t_AT'].to_numpy() ))
	
	payoff_array_smaller_t = np.maximum(terminal_price_array_smaller_t-K, 0)
	payoff_array_larger_t = np.maximum(terminal_price_array_larger_t-K, 0)
	
	time_step = (T-t)/n_steps
	
	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	
	theta_value = np.mean(theta_array)
	
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	terminal_price_array_smaller_r = np.concatenate( ( full_terminal_price_df['terminal_price_smaller_r'].to_numpy(), full_terminal_price_df['terminal_price_smaller_r_AT'].to_numpy() ) )
	terminal_price_array_larger_r = np.concatenate( ( full_terminal_price_df['terminal_price_larger_r'].to_numpy(), full_terminal_price_df['terminal_price_larger_r_AT'].to_numpy()  ) )
	
	payoff_array_smaller_r = np.maximum(terminal_price_array_smaller_r-K, 0)
	payoff_array_larger_r = np.maximum(terminal_price_array_larger_r-K, 0)
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	
	rho_value = np.mean(rho_array)
	
	rho_StandardError = stats.sem(rho_array)
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)

	
#### define a function to find the Black-Scholes European put price using the Monte-Carlo simulation ########################
#### uses generate_terminal_price to find the final stock price and the payoff for the given number of simulations #####

def MonteCarloVanillaEuropeanPutWithGreeks(S, K, r, sigma, t, T, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculate the Black Scholes Vanilla European Put using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	
	OUTPUT:
	option value
	delta
	gamma
	vega
	theta 
	rho
	standard error of the option value
	standard error of delta
	standard error of gamma
	standard error of vega
	standard error of theta
	standard error of rho
	'''

	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	
	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_Antithetic, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'terminal_price_AT', 'terminal_price_smaller_S_AT', 'terminal_price_larger_S_AT', 'terminal_price_smaller_sigma_AT', 'terminal_price_larger_sigma_AT', 'terminal_price_smaller_t_AT', 'terminal_price_larger_t_AT', 'terminal_price_smaller_r_AT', 'terminal_price_larger_r_AT'] )
	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = np.concatenate( ( full_terminal_price_df['terminal_price'].to_numpy(), full_terminal_price_df['terminal_price_AT'].to_numpy() ) )
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(2*n_simulations)
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(K-terminal_price_array, 0)
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value 
	option_value_StandardError = stats.sem(option_value_array)

	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = np.concatenate( ( full_terminal_price_df['terminal_price_smaller_S'].to_numpy(), full_terminal_price_df['terminal_price_smaller_S_AT'].to_numpy() ) )
	terminal_price_array_larger_S = np.concatenate( ( full_terminal_price_df['terminal_price_larger_S'].to_numpy(), full_terminal_price_df['terminal_price_larger_S_AT'].to_numpy() ) )
	
	payoff_array_smaller_S = np.maximum(K-terminal_price_array_smaller_S, 0)
	payoff_array_larger_S = np.maximum(K-terminal_price_array_larger_S, 0)
	
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	
	delta_value = np.mean(delta_array)
	
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(0.01**2)
	
	gamma_value = np.mean(gamma_array)
	
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	terminal_price_array_smaller_sigma = np.concatenate( ( full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy(), full_terminal_price_df['terminal_price_smaller_sigma_AT'].to_numpy() ))
	terminal_price_array_larger_sigma = np.concatenate( ( full_terminal_price_df['terminal_price_larger_sigma'].to_numpy(), full_terminal_price_df['terminal_price_larger_sigma_AT'].to_numpy() ))
		
	payoff_array_smaller_sigma = np.maximum(K-terminal_price_array_smaller_sigma, 0)
	payoff_array_larger_sigma = np.maximum(K-terminal_price_array_larger_sigma, 0)
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	
	vega_value = np.mean(vega_array)
	
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	terminal_price_array_smaller_t = np.concatenate( ( full_terminal_price_df['terminal_price_smaller_t'].to_numpy(), full_terminal_price_df['terminal_price_smaller_t_AT'].to_numpy() ))
	terminal_price_array_larger_t = np.concatenate( ( full_terminal_price_df['terminal_price_larger_t'].to_numpy(), full_terminal_price_df['terminal_price_larger_t_AT'].to_numpy() ))

	
	payoff_array_smaller_t = np.maximum(K-terminal_price_array_smaller_t, 0)
	payoff_array_larger_t = np.maximum(K-terminal_price_array_larger_t, 0)
	
	time_step = (T-t)/n_steps

	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	
	theta_value = np.mean(theta_array)
	
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	terminal_price_array_smaller_r = np.concatenate( ( full_terminal_price_df['terminal_price_smaller_r'].to_numpy(), full_terminal_price_df['terminal_price_smaller_r_AT'].to_numpy() ) )
	terminal_price_array_larger_r = np.concatenate( ( full_terminal_price_df['terminal_price_larger_r'].to_numpy(), full_terminal_price_df['terminal_price_larger_r_AT'].to_numpy()  ) )
	
	payoff_array_smaller_r = np.maximum(K-terminal_price_array_smaller_r, 0)
	payoff_array_larger_r = np.maximum(K-terminal_price_array_larger_r, 0)
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	
	rho_value = np.mean(rho_array)
	
	rho_StandardError = stats.sem(rho_array)
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)

#################

def main():
	print("Number of cpus : ", mp.cpu_count())
	pool = Pool(processes=(mp.cpu_count() - 1))

	print('\n')

	print('Call Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')
	print('\nEuropean Call Analytic\n', BlackScholesVanillaEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))
	print('\nEuropean Call Monte-Carlo\n', MonteCarloVanillaEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))

	print('\n')

	print('Put Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')
	print('\nEuropean Put Analytic\n', BlackScholesVanillaEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))
	print('\nEuropean Put Monte-Carlo\n', MonteCarloVanillaEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))
	
if __name__ == "__main__":
	main()	
	
