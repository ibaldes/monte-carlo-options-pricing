import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
'''
GIVES ANALYTIC FUNCTIONS FOR BLACK SCHOLES VANILLA EUROPEAN OPTIONS AND WITH GREEKS
ALSO REPRODUCES THE PRICING USING MONTE-CARLO AS A CHECK OF OUR NUMERICAL METHODS
'''

StandardBaseSeed = 0

#######

def BlackScholesVanillaEuropeanCall(S,K,r,sigma,t,T):

	'''
	Calculates the Black Scholes Vanilla European Call price using the analytic formula

	S is the stock price at time t  in dollars
	K is the strike price in dollars
	r is the risk-free interest rate
	sigma is the annual volatility (100% volatility = 1)
	t is the time the option price is evaluated
	T is the expiration time in years
	
	Output is the Call price in dollars 
	'''
		
	d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	
	Callprice = S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2) 
	
	return(Callprice)

def BlackScholesVanillaEuropeanPut(S,K,r,sigma,t,T):

	'''
	Calculates the Black Scholes Vanilla European Put price using the analytic formula

	S is the stock price at time t  in dollars
	K is the strike price in dollars
	r is the risk-free interest rate
	sigma is the annual volatility (100% volatility = 1)
	t is the time the option price is evaluated
	T is the expiration time in years
	
	Output is the Put price in dollars 
	'''
	
	d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	
	Putprice = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1) 
	
	return(Putprice)

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



#### define a function to find the Black-Scholes European call price using the Monte-Carlo simulation ########################

def MonteCarloVanillaEuropeanCall(S, K, r, sigma, t, T, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

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
	standard error on the option value
	'''
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	time_step = (T-t)/n_steps

	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps))			### Generate a Brownian motion array
	
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry 
	
	#################################################################################
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array = np.exp(log_path_array[:, -1])
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(terminal_price_array-K, 0)
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	#### return the option value and standard error  ###
	return(option_value, option_value_StandardError)

###################################################################################################
	
def MonteCarloVanillaEuropeanPut(S, K, r, sigma, t, T, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculate the Black Scholes Vanilla European put using Monte-Carlo and the following arguments:
	
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
	standard error on the option value
	'''
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	time_step = (T-t)/n_steps	
	
	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps))			### Generate a Brownian motion array
	
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry 
	
	#################################################################################
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array = np.exp(log_path_array[:, -1])
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(K-terminal_price_array, 0)
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	
	#### return the option value and standard error ###
	return(option_value, option_value_StandardError)	
	
############################### Now include the Greeks as an output #################################

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
	
	time_step = (T-t)/n_steps	
	
	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps+1))			### Generate a Brownian motion array - 1 additional step for when we calculate theta
	
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
#	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry - not needed here 
	
	#################################################################################
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array = np.exp(log_path_array[:, -2])
	
	##### perturb in S for Delta and Gamma ####
	terminal_price_array_smaller_S = terminal_price_array*(S-0.01)/S
	terminal_price_array_larger_S = terminal_price_array*(S+0.01)/S 	
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 ) 	
	
	terminal_price_array_smaller_sigma = np.exp(log_path_array_smaller_sigma[:, -2])
	terminal_price_array_larger_sigma = np.exp(log_path_array_larger_sigma[:, -2])
	
	##### perturb in t for theta ##############
	
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	
	#### perturb in r for rho #################
	
	log_step_array_smaller_r = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array
	log_step_array_larger_r = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array	
	
	log_path_array_smaller_r = np.log(S) + np.cumsum( log_step_array_smaller_r, axis = 1 ) 
	log_path_array_larger_r = np.log(S) + np.cumsum( log_step_array_larger_r, axis = 1 ) 	
	
	terminal_price_array_smaller_r = np.exp(log_path_array_smaller_r[:, -2])
	terminal_price_array_larger_r = np.exp(log_path_array_larger_r[:, -2])	
	
	######################################################################
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
	
	payoff_array_smaller_sigma = np.maximum(terminal_price_array_smaller_sigma-K, 0)
	payoff_array_larger_sigma = np.maximum(terminal_price_array_larger_sigma-K, 0)
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	
	vega_value = np.mean(vega_array)
	
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	payoff_array_smaller_t = np.maximum(terminal_price_array_smaller_t-K, 0)
	payoff_array_larger_t = np.maximum(terminal_price_array_larger_t-K, 0)
	
	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	
	theta_value = np.mean(theta_array)
	
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########

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
	
	time_step = (T-t)/n_steps	
	
	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps+1))			### Generate a Brownian motion array - 1 additional step for when we calculate theta
	
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	
	#################################################################################
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array = np.exp(log_path_array[:, -2])
	
	##### perturb in S for Delta and Gamma ####
	terminal_price_array_smaller_S = terminal_price_array*(S-0.01)/S
	terminal_price_array_larger_S = terminal_price_array*(S+0.01)/S 	
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 ) 	
	
	terminal_price_array_smaller_sigma = np.exp(log_path_array_smaller_sigma[:, -2])
	terminal_price_array_larger_sigma = np.exp(log_path_array_larger_sigma[:, -2])
	
	##### perturb in t for theta ##############
	
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	
	#### perturb in r for rho #################
	
	log_step_array_smaller_r = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array
	log_step_array_larger_r = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array	
	
	log_path_array_smaller_r = np.log(S) + np.cumsum( log_step_array_smaller_r, axis = 1 ) 
	log_path_array_larger_r = np.log(S) + np.cumsum( log_step_array_larger_r, axis = 1 ) 	
	
	terminal_price_array_smaller_r = np.exp(log_path_array_smaller_r[:, -2])
	terminal_price_array_larger_r = np.exp(log_path_array_larger_r[:, -2])	
	
	
	###### CALCULATE THE OPTION PRICE ###################
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
	
	payoff_array_smaller_sigma = np.maximum(K-terminal_price_array_smaller_sigma, 0)
	payoff_array_larger_sigma = np.maximum(K-terminal_price_array_larger_sigma, 0)
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	
	vega_value = np.mean(vega_array)
	
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
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

	print('Call Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')
	print('\nEuropean Call Analytic\n', BlackScholesVanillaEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))
	print('\nEuropean Call Monte-Carlo\n', MonteCarloVanillaEuropeanCall(80, 85, 0.05, 0.4, 1, 1.25))	
	print('\nEuropean Call Monte-Carlo\n', MonteCarloVanillaEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))

	print('\n')

	print('Put Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')
	print('\nEuropean Put Analytic\n', BlackScholesVanillaEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))
	print('\nEuropean Put Monte-Carlo\n', MonteCarloVanillaEuropeanPut(80, 85, 0.05, 0.4, 1, 1.25))	
	print('\nEuropean Put Monte-Carlo\n', MonteCarloVanillaEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))
	
if __name__ == "__main__":
	main()	
	
