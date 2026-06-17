import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

'''
PRICING OF EUROPEAN VANILLA OPTIONS USING MONTE-CARLO AS A CHECK OF OUR NUMERICAL METHODS
USES ANTITHETIC VARIATES FOR VARIANCE REDUCTION
USES PATHWISE METHOD FOR ALL THE MONTE-CARLO GREEKS (DELTA, GAMMA, VEGA, THETA, RHO).
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


#### Introduce smoothed functions for pathwise derivative

epsilon_benchmark = 0.1 	# small smoothing parameter

def Heaviside_smoothed(x, epsilon=epsilon_benchmark):
	'''
	Gives the smoothed Heaviside step function. Used for the pathwise derivative.
	Inputs:
	Heaviside Variable x 
	Small smoothing parameter epsilon (default 0.1)
	'''	
	return( 1/2*( np.tanh(x/epsilon) + 1 ) )

def R_epsilon(x, epsilon=epsilon_benchmark):
	''' 
	Integral over the smoothed Heaviside function
	Inputs:
	Variable x
	Small smoothing parameter epsilon (default 0.1)		
	'''
	result, estimated_error = quad( Heaviside_smoothed, -np.inf, x, args =(epsilon) )
	return(result)
	
def Heaviside_dx_smoothed(x, epsilon=epsilon_benchmark):
	'''
	Gives the derivative of the smoothed Heaviside step function with x.
	(Approximates the dirac delta).
	Used for the pathwise derivative.
	Inputs:
	Heaviside Variable x 
	Small smoothing parameter epsilon (default 0.1)
	'''	
	return( 1/(2*epsilon)*( 1-  (np.tanh(x/epsilon)**2 ) ) )

def Heaviside_dx2_smoothed(x, epsilon=epsilon_benchmark):
	'''
	Gives the second derivative of the smoothed Heaviside step function with x.
	(Derivative of the the dirac delta).
	Used for the pathwise derivative.
	Inputs:
	Heaviside Variable x 
	Small smoothing parameter epsilon (default 0.1)
	'''	
	return( -1/(epsilon**2)*np.tanh(x/epsilon)*( 1-  (np.tanh(x/epsilon)**2 ) ) )	

##########################


###########################################################################

#### define a function to find the Black-Scholes European call price and using the Monte-Carlo simulation ########################

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
	
	time_step = (T-t)/n_steps	
	
	#################################################################################
	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps+1))			### Generate a Brownian motion array - 1 additional step for when we calculate theta	
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array = np.exp(log_path_array[:, -2])	

	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array_AT = np.exp(log_path_array_AT[:, -2])
	
	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()	
	terminal_price_array_AT = terminal_price_array_AT.flatten()
	
	################### concatenate arrays #################			
	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT) )

	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######

	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(terminal_price_array-K, 0)
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) - Using Pathwise method #########
	Extra_smoothing = 0.3
	delta_array = np.exp(-r*(T-t))*Heaviside_smoothed(terminal_price_array-K, Extra_smoothing)*(terminal_price_array/S)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) - Using Pathwise method #####
	Extra_smoothing = 1.0
	gamma_array = np.exp(-r*(T-t))*Heaviside_dx_smoothed(terminal_price_array-K, Extra_smoothing)*(terminal_price_array/S)**2
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	Z_final_value_array = 1/(sigma*np.sqrt(T-t))*(np.log(terminal_price_array/S) - (r - 0.5*sigma**2)*(T-t)) #### Final Z ~ N(0,1) values needed for some of the pathwise derivatives
	
	Extra_smoothing = 0.1
	vega_array = np.exp(-r*(T-t))*Heaviside_smoothed(terminal_price_array-K, Extra_smoothing)*terminal_price_array*(-sigma*(T-t)+Z_final_value_array*np.sqrt(T-t))
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	Extra_smoothing = 0.1
	theta_array = np.exp(-r*(T-t))*Heaviside_smoothed(terminal_price_array - K, Extra_smoothing)*terminal_price_array*(-1)*( (r - 0.5*sigma**2) + 0.5*sigma*Z_final_value_array/np.sqrt(T-t) ) 
	
	theta_value_part = np.mean(theta_array)
	theta_value = -1*(r*option_value + theta_value_part)	#### remembering I am using an overall minus sign in the definition of theta.
	
	theta_StandardError = np.sqrt( r**2*option_value_StandardError**2 + stats.sem(theta_array)**2 )
	
	#### Calculate Rho = dV/dr (partial)   #########
	rho_array = np.exp(-r*(T-t))*(T-t)*terminal_price_array*Heaviside_smoothed(terminal_price_array-K, Extra_smoothing)
	rho_value_part = np.mean(rho_array)
	
	rho_value = -(T-t)*option_value + rho_value_part
	rho_StandardError = np.sqrt( (T-t)**2*option_value_StandardError**2 + stats.sem(rho_array)**2 )
	
	
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
	
	time_step = (T-t)/n_steps	
	
	#################################################################################
	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps+1))			### Generate a Brownian motion array - 1 additional step for when we calculate theta	
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array = np.exp(log_path_array[:, -2])
	
	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array_AT = np.exp(log_path_array_AT[:, -2])
	
	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()	
	terminal_price_array_AT = terminal_price_array_AT.flatten()

	################### concatenate arrays #################			
	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT) )
	
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
	#### Calculate Delta = dV/dS (partial) - Using Pathwise method #########
	Extra_smoothing = 0.3
	delta_array = np.exp(-r*(T-t))*(-1)*Heaviside_smoothed(K-terminal_price_array, Extra_smoothing)*(terminal_price_array/S)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) - Using Pathwise method #####
	Extra_smoothing = 1.0	
	gamma_array = np.exp(-r*(T-t))*(-1)**2*Heaviside_dx_smoothed(K-terminal_price_array, Extra_smoothing)*(terminal_price_array/S)**2
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	Z_final_value_array = 1/(sigma*np.sqrt(T-t))*(np.log(terminal_price_array/S) - (r - 0.5*sigma**2)*(T-t)) #### Final Z ~ N(0,1) values needed for some of the pathwise derivatives
	
	Extra_smoothing = 0.1
	vega_array = np.exp(-r*(T-t))*-Heaviside_smoothed(K-terminal_price_array, Extra_smoothing)*terminal_price_array*(-sigma*(T-t)+Z_final_value_array*np.sqrt(T-t))
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	Extra_smoothing = 0.1
	theta_array = np.exp(-r*(T-t))*-Heaviside_smoothed(K-terminal_price_array, Extra_smoothing)*terminal_price_array*(-1)*( (r - 0.5*sigma**2) + 0.5*sigma*Z_final_value_array/np.sqrt(T-t) ) 
	
	theta_value_part = np.mean(theta_array)
	theta_value = -1*(r*option_value + theta_value_part)	#### remembering I am using an overall minus sign in the definition of theta.
	
	theta_StandardError = np.sqrt( r**2*option_value_StandardError**2 + stats.sem(theta_array)**2 )
	
	#### Calculate Rho = dV/dr (partial)   #########
	Extra_smoothing = 0.1	
	rho_array = np.exp(-r*(T-t))*(T-t)*terminal_price_array*-Heaviside_smoothed(K-terminal_price_array, Extra_smoothing)
	rho_value_part = np.mean(rho_array)
	
	rho_value = -(T-t)*option_value + rho_value_part
	rho_StandardError = np.sqrt( (T-t)**2*option_value_StandardError**2 + stats.sem(rho_array)**2 )
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)

#################

def main():
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
	
