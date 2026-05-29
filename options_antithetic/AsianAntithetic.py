import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

###

'''
GIVES ANALYTIC AND MONTE-CARLO IMPLEMENTATION OF ASIAN OPTIONS
RETURNS THE OPTION PRICE, GREEKS, AND STANDARD ERRORS
'''

StandardBaseSeed = 0


##################################################################################################################################################

def ApproxAvgPriceCall(S,K,r,sigma,t,T,Savgsofar=None):	
	'''
	Approximate analytic formula for the Asian average price call.
	
	Note: The Payoff is given by Max[Savg - K,0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
	K is the strike price	
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savg is the average price between time 0 and time t
	
	Output is the call price in dollars
	'''
	#################################################################
	#################################################################
	
	####### Implement formula from Hull, 11ed., Chapter 26, for the continuos case ########
	M1 = ( np.exp(r*(T-t))-1 )/( r*(T-t) )*S 
	M2 = 2*np.exp( (2*r + sigma**2)*(T-t) )*S**2 / (  (r + sigma**2)*(2*r + sigma**2)*(T-t)**2 ) + 2*S**2/( r*(T-t)**2 )*( 1/(2*r+sigma**2) - np.exp( r*(T-t))/(r+sigma**2) )
	sigmanewsq = 1/(T-t)*np.log(M2/M1**2)
	F0 = M1 
	d1 = ( np.log(F0/K) + 0.5*sigmanewsq*(T-t) )/( np.sqrt(sigmanewsq)*np.sqrt(T-t) )
	d2 = d1 - np.sqrt(sigmanewsq)*np.sqrt(T-t)
	c = np.exp(-r*(T-t))*(F0*norm.cdf(d1) - K*norm.cdf(d2))
	p = np.exp(-r*(T-t))*(K*norm.cdf(-d2) - F0*norm.cdf(-d1))
	
	AvgPriceCall = c

	return(AvgPriceCall)


##################################################################################################################################################

def ApproxAvgPricePut(S,K,r,sigma,t,T,Savgsofar=None):
	'''
	Approximate analytic formula for the Asian put.
	
	Note: The Payoff is given by Max[K-Savg,0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
	K is the strike price	
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savg is the average price between time 0 and time t
	
	Output is the call price in dollars
	'''
	#################################################################
	#################################################################
	
	####### Implement formula from Hull, 11ed., Chapter 26, for the continuos case ########
	M1 = ( np.exp(r*(T-t))-1 )/( r*(T-t) )*S 
	M2 = 2*np.exp( (2*r + sigma**2)*(T-t) )*S**2 / (  (r + sigma**2)*(2*r + sigma**2)*(T-t)**2 ) + 2*S**2/( r*(T-t)**2 )*( 1/(2*r+sigma**2) - np.exp( r*(T-t))/(r+sigma**2) )
	sigmanewsq = 1/(T-t)*np.log(M2/M1**2)
	F0 = M1 
	d1 = ( np.log(F0/K) + 0.5*sigmanewsq*(T-t) )/( np.sqrt(sigmanewsq)*np.sqrt(T-t) )
	d2 = d1 - np.sqrt(sigmanewsq)*np.sqrt(T-t)
	c = np.exp(-r*(T-t))*(F0*norm.cdf(d1) - K*norm.cdf(d2))
	p = np.exp(-r*(T-t))*(K*norm.cdf(-d2) - F0*norm.cdf(-d1))
	
	AvgPricePut = p

	return(AvgPricePut)		
 
##################################################################################################################################################

def ApproxAvgPriceCallWithGreeks(S,K,r,sigma,t,T,Savgsofar=None):	
	'''
	Approximate analytic formula for the Asian average price call.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method
	Note: The Payoff is given by Max[Savg - K,0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
	K is the strike price	
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savg is the average price between time 0 and time t
		
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	##########################
	##########################
	
	
	#### Call Price Using Analytic Formula #######
	CallPrice = ApproxAvgPriceCall(S,K,r,sigma,t,T,Savgsofar)
	
	small_time_step = (T-t)/100

	### The Greeks ####
	Delta = ( ApproxAvgPriceCall(S+0.01,K,r,sigma,t,T,Savgsofar) - ApproxAvgPriceCall(S-0.01,K,r,sigma,t,T,Savgsofar) ) / (2*0.01)
	Gamma = ( ApproxAvgPriceCall(S+0.01,K,r,sigma,t,T,Savgsofar) - 2*ApproxAvgPriceCall(S,K,r,sigma,t,T,Savgsofar) + ApproxAvgPriceCall(S-0.01,K,r,sigma,t,T,Savgsofar))/(0.01**2)
	Vega = ( ApproxAvgPriceCall(S,K,r,sigma+0.01,t,T,Savgsofar) - ApproxAvgPriceCall(S,K,r,sigma-0.01,t,T,Savgsofar) ) / (2*0.01)
	Theta = -( ApproxAvgPriceCall(S,K,r,sigma,t+small_time_step,T,Savgsofar) - ApproxAvgPriceCall(S,K,r,sigma,t-small_time_step,T,Savgsofar) ) / (2*small_time_step)
	Rho = ( ApproxAvgPriceCall(S,K,r+1e-4,sigma,t,T,Savgsofar) - ApproxAvgPriceCall(S,K,r-1e-4,sigma,t,T,Savgsofar) ) / (2*1e-4)
	
	return(CallPrice, Delta, Gamma, Vega, Theta, Rho)

##################################################################################################################################################

def ApproxAvgPricePutWithGreeks(S,K,r,sigma,t,T,Savgsofar=None):	
	'''
	Approximate analytic formula for the Asian average price put.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method
	Note: The Payoff is given by Max[K - Savg ,0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savg is the average price between time 0 and time t
		
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	##########################
	##########################
	
	
	#### Put Price Using Analytic Formula #######
	PutPrice = ApproxAvgPricePut(S,K,r,sigma,t,T,Savgsofar)
	
	small_time_step = (T-t)/100

	### The Greeks ####
	Delta = ( ApproxAvgPricePut(S+0.01,K,r,sigma,t,T,Savgsofar) - ApproxAvgPricePut(S-0.01,K,r,sigma,t,T,Savgsofar) ) / (2*0.01)
	Gamma = ( ApproxAvgPricePut(S+0.01,K,r,sigma,t,T,Savgsofar) - 2*ApproxAvgPricePut(S,K,r,sigma,t,T,Savgsofar) + ApproxAvgPricePut(S-0.01,K,r,sigma,t,T,Savgsofar))/(0.01**2)
	Vega = ( ApproxAvgPricePut(S,K,r,sigma+0.01,t,T,Savgsofar) - ApproxAvgPricePut(S,K,r,sigma-0.01,t,T,Savgsofar) ) / (2*0.01)
	Theta = -( ApproxAvgPricePut(S,K,r,sigma,t+small_time_step,T,Savgsofar) - ApproxAvgPricePut(S,K,r,sigma,t-small_time_step,T,Savgsofar) ) / (2*small_time_step)
	Rho = ( ApproxAvgPricePut(S,K,r+1e-4,sigma,t,T,Savgsofar) - ApproxAvgPricePut(S,K,r-1e-4,sigma,t,T,Savgsofar) ) / (2*1e-4)
	
	return(PutPrice, Delta, Gamma, Vega, Theta, Rho)

##################################################################################################################################################

def ApproxAvgStrikeCall(S,r,sigma,t,T,Savgsofar=None):
	'''
	Approximate analytic formula for the Asian average strike call. Turnbull–Wakeman Approximation and Margrabe's formula.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method
	Note: The Payoff is given by Max[S(T) - Savg,0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savg is the average price between time 0 and time t
		
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''

	M1 = ( np.exp(r*(T-t))-1 )/( r*(T-t) )*S 
	M2 = 2*np.exp( (2*r + sigma**2)*(T-t) )*S**2 / (  (r + sigma**2)*(2*r + sigma**2)*(T-t)**2 ) + 2*S**2/( r*(T-t)**2 )*( 1/(2*r+sigma**2) - np.exp( r*(T-t))/(r+sigma**2) )
	sigmanewsq = 1/(T-t)*np.log(M2/M1**2)
	F0 = M1
	sigmaexchangesq = sigmanewsq + sigma**2 - np.sqrt(3)*np.sqrt(sigmanewsq)*sigma
	
	d1 = ( np.log( S*np.exp(r*(T-t)) / F0  ) + 0.5*sigmaexchangesq*(T-t) )/( np.sqrt(sigmaexchangesq)*np.sqrt(T-t) )
	d2 = d1 - np.sqrt(sigmaexchangesq)*np.sqrt(T-t) 
	
	
	CallPrice = np.exp(-r*(T-t))*( S*np.exp(r*(T-t))*norm.cdf(d1) - F0*norm.cdf(d2) )
	
	return(CallPrice)
	
	
	

##################################################################################################################################################

def ApproxAvgStrikePut(S,r,sigma,t,T,Savgsofar=None):
	'''
	Approximate analytic formula for the Asian average strike put. Turnbull–Wakeman Approximation and Margrabe's formula.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method.
	Note: The Payoff is given by Max[Savg - S(T),0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savg is the average price between time 0 and time t
		
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	
	M1 = ( np.exp(r*(T-t))-1 )/( r*(T-t) )*S 
	M2 = 2*np.exp( (2*r + sigma**2)*(T-t) )*S**2 / (  (r + sigma**2)*(2*r + sigma**2)*(T-t)**2 ) + 2*S**2/( r*(T-t)**2 )*( 1/(2*r+sigma**2) - np.exp( r*(T-t))/(r+sigma**2) )
	sigmanewsq = 1/(T-t)*np.log(M2/M1**2)
	F0 = M1
	sigmaexchangesq = sigmanewsq + sigma**2 - np.sqrt(3)*np.sqrt(sigmanewsq)*sigma
	
	d1 = ( np.log( F0 / (S*np.exp(r*(T-t)) ) ) + 0.5*sigmaexchangesq*(T-t) )/( np.sqrt(sigmaexchangesq)*np.sqrt(T-t) )
	d2 = d1 - np.sqrt(sigmaexchangesq)*np.sqrt(T-t) 
	
	PutPrice = np.exp(-r*(T-t))*( F0*norm.cdf(d1) - S*np.exp(r*(T-t))*norm.cdf(d2) )	
	
	return(PutPrice)

##################################################################################################################################################

def ApproxAvgStrikeCallWithGreeks(S,r,sigma,t,T,Savgsofar=None):
	'''
	Approximate analytic formula for the Asian average strike call. Turnbull–Wakeman Approximation and Margrabe's formula.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method
	Note: The Payoff is given by Max[Savg - K,0], where Savg is the arithmetic average price of the underlying. 

	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savg is the average price between time 0 and time t
		
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	
	CallPrice = ApproxAvgStrikeCall(S,r,sigma,t,T,Savgsofar)
	
	small_time_step = (T-t)/100

	### The Greeks ####
	Delta = ( ApproxAvgStrikeCall(S+0.01,r,sigma,t,T,Savgsofar) - ApproxAvgStrikeCall(S-0.01,r,sigma,t,T,Savgsofar) ) / (2*0.01)
	Gamma = ( ApproxAvgStrikeCall(S+0.01,r,sigma,t,T,Savgsofar) - 2*ApproxAvgStrikeCall(S,r,sigma,t,T,Savgsofar) + ApproxAvgStrikeCall(S-0.01,r,sigma,t,T,Savgsofar))/(0.01**2)
	Vega = ( ApproxAvgStrikeCall(S,r,sigma+0.01,t,T,Savgsofar) - ApproxAvgStrikeCall(S,r,sigma-0.01,t,T,Savgsofar) ) / (2*0.01)
	Theta = -( ApproxAvgStrikeCall(S,r,sigma,t+small_time_step,T,Savgsofar) - ApproxAvgStrikeCall(S,r,sigma,t-small_time_step,T,Savgsofar) ) / (2*small_time_step)
	Rho = ( ApproxAvgStrikeCall(S,r+1e-4,sigma,t,T,Savgsofar) - ApproxAvgStrikeCall(S,r-1e-4,sigma,t,T,Savgsofar) ) / (2*1e-4)
	
	return(CallPrice, Delta, Gamma, Vega, Theta, Rho)		


##################################################################################################################################################

def ApproxAvgStrikePutWithGreeks(S,r,sigma,t,T,Savgsofar=None):
	'''
	Approximate analytic formula for the Asian average strike put. Turnbull–Wakeman Approximation and Margrabe's formula.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method.
	Note: The Payoff is given by Max[Savg - S(T),0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savg is the average price between time 0 and time t
		
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	
	PutPrice = ApproxAvgStrikePut(S,r,sigma,t,T,Savgsofar)
	
	small_time_step = (T-t)/100

	### The Greeks ####
	Delta = ( ApproxAvgStrikePut(S+0.01,r,sigma,t,T,Savgsofar) - ApproxAvgStrikePut(S-0.01,r,sigma,t,T,Savgsofar) ) / (2*0.01)
	Gamma = ( ApproxAvgStrikePut(S+0.01,r,sigma,t,T,Savgsofar) - 2*ApproxAvgStrikePut(S,r,sigma,t,T,Savgsofar) + ApproxAvgStrikePut(S-0.01,r,sigma,t,T,Savgsofar))/(0.01**2)
	Vega = ( ApproxAvgStrikePut(S,r,sigma+0.01,t,T,Savgsofar) - ApproxAvgStrikePut(S,r,sigma-0.01,t,T,Savgsofar) ) / (2*0.01)
	Theta = -( ApproxAvgStrikePut(S,r,sigma,t+small_time_step,T,Savgsofar) - ApproxAvgStrikePut(S,r,sigma,t-small_time_step,T,Savgsofar) ) / (2*small_time_step)
	Rho = ( ApproxAvgStrikePut(S,r+1e-4,sigma,t,T,Savgsofar) - ApproxAvgStrikePut(S,r-1e-4,sigma,t,T,Savgsofar) ) / (2*1e-4)
	
	return(PutPrice, Delta, Gamma, Vega, Theta, Rho)	


##################################################################
##################### MONTE-CARLO ################################


def MonteCarloAvgPriceCallWithGreeks(S, K, r, sigma, t, T, Smintodate=None, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculates the Asian average price call using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	K is the strike price.
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savgsofar is the average price from time 0 to time t [default is none, e.g. appropriate when t = 0, just as the option comes into existence]
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	BaseSeed is the starting seed for the Seeds input into the Monte-Carlo
	
	Note: The Payoff is given by Max[Savg - K,0], where Savg is the arithmetic average price of the underlying.
	
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
	###########################
	###########################
	
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
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 )
	
	log_path_array_smaller_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_sigma ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_sigma ), axis=1 ) ### add the initial price as the first entry 	 		 	 	
	
	#### perturb in r for rho #################
	log_step_array_smaller_r = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array
	log_step_array_larger_r = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array	
	
	log_path_array_smaller_r = np.log(S) + np.cumsum( log_step_array_smaller_r, axis = 1 ) 
	log_path_array_larger_r = np.log(S) + np.cumsum( log_step_array_larger_r, axis = 1 )

	log_path_array_smaller_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_r ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_r ), axis=1 ) ### add the initial price as the first entry 		

	############ Average prices ###########################
	AvgPrice_array = np.mean( np.exp( log_path_array[:, :-1] ) , axis=1, keepdims=True) ### ignore the last column as this is the additional time step for theta calculation
	
	AvgPrice_array_smaller_S = AvgPrice_array*(S-0.01)/S
	AvgPrice_array_larger_S = AvgPrice_array*(S+0.01)/S
	
	AvgPrice_array_smaller_sigma = np.mean( np.exp( log_path_array_smaller_sigma[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_sigma =  np.mean( np.exp( log_path_array_larger_sigma[:, :-1] ) , axis=1, keepdims=True)
	
	AvgPrice_array_smaller_t = np.mean( np.exp( log_path_array ) , axis=1, keepdims=True) ### keep the last column as this is the additional time step for a longer time to maturity
	AvgPrice_array_larger_t = np.mean( np.exp( log_path_array[:, :-2] ) , axis=1, keepdims=True)  ### ignore the last two columns as the time to maturity is shorter
	
	AvgPrice_array_smaller_r = np.mean( np.exp( log_path_array_smaller_r[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_r = np.mean( np.exp( log_path_array_larger_r[:, :-1] ) , axis=1, keepdims=True)

	##### flatten arrays #####
	AvgPrice_array = AvgPrice_array.flatten()
	AvgPrice_array_smaller_S = AvgPrice_array_smaller_S.flatten()
	AvgPrice_array_larger_S = AvgPrice_array_larger_S.flatten() 	
	AvgPrice_array_smaller_sigma = AvgPrice_array_smaller_sigma.flatten()
	AvgPrice_array_larger_sigma = AvgPrice_array_larger_sigma.flatten()
	AvgPrice_array_smaller_t = AvgPrice_array_smaller_t.flatten()
	AvgPrice_array_larger_t = AvgPrice_array_larger_t.flatten()
	AvgPrice_array_smaller_r = AvgPrice_array_smaller_r.flatten()
	AvgPrice_array_larger_r = AvgPrice_array_larger_r.flatten()	

	###############################################################
	
	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array	
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	log_path_array_AT = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_AT ), axis=1 ) ### add the initial price as the first entry
		
	#################################################################################
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma_AT = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array_AT
	log_step_array_larger_sigma_AT = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array_AT	
	
	log_path_array_smaller_sigma_AT = np.log(S) + np.cumsum( log_step_array_smaller_sigma_AT, axis = 1 ) 
	log_path_array_larger_sigma_AT = np.log(S) + np.cumsum( log_step_array_larger_sigma_AT, axis = 1 )
	
	log_path_array_smaller_sigma_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_smaller_sigma_AT), axis=1 ) # add the initial price as the first entry
	log_path_array_larger_sigma_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_larger_sigma_AT), axis=1 ) # add the initial price as the first entry 	 		 	 	
	
	#### perturb in r for rho #################
	log_step_array_smaller_r_AT = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT
	log_step_array_larger_r_AT = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT	
	
	log_path_array_smaller_r_AT = np.log(S) + np.cumsum( log_step_array_smaller_r_AT, axis = 1 ) 
	log_path_array_larger_r_AT = np.log(S) + np.cumsum( log_step_array_larger_r_AT, axis = 1 )

	log_path_array_smaller_r_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_smaller_r_AT ), axis=1 ) # add the initial price as the first entry
	log_path_array_larger_r_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_larger_r_AT ), axis=1 ) # add the initial price as the first entry 		

	############ Average prices ###########################
	AvgPrice_array_AT = np.mean( np.exp( log_path_array_AT[:, :-1] ) , axis=1, keepdims=True) ### ignore the last column as this is the additional time step for theta calculation
	
	AvgPrice_array_smaller_S_AT = AvgPrice_array_AT*(S-0.01)/S
	AvgPrice_array_larger_S_AT = AvgPrice_array_AT*(S+0.01)/S
	
	AvgPrice_array_smaller_sigma_AT = np.mean( np.exp( log_path_array_smaller_sigma_AT[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_sigma_AT =  np.mean( np.exp( log_path_array_larger_sigma_AT[:, :-1] ) , axis=1, keepdims=True)
	
	AvgPrice_array_smaller_t_AT = np.mean( np.exp( log_path_array_AT ) , axis=1, keepdims=True) ### keep the last column as this is the additional time step for a longer time to maturity
	AvgPrice_array_larger_t_AT = np.mean( np.exp( log_path_array_AT[:, :-2] ) , axis=1, keepdims=True)  ### ignore the last two columns as the time to maturity is shorter
	
	AvgPrice_array_smaller_r_AT = np.mean( np.exp( log_path_array_smaller_r_AT[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_r_AT = np.mean( np.exp( log_path_array_larger_r_AT[:, :-1] ) , axis=1, keepdims=True)

	##### flatten arrays #####
	AvgPrice_array_AT = AvgPrice_array_AT.flatten()
	AvgPrice_array_smaller_S_AT = AvgPrice_array_smaller_S_AT.flatten()
	AvgPrice_array_larger_S_AT = AvgPrice_array_larger_S_AT.flatten() 	
	AvgPrice_array_smaller_sigma_AT = AvgPrice_array_smaller_sigma_AT.flatten()
	AvgPrice_array_larger_sigma_AT = AvgPrice_array_larger_sigma_AT.flatten()
	AvgPrice_array_smaller_t_AT = AvgPrice_array_smaller_t_AT.flatten()
	AvgPrice_array_larger_t_AT = AvgPrice_array_larger_t_AT.flatten()
	AvgPrice_array_smaller_r_AT = AvgPrice_array_smaller_r_AT.flatten()
	AvgPrice_array_larger_r_AT = AvgPrice_array_larger_r_AT.flatten()
	
	############## Concatenate #########################
	AvgPrice_array = np.concatenate( (AvgPrice_array, AvgPrice_array_AT) )
	AvgPrice_array_smaller_S = np.concatenate( (AvgPrice_array_smaller_S, AvgPrice_array_smaller_S_AT) )
	AvgPrice_array_larger_S = np.concatenate( (AvgPrice_array_larger_S, AvgPrice_array_larger_S_AT) ) 	
	AvgPrice_array_smaller_sigma = np.concatenate( (AvgPrice_array_smaller_sigma, AvgPrice_array_smaller_sigma_AT) )
	AvgPrice_array_larger_sigma = np.concatenate( (AvgPrice_array_larger_sigma, AvgPrice_array_larger_sigma_AT) )
	AvgPrice_array_smaller_t = np.concatenate( (AvgPrice_array_smaller_t, AvgPrice_array_smaller_t_AT) )
	AvgPrice_array_larger_t = np.concatenate( (AvgPrice_array_larger_t, AvgPrice_array_larger_t_AT) )
	AvgPrice_array_smaller_r = np.concatenate( (AvgPrice_array_smaller_r, AvgPrice_array_smaller_r_AT) )
	AvgPrice_array_larger_r = np.concatenate( (AvgPrice_array_larger_r, AvgPrice_array_larger_r_AT) )
	
	###### CALCULATE THE OPTION PRICE ###################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( AvgPrice_array - K, 0 )
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	payoff_array_smaller_S = np.maximum( AvgPrice_array_smaller_S - K, 0 )
	payoff_array_larger_S = np.maximum( AvgPrice_array_larger_S - K, 0 )
	
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
	payoff_array_smaller_sigma =  np.maximum( AvgPrice_array_smaller_sigma - K, 0 )
	payoff_array_larger_sigma = np.maximum( AvgPrice_array_larger_sigma - K, 0 )
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	payoff_array_smaller_t =  np.maximum( AvgPrice_array_smaller_t - K, 0 )
	payoff_array_larger_t = np.maximum( AvgPrice_array_larger_t - K, 0 )

	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t	# Think if additional correction should be taken into account to implement partial here
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t	# Think if additional correction should be taken into account to implement partial here	
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	payoff_array_smaller_r =  np.maximum( AvgPrice_array_smaller_r - K, 0 )
	payoff_array_larger_r =  np.maximum( AvgPrice_array_larger_r - K, 0 )
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	rho_value = np.mean(rho_array)
	rho_StandardError = stats.sem(rho_array)	
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)


##################################################################		


def MonteCarloAvgPricePutWithGreeks(S, K, r, sigma, t, T, Smintodate=None, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculates the Asian average price put using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	K is the strike price.
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savgsofar is the average price from time 0 to time t [default is none, e.g. appropriate when t = 0, just as the option comes into existence]
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	BaseSeed is the starting seed for the Seeds input into the Monte-Carlo
	
	Note: The Payoff is given by Max[ K - Savg ,0], where Savg is the arithmetic average price of the underlying.
	
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
	###########################
	###########################
	
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
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 )
	
	log_path_array_smaller_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_sigma ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_sigma ), axis=1 ) ### add the initial price as the first entry 	 		 	 	
	
	#### perturb in r for rho #################
	log_step_array_smaller_r = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array
	log_step_array_larger_r = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array	
	
	log_path_array_smaller_r = np.log(S) + np.cumsum( log_step_array_smaller_r, axis = 1 ) 
	log_path_array_larger_r = np.log(S) + np.cumsum( log_step_array_larger_r, axis = 1 )

	log_path_array_smaller_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_r ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_r ), axis=1 ) ### add the initial price as the first entry 		

	############ Average prices ###########################
	AvgPrice_array = np.mean( np.exp( log_path_array[:, :-1] ) , axis=1, keepdims=True) ### ignore the last column as this is the additional time step for theta calculation
	
	AvgPrice_array_smaller_S = AvgPrice_array*(S-0.01)/S
	AvgPrice_array_larger_S = AvgPrice_array*(S+0.01)/S
	
	AvgPrice_array_smaller_sigma = np.mean( np.exp( log_path_array_smaller_sigma[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_sigma =  np.mean( np.exp( log_path_array_larger_sigma[:, :-1] ) , axis=1, keepdims=True)
	
	AvgPrice_array_smaller_t = np.mean( np.exp( log_path_array ) , axis=1, keepdims=True) ### keep the last column as this is the additional time step for a longer time to maturity
	AvgPrice_array_larger_t = np.mean( np.exp( log_path_array[:, :-2] ) , axis=1, keepdims=True)  ### ignore the last two columns as the time to maturity is shorter
	
	AvgPrice_array_smaller_r = np.mean( np.exp( log_path_array_smaller_r[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_r = np.mean( np.exp( log_path_array_larger_r[:, :-1] ) , axis=1, keepdims=True)

	##### flatten arrays #####
	AvgPrice_array = AvgPrice_array.flatten()
	AvgPrice_array_smaller_S = AvgPrice_array_smaller_S.flatten()
	AvgPrice_array_larger_S = AvgPrice_array_larger_S.flatten() 	
	AvgPrice_array_smaller_sigma = AvgPrice_array_smaller_sigma.flatten()
	AvgPrice_array_larger_sigma = AvgPrice_array_larger_sigma.flatten()
	AvgPrice_array_smaller_t = AvgPrice_array_smaller_t.flatten()
	AvgPrice_array_larger_t = AvgPrice_array_larger_t.flatten()
	AvgPrice_array_smaller_r = AvgPrice_array_smaller_r.flatten()
	AvgPrice_array_larger_r = AvgPrice_array_larger_r.flatten()	

	###############################################################
	
	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array	
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	log_path_array_AT = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_AT ), axis=1 ) ### add the initial price as the first entry
		
	#################################################################################
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma_AT = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array_AT
	log_step_array_larger_sigma_AT = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array_AT	
	
	log_path_array_smaller_sigma_AT = np.log(S) + np.cumsum( log_step_array_smaller_sigma_AT, axis = 1 ) 
	log_path_array_larger_sigma_AT = np.log(S) + np.cumsum( log_step_array_larger_sigma_AT, axis = 1 )
	
	log_path_array_smaller_sigma_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_smaller_sigma_AT), axis=1 ) # add the initial price as the first entry
	log_path_array_larger_sigma_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_larger_sigma_AT), axis=1 ) # add the initial price as the first entry 	 		 	 	
	
	#### perturb in r for rho #################
	log_step_array_smaller_r_AT = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT
	log_step_array_larger_r_AT = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT	
	
	log_path_array_smaller_r_AT = np.log(S) + np.cumsum( log_step_array_smaller_r_AT, axis = 1 ) 
	log_path_array_larger_r_AT = np.log(S) + np.cumsum( log_step_array_larger_r_AT, axis = 1 )

	log_path_array_smaller_r_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_smaller_r_AT ), axis=1 ) # add the initial price as the first entry
	log_path_array_larger_r_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_larger_r_AT ), axis=1 ) # add the initial price as the first entry 		

	############ Average prices ###########################
	AvgPrice_array_AT = np.mean( np.exp( log_path_array_AT[:, :-1] ) , axis=1, keepdims=True) ### ignore the last column as this is the additional time step for theta calculation
	
	AvgPrice_array_smaller_S_AT = AvgPrice_array_AT*(S-0.01)/S
	AvgPrice_array_larger_S_AT = AvgPrice_array_AT*(S+0.01)/S
	
	AvgPrice_array_smaller_sigma_AT = np.mean( np.exp( log_path_array_smaller_sigma_AT[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_sigma_AT =  np.mean( np.exp( log_path_array_larger_sigma_AT[:, :-1] ) , axis=1, keepdims=True)
	
	AvgPrice_array_smaller_t_AT = np.mean( np.exp( log_path_array_AT ) , axis=1, keepdims=True) ### keep the last column as this is the additional time step for a longer time to maturity
	AvgPrice_array_larger_t_AT = np.mean( np.exp( log_path_array_AT[:, :-2] ) , axis=1, keepdims=True)  ### ignore the last two columns as the time to maturity is shorter
	
	AvgPrice_array_smaller_r_AT = np.mean( np.exp( log_path_array_smaller_r_AT[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_r_AT = np.mean( np.exp( log_path_array_larger_r_AT[:, :-1] ) , axis=1, keepdims=True)

	##### flatten arrays #####
	AvgPrice_array_AT = AvgPrice_array_AT.flatten()
	AvgPrice_array_smaller_S_AT = AvgPrice_array_smaller_S_AT.flatten()
	AvgPrice_array_larger_S_AT = AvgPrice_array_larger_S_AT.flatten() 	
	AvgPrice_array_smaller_sigma_AT = AvgPrice_array_smaller_sigma_AT.flatten()
	AvgPrice_array_larger_sigma_AT = AvgPrice_array_larger_sigma_AT.flatten()
	AvgPrice_array_smaller_t_AT = AvgPrice_array_smaller_t_AT.flatten()
	AvgPrice_array_larger_t_AT = AvgPrice_array_larger_t_AT.flatten()
	AvgPrice_array_smaller_r_AT = AvgPrice_array_smaller_r_AT.flatten()
	AvgPrice_array_larger_r_AT = AvgPrice_array_larger_r_AT.flatten()
	
	############## Concatenate #########################
	AvgPrice_array = np.concatenate( (AvgPrice_array, AvgPrice_array_AT) )
	AvgPrice_array_smaller_S = np.concatenate( (AvgPrice_array_smaller_S, AvgPrice_array_smaller_S_AT) )
	AvgPrice_array_larger_S = np.concatenate( (AvgPrice_array_larger_S, AvgPrice_array_larger_S_AT) ) 	
	AvgPrice_array_smaller_sigma = np.concatenate( (AvgPrice_array_smaller_sigma, AvgPrice_array_smaller_sigma_AT) )
	AvgPrice_array_larger_sigma = np.concatenate( (AvgPrice_array_larger_sigma, AvgPrice_array_larger_sigma_AT) )
	AvgPrice_array_smaller_t = np.concatenate( (AvgPrice_array_smaller_t, AvgPrice_array_smaller_t_AT) )
	AvgPrice_array_larger_t = np.concatenate( (AvgPrice_array_larger_t, AvgPrice_array_larger_t_AT) )
	AvgPrice_array_smaller_r = np.concatenate( (AvgPrice_array_smaller_r, AvgPrice_array_smaller_r_AT) )
	AvgPrice_array_larger_r = np.concatenate( (AvgPrice_array_larger_r, AvgPrice_array_larger_r_AT) )


	############################################################### 
	###### CALCULATE THE OPTION PRICE ###################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( K - AvgPrice_array , 0 )
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	payoff_array_smaller_S = np.maximum( K - AvgPrice_array_smaller_S, 0 )
	payoff_array_larger_S = np.maximum( K - AvgPrice_array_larger_S , 0 )
	
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
	payoff_array_smaller_sigma =  np.maximum( K - AvgPrice_array_smaller_sigma, 0 )
	payoff_array_larger_sigma = np.maximum( K - AvgPrice_array_larger_sigma, 0 )
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	payoff_array_smaller_t =  np.maximum( K - AvgPrice_array_smaller_t , 0 )
	payoff_array_larger_t = np.maximum( K - AvgPrice_array_larger_t , 0 )

	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t	# Think if additional correction should be taken into account to implement partial here
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t	# Think if additional correction should be taken into account to implement partial here	
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	payoff_array_smaller_r =  np.maximum( K - AvgPrice_array_smaller_r , 0 )
	payoff_array_larger_r =  np.maximum( K - AvgPrice_array_larger_r , 0 )
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	rho_value = np.mean(rho_array)
	rho_StandardError = stats.sem(rho_array)	
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)
	
#################### AVERAGE STRIKE CALL ###########################

def MonteCarloAvgStrikeCallWithGreeks(S, r, sigma, t, T, Smintodate=None, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculates the Asian average strike call using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savgsofar is the average price from time 0 to time t [default is none, e.g. appropriate when t = 0, just as the option comes into existence]
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	BaseSeed is the starting seed for the Seeds input into the Monte-Carlo
	
	Note: The Payoff is given by Max[S(T) - Savg,0], where Savg is the arithmetic average price of the underlying.
	
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
	###########################
	###########################
	
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
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 )
	
	log_path_array_smaller_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_sigma ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_sigma ), axis=1 ) ### add the initial price as the first entry 	 		 	 	
	
	#### perturb in r for rho #################
	log_step_array_smaller_r = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array
	log_step_array_larger_r = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array	
	
	log_path_array_smaller_r = np.log(S) + np.cumsum( log_step_array_smaller_r, axis = 1 ) 
	log_path_array_larger_r = np.log(S) + np.cumsum( log_step_array_larger_r, axis = 1 )

	log_path_array_smaller_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_r ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_r ), axis=1 ) ### add the initial price as the first entry 
	
	############ Terminal prices ##########################
	terminal_price_array = np.exp(log_path_array[:, -2])		
	
	price_step = 0.01
	terminal_price_array_smaller_S = terminal_price_array*(S-price_step)/S
	terminal_price_array_larger_S = terminal_price_array*(S+price_step)/S 
	
	terminal_price_array_smaller_sigma = np.exp(log_path_array_smaller_sigma[:, -2])
	terminal_price_array_larger_sigma = np.exp(log_path_array_larger_sigma[:, -2])
	
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	
	terminal_price_array_smaller_r = np.exp(log_path_array_smaller_r[:, -2])
	terminal_price_array_larger_r = np.exp(log_path_array_larger_r[:, -2])	
	
	############ Average prices ###########################
	AvgPrice_array = np.mean( np.exp( log_path_array[:, :-1] ) , axis=1, keepdims=True) ### ignore the last column as this is the additional time step for theta calculation
	
	AvgPrice_array_smaller_S = AvgPrice_array*(S-price_step)/S
	AvgPrice_array_larger_S = AvgPrice_array*(S+price_step)/S
	
	AvgPrice_array_smaller_sigma = np.mean( np.exp( log_path_array_smaller_sigma[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_sigma =  np.mean( np.exp( log_path_array_larger_sigma[:, :-1] ) , axis=1, keepdims=True)
	
	AvgPrice_array_smaller_t = np.mean( np.exp( log_path_array ) , axis=1, keepdims=True) ### keep the last column as this is the additional time step for a longer time to maturity
	AvgPrice_array_larger_t = np.mean( np.exp( log_path_array[:, :-2] ) , axis=1, keepdims=True)  ### ignore the last two columns as the time to maturity is shorter
	
	AvgPrice_array_smaller_r = np.mean( np.exp( log_path_array_smaller_r[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_r = np.mean( np.exp( log_path_array_larger_r[:, :-1] ) , axis=1, keepdims=True)

	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()
	terminal_price_array_smaller_S = terminal_price_array_smaller_S.flatten()
	terminal_price_array_larger_S = terminal_price_array_larger_S.flatten() 	
	terminal_price_array_smaller_sigma = terminal_price_array_smaller_sigma.flatten()
	terminal_price_array_larger_sigma = terminal_price_array_larger_sigma.flatten()
	terminal_price_array_smaller_t = terminal_price_array_smaller_t.flatten()
	terminal_price_array_larger_t = terminal_price_array_larger_t.flatten()
	terminal_price_array_smaller_r = terminal_price_array_smaller_r.flatten()
	terminal_price_array_larger_r = terminal_price_array_larger_r.flatten()
	
	AvgPrice_array = AvgPrice_array.flatten()
	AvgPrice_array_smaller_S = AvgPrice_array_smaller_S.flatten()
	AvgPrice_array_larger_S = AvgPrice_array_larger_S.flatten() 	
	AvgPrice_array_smaller_sigma = AvgPrice_array_smaller_sigma.flatten()
	AvgPrice_array_larger_sigma = AvgPrice_array_larger_sigma.flatten()
	AvgPrice_array_smaller_t = AvgPrice_array_smaller_t.flatten()
	AvgPrice_array_larger_t = AvgPrice_array_larger_t.flatten()
	AvgPrice_array_smaller_r = AvgPrice_array_smaller_r.flatten()
	AvgPrice_array_larger_r = AvgPrice_array_larger_r.flatten()	

	###############################################################
	
	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array	
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	log_path_array_AT = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_AT ), axis=1 ) ### add the initial price as the first entry
		
	#################################################################################
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma_AT = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array_AT
	log_step_array_larger_sigma_AT = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array_AT	
	
	log_path_array_smaller_sigma_AT = np.log(S) + np.cumsum( log_step_array_smaller_sigma_AT, axis = 1 ) 
	log_path_array_larger_sigma_AT = np.log(S) + np.cumsum( log_step_array_larger_sigma_AT, axis = 1 )
	
	log_path_array_smaller_sigma_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_smaller_sigma_AT), axis=1 ) # add the initial price as the first entry
	log_path_array_larger_sigma_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_larger_sigma_AT), axis=1 ) # add the initial price as the first entry 	 		 	 	
	
	#### perturb in r for rho #################
	log_step_array_smaller_r_AT = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT
	log_step_array_larger_r_AT = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT	
	
	log_path_array_smaller_r_AT = np.log(S) + np.cumsum( log_step_array_smaller_r_AT, axis = 1 ) 
	log_path_array_larger_r_AT = np.log(S) + np.cumsum( log_step_array_larger_r_AT, axis = 1 )

	log_path_array_smaller_r_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_smaller_r_AT ), axis=1 ) # add the initial price as the first entry
	log_path_array_larger_r_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_larger_r_AT ), axis=1 ) # add the initial price as the first entry 		

	############ Average prices ###########################
	AvgPrice_array_AT = np.mean( np.exp( log_path_array_AT[:, :-1] ) , axis=1, keepdims=True) ### ignore the last column as this is the additional time step for theta calculation
	
	AvgPrice_array_smaller_S_AT = AvgPrice_array_AT*(S-price_step)/S
	AvgPrice_array_larger_S_AT = AvgPrice_array_AT*(S+price_step)/S
	
	AvgPrice_array_smaller_sigma_AT = np.mean( np.exp( log_path_array_smaller_sigma_AT[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_sigma_AT =  np.mean( np.exp( log_path_array_larger_sigma_AT[:, :-1] ) , axis=1, keepdims=True)
	
	AvgPrice_array_smaller_t_AT = np.mean( np.exp( log_path_array_AT ) , axis=1, keepdims=True) ### keep the last column as this is the additional time step for a longer time to maturity
	AvgPrice_array_larger_t_AT = np.mean( np.exp( log_path_array_AT[:, :-2] ) , axis=1, keepdims=True)  ### ignore the last two columns as the time to maturity is shorter
	
	AvgPrice_array_smaller_r_AT = np.mean( np.exp( log_path_array_smaller_r_AT[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_r_AT = np.mean( np.exp( log_path_array_larger_r_AT[:, :-1] ) , axis=1, keepdims=True)

	############ Terminal prices ##########################
	terminal_price_array_AT = np.exp(log_path_array_AT[:, -2])		
	
	terminal_price_array_smaller_S_AT = terminal_price_array_AT*(S-price_step)/S
	terminal_price_array_larger_S_AT = terminal_price_array_AT*(S+price_step)/S 
	
	terminal_price_array_smaller_sigma_AT = np.exp(log_path_array_smaller_sigma_AT[:, -2])
	terminal_price_array_larger_sigma_AT = np.exp(log_path_array_larger_sigma_AT[:, -2])
	
	terminal_price_array_smaller_t_AT = np.exp(log_path_array_AT[:, -1])
	terminal_price_array_larger_t_AT = np.exp(log_path_array_AT[:, -3])
	
	terminal_price_array_smaller_r_AT = np.exp(log_path_array_smaller_r_AT[:, -2])
	terminal_price_array_larger_r_AT = np.exp(log_path_array_larger_r_AT[:, -2])		

	##### flatten arrays #####
	terminal_price_array_AT = terminal_price_array_AT.flatten()
	terminal_price_array_smaller_S_AT = terminal_price_array_smaller_S_AT.flatten()
	terminal_price_array_larger_S_AT = terminal_price_array_larger_S_AT.flatten() 	
	terminal_price_array_smaller_sigma_AT = terminal_price_array_smaller_sigma_AT.flatten()
	terminal_price_array_larger_sigma_AT = terminal_price_array_larger_sigma_AT.flatten()
	terminal_price_array_smaller_t_AT = terminal_price_array_smaller_t_AT.flatten()
	terminal_price_array_larger_t_AT = terminal_price_array_larger_t_AT.flatten()
	terminal_price_array_smaller_r_AT = terminal_price_array_smaller_r_AT.flatten()
	terminal_price_array_larger_r_AT = terminal_price_array_larger_r_AT.flatten()
	
	AvgPrice_array_AT = AvgPrice_array_AT.flatten()
	AvgPrice_array_smaller_S_AT = AvgPrice_array_smaller_S_AT.flatten()
	AvgPrice_array_larger_S_AT = AvgPrice_array_larger_S_AT.flatten() 	
	AvgPrice_array_smaller_sigma_AT = AvgPrice_array_smaller_sigma_AT.flatten()
	AvgPrice_array_larger_sigma_AT = AvgPrice_array_larger_sigma_AT.flatten()
	AvgPrice_array_smaller_t_AT = AvgPrice_array_smaller_t_AT.flatten()
	AvgPrice_array_larger_t_AT = AvgPrice_array_larger_t_AT.flatten()
	AvgPrice_array_smaller_r_AT = AvgPrice_array_smaller_r_AT.flatten()
	AvgPrice_array_larger_r_AT = AvgPrice_array_larger_r_AT.flatten()
	
	############## Concatenate #########################
	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT ))
	terminal_price_array_smaller_S = np.concatenate( (terminal_price_array_smaller_S, terminal_price_array_smaller_S_AT ))
	terminal_price_array_larger_S = np.concatenate( (terminal_price_array_larger_S, terminal_price_array_larger_S_AT )) 	
	terminal_price_array_smaller_sigma = np.concatenate( (terminal_price_array_smaller_sigma, terminal_price_array_smaller_sigma_AT ))
	terminal_price_array_larger_sigma = np.concatenate( (terminal_price_array_larger_sigma, terminal_price_array_larger_sigma_AT))
	terminal_price_array_smaller_t = np.concatenate( (terminal_price_array_smaller_t, terminal_price_array_smaller_t_AT ))
	terminal_price_array_larger_t = np.concatenate( (terminal_price_array_larger_t, terminal_price_array_larger_t_AT ))
	terminal_price_array_smaller_r = np.concatenate( (terminal_price_array_smaller_r, terminal_price_array_smaller_r_AT ))
	terminal_price_array_larger_r = np.concatenate( (terminal_price_array_larger_r,terminal_price_array_larger_r_AT ))
	
	
	AvgPrice_array = np.concatenate( (AvgPrice_array, AvgPrice_array_AT) )
	AvgPrice_array_smaller_S = np.concatenate( (AvgPrice_array_smaller_S, AvgPrice_array_smaller_S_AT) )
	AvgPrice_array_larger_S = np.concatenate( (AvgPrice_array_larger_S, AvgPrice_array_larger_S_AT) ) 	
	AvgPrice_array_smaller_sigma = np.concatenate( (AvgPrice_array_smaller_sigma, AvgPrice_array_smaller_sigma_AT) )
	AvgPrice_array_larger_sigma = np.concatenate( (AvgPrice_array_larger_sigma, AvgPrice_array_larger_sigma_AT) )
	AvgPrice_array_smaller_t = np.concatenate( (AvgPrice_array_smaller_t, AvgPrice_array_smaller_t_AT) )
	AvgPrice_array_larger_t = np.concatenate( (AvgPrice_array_larger_t, AvgPrice_array_larger_t_AT) )
	AvgPrice_array_smaller_r = np.concatenate( (AvgPrice_array_smaller_r, AvgPrice_array_smaller_r_AT) )
	AvgPrice_array_larger_r = np.concatenate( (AvgPrice_array_larger_r, AvgPrice_array_larger_r_AT) )
	
	###### CALCULATE THE OPTION PRICE ###################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( terminal_price_array - AvgPrice_array , 0 )
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	payoff_array_smaller_S = np.maximum( terminal_price_array_smaller_S - AvgPrice_array_smaller_S , 0 )
	payoff_array_larger_S = np.maximum( terminal_price_array_larger_S - AvgPrice_array_larger_S , 0 )
	
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*price_step)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(price_step**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	payoff_array_smaller_sigma =  np.maximum( terminal_price_array_smaller_sigma - AvgPrice_array_smaller_sigma , 0 )
	payoff_array_larger_sigma = np.maximum( terminal_price_array_larger_sigma - AvgPrice_array_larger_sigma , 0 )
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	payoff_array_smaller_t =  np.maximum( terminal_price_array_smaller_t - AvgPrice_array_smaller_t , 0 )
	payoff_array_larger_t = np.maximum( terminal_price_array_larger_t - AvgPrice_array_larger_t , 0 )

	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t	# Think if additional correction should be taken into account to implement partial here
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t	# Think if additional correction should be taken into account to implement partial here	
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	payoff_array_smaller_r =  np.maximum( terminal_price_array_smaller_r - AvgPrice_array_smaller_r , 0 )
	payoff_array_larger_r =  np.maximum( terminal_price_array_larger_r - AvgPrice_array_larger_r , 0 )
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	rho_value = np.mean(rho_array)
	rho_StandardError = stats.sem(rho_array)	
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)



#################### AVERAGE STRIKE PUT ###########################

def MonteCarloAvgStrikePutWithGreeks(S, r, sigma, t, T, Smintodate=None, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculates the Asian average strike put using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Savgsofar is the average price from time 0 to time t [default is none, e.g. appropriate when t = 0, just as the option comes into existence]
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	BaseSeed is the starting seed for the Seeds input into the Monte-Carlo
	
	Note: The Payoff is given by Max[Savg-S(T),0], where Savg is the arithmetic average price of the underlying.
	
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
	###########################
	###########################
	
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
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 )
	
	log_path_array_smaller_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_sigma ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_sigma ), axis=1 ) ### add the initial price as the first entry 	 		 	 	
	
	#### perturb in r for rho #################
	log_step_array_smaller_r = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array
	log_step_array_larger_r = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array	
	
	log_path_array_smaller_r = np.log(S) + np.cumsum( log_step_array_smaller_r, axis = 1 ) 
	log_path_array_larger_r = np.log(S) + np.cumsum( log_step_array_larger_r, axis = 1 )

	log_path_array_smaller_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_r ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_r ), axis=1 ) ### add the initial price as the first entry 
	
	############ Terminal prices ##########################
	terminal_price_array = np.exp(log_path_array[:, -2])		
	
	price_step = 0.01
	terminal_price_array_smaller_S = terminal_price_array*(S-price_step)/S
	terminal_price_array_larger_S = terminal_price_array*(S+price_step)/S 
	
	terminal_price_array_smaller_sigma = np.exp(log_path_array_smaller_sigma[:, -2])
	terminal_price_array_larger_sigma = np.exp(log_path_array_larger_sigma[:, -2])
	
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	
	terminal_price_array_smaller_r = np.exp(log_path_array_smaller_r[:, -2])
	terminal_price_array_larger_r = np.exp(log_path_array_larger_r[:, -2])	
	
	############ Average prices ###########################
	AvgPrice_array = np.mean( np.exp( log_path_array[:, :-1] ) , axis=1, keepdims=True) ### ignore the last column as this is the additional time step for theta calculation
	
	AvgPrice_array_smaller_S = AvgPrice_array*(S-price_step)/S
	AvgPrice_array_larger_S = AvgPrice_array*(S+price_step)/S
	
	AvgPrice_array_smaller_sigma = np.mean( np.exp( log_path_array_smaller_sigma[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_sigma =  np.mean( np.exp( log_path_array_larger_sigma[:, :-1] ) , axis=1, keepdims=True)
	
	AvgPrice_array_smaller_t = np.mean( np.exp( log_path_array ) , axis=1, keepdims=True) ### keep the last column as this is the additional time step for a longer time to maturity
	AvgPrice_array_larger_t = np.mean( np.exp( log_path_array[:, :-2] ) , axis=1, keepdims=True)  ### ignore the last two columns as the time to maturity is shorter
	
	AvgPrice_array_smaller_r = np.mean( np.exp( log_path_array_smaller_r[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_r = np.mean( np.exp( log_path_array_larger_r[:, :-1] ) , axis=1, keepdims=True)

	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()
	terminal_price_array_smaller_S = terminal_price_array_smaller_S.flatten()
	terminal_price_array_larger_S = terminal_price_array_larger_S.flatten() 	
	terminal_price_array_smaller_sigma = terminal_price_array_smaller_sigma.flatten()
	terminal_price_array_larger_sigma = terminal_price_array_larger_sigma.flatten()
	terminal_price_array_smaller_t = terminal_price_array_smaller_t.flatten()
	terminal_price_array_larger_t = terminal_price_array_larger_t.flatten()
	terminal_price_array_smaller_r = terminal_price_array_smaller_r.flatten()
	terminal_price_array_larger_r = terminal_price_array_larger_r.flatten()
	
	AvgPrice_array = AvgPrice_array.flatten()
	AvgPrice_array_smaller_S = AvgPrice_array_smaller_S.flatten()
	AvgPrice_array_larger_S = AvgPrice_array_larger_S.flatten() 	
	AvgPrice_array_smaller_sigma = AvgPrice_array_smaller_sigma.flatten()
	AvgPrice_array_larger_sigma = AvgPrice_array_larger_sigma.flatten()
	AvgPrice_array_smaller_t = AvgPrice_array_smaller_t.flatten()
	AvgPrice_array_larger_t = AvgPrice_array_larger_t.flatten()
	AvgPrice_array_smaller_r = AvgPrice_array_smaller_r.flatten()
	AvgPrice_array_larger_r = AvgPrice_array_larger_r.flatten()	

	###############################################################
	
	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array	
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	log_path_array_AT = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_AT ), axis=1 ) ### add the initial price as the first entry
		
	#################################################################################
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma_AT = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array_AT
	log_step_array_larger_sigma_AT = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array_AT	
	
	log_path_array_smaller_sigma_AT = np.log(S) + np.cumsum( log_step_array_smaller_sigma_AT, axis = 1 ) 
	log_path_array_larger_sigma_AT = np.log(S) + np.cumsum( log_step_array_larger_sigma_AT, axis = 1 )
	
	log_path_array_smaller_sigma_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_smaller_sigma_AT), axis=1 ) # add the initial price as the first entry
	log_path_array_larger_sigma_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_larger_sigma_AT), axis=1 ) # add the initial price as the first entry 	 		 	 	
	
	#### perturb in r for rho #################
	log_step_array_smaller_r_AT = (r - 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT
	log_step_array_larger_r_AT = (r + 1e-4 - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT	
	
	log_path_array_smaller_r_AT = np.log(S) + np.cumsum( log_step_array_smaller_r_AT, axis = 1 ) 
	log_path_array_larger_r_AT = np.log(S) + np.cumsum( log_step_array_larger_r_AT, axis = 1 )

	log_path_array_smaller_r_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_smaller_r_AT ), axis=1 ) # add the initial price as the first entry
	log_path_array_larger_r_AT = np.concatenate((np.full( shape = (n_simulations, 1) , fill_value = np.log(S)), log_path_array_larger_r_AT ), axis=1 ) # add the initial price as the first entry 		

	############ Average prices ###########################
	AvgPrice_array_AT = np.mean( np.exp( log_path_array_AT[:, :-1] ) , axis=1, keepdims=True) ### ignore the last column as this is the additional time step for theta calculation
	
	AvgPrice_array_smaller_S_AT = AvgPrice_array_AT*(S-price_step)/S
	AvgPrice_array_larger_S_AT = AvgPrice_array_AT*(S+price_step)/S
	
	AvgPrice_array_smaller_sigma_AT = np.mean( np.exp( log_path_array_smaller_sigma_AT[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_sigma_AT =  np.mean( np.exp( log_path_array_larger_sigma_AT[:, :-1] ) , axis=1, keepdims=True)
	
	AvgPrice_array_smaller_t_AT = np.mean( np.exp( log_path_array_AT ) , axis=1, keepdims=True) ### keep the last column as this is the additional time step for a longer time to maturity
	AvgPrice_array_larger_t_AT = np.mean( np.exp( log_path_array_AT[:, :-2] ) , axis=1, keepdims=True)  ### ignore the last two columns as the time to maturity is shorter
	
	AvgPrice_array_smaller_r_AT = np.mean( np.exp( log_path_array_smaller_r_AT[:, :-1] ) , axis=1, keepdims=True)
	AvgPrice_array_larger_r_AT = np.mean( np.exp( log_path_array_larger_r_AT[:, :-1] ) , axis=1, keepdims=True)

	############ Terminal prices ##########################
	terminal_price_array_AT = np.exp(log_path_array_AT[:, -2])		
	
	terminal_price_array_smaller_S_AT = terminal_price_array_AT*(S-price_step)/S
	terminal_price_array_larger_S_AT = terminal_price_array_AT*(S+price_step)/S 
	
	terminal_price_array_smaller_sigma_AT = np.exp(log_path_array_smaller_sigma_AT[:, -2])
	terminal_price_array_larger_sigma_AT = np.exp(log_path_array_larger_sigma_AT[:, -2])
	
	terminal_price_array_smaller_t_AT = np.exp(log_path_array_AT[:, -1])
	terminal_price_array_larger_t_AT = np.exp(log_path_array_AT[:, -3])
	
	terminal_price_array_smaller_r_AT = np.exp(log_path_array_smaller_r_AT[:, -2])
	terminal_price_array_larger_r_AT = np.exp(log_path_array_larger_r_AT[:, -2])		

	##### flatten arrays #####
	terminal_price_array_AT = terminal_price_array_AT.flatten()
	terminal_price_array_smaller_S_AT = terminal_price_array_smaller_S_AT.flatten()
	terminal_price_array_larger_S_AT = terminal_price_array_larger_S_AT.flatten() 	
	terminal_price_array_smaller_sigma_AT = terminal_price_array_smaller_sigma_AT.flatten()
	terminal_price_array_larger_sigma_AT = terminal_price_array_larger_sigma_AT.flatten()
	terminal_price_array_smaller_t_AT = terminal_price_array_smaller_t_AT.flatten()
	terminal_price_array_larger_t_AT = terminal_price_array_larger_t_AT.flatten()
	terminal_price_array_smaller_r_AT = terminal_price_array_smaller_r_AT.flatten()
	terminal_price_array_larger_r_AT = terminal_price_array_larger_r_AT.flatten()
	
	AvgPrice_array_AT = AvgPrice_array_AT.flatten()
	AvgPrice_array_smaller_S_AT = AvgPrice_array_smaller_S_AT.flatten()
	AvgPrice_array_larger_S_AT = AvgPrice_array_larger_S_AT.flatten() 	
	AvgPrice_array_smaller_sigma_AT = AvgPrice_array_smaller_sigma_AT.flatten()
	AvgPrice_array_larger_sigma_AT = AvgPrice_array_larger_sigma_AT.flatten()
	AvgPrice_array_smaller_t_AT = AvgPrice_array_smaller_t_AT.flatten()
	AvgPrice_array_larger_t_AT = AvgPrice_array_larger_t_AT.flatten()
	AvgPrice_array_smaller_r_AT = AvgPrice_array_smaller_r_AT.flatten()
	AvgPrice_array_larger_r_AT = AvgPrice_array_larger_r_AT.flatten()
	
	############## Concatenate #########################
	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT ))
	terminal_price_array_smaller_S = np.concatenate( (terminal_price_array_smaller_S, terminal_price_array_smaller_S_AT ))
	terminal_price_array_larger_S = np.concatenate( (terminal_price_array_larger_S, terminal_price_array_larger_S_AT )) 	
	terminal_price_array_smaller_sigma = np.concatenate( (terminal_price_array_smaller_sigma, terminal_price_array_smaller_sigma_AT ))
	terminal_price_array_larger_sigma = np.concatenate( (terminal_price_array_larger_sigma, terminal_price_array_larger_sigma_AT))
	terminal_price_array_smaller_t = np.concatenate( (terminal_price_array_smaller_t, terminal_price_array_smaller_t_AT ))
	terminal_price_array_larger_t = np.concatenate( (terminal_price_array_larger_t, terminal_price_array_larger_t_AT ))
	terminal_price_array_smaller_r = np.concatenate( (terminal_price_array_smaller_r, terminal_price_array_smaller_r_AT ))
	terminal_price_array_larger_r = np.concatenate( (terminal_price_array_larger_r,terminal_price_array_larger_r_AT ))
	
	
	AvgPrice_array = np.concatenate( (AvgPrice_array, AvgPrice_array_AT) )
	AvgPrice_array_smaller_S = np.concatenate( (AvgPrice_array_smaller_S, AvgPrice_array_smaller_S_AT) )
	AvgPrice_array_larger_S = np.concatenate( (AvgPrice_array_larger_S, AvgPrice_array_larger_S_AT) ) 	
	AvgPrice_array_smaller_sigma = np.concatenate( (AvgPrice_array_smaller_sigma, AvgPrice_array_smaller_sigma_AT) )
	AvgPrice_array_larger_sigma = np.concatenate( (AvgPrice_array_larger_sigma, AvgPrice_array_larger_sigma_AT) )
	AvgPrice_array_smaller_t = np.concatenate( (AvgPrice_array_smaller_t, AvgPrice_array_smaller_t_AT) )
	AvgPrice_array_larger_t = np.concatenate( (AvgPrice_array_larger_t, AvgPrice_array_larger_t_AT) )
	AvgPrice_array_smaller_r = np.concatenate( (AvgPrice_array_smaller_r, AvgPrice_array_smaller_r_AT) )
	AvgPrice_array_larger_r = np.concatenate( (AvgPrice_array_larger_r, AvgPrice_array_larger_r_AT) )

	
	###### CALCULATE THE OPTION PRICE ###################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( AvgPrice_array - terminal_price_array , 0 )
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	payoff_array_smaller_S = np.maximum( AvgPrice_array_smaller_S - terminal_price_array_smaller_S , 0 )
	payoff_array_larger_S = np.maximum( AvgPrice_array_larger_S - terminal_price_array_larger_S  , 0 )
	
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*price_step)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(price_step**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	payoff_array_smaller_sigma =  np.maximum(  AvgPrice_array_smaller_sigma - terminal_price_array_smaller_sigma , 0 )
	payoff_array_larger_sigma = np.maximum( AvgPrice_array_larger_sigma - terminal_price_array_larger_sigma  , 0 )
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	payoff_array_smaller_t =  np.maximum(  AvgPrice_array_smaller_t - terminal_price_array_smaller_t  , 0 )
	payoff_array_larger_t = np.maximum( AvgPrice_array_larger_t - terminal_price_array_larger_t , 0 )

	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t	# Think if additional correction should be taken into account to implement partial here
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t	# Think if additional correction should be taken into account to implement partial here	
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	payoff_array_smaller_r =  np.maximum( AvgPrice_array_smaller_r - terminal_price_array_smaller_r  , 0 )
	payoff_array_larger_r =  np.maximum( AvgPrice_array_larger_r - terminal_price_array_larger_r  , 0 )
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	rho_value = np.mean(rho_array)
	rho_StandardError = stats.sem(rho_array)	
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)




##################################################################		

def main():
	print('\n')
	print(ApproxAvgPriceCall(50,60,0.1,0.4,0,1))
	print(ApproxAvgPriceCallWithGreeks(50,60,0.1,0.4,0,1))
	print(MonteCarloAvgPriceCallWithGreeks(50,60,0.1,0.4,0,1))
	
	print('\n')
	print(ApproxAvgPricePut(50,60,0.1,0.4,0,1))
	print(ApproxAvgPricePutWithGreeks(50,60,0.1,0.4,0,1))
	print(MonteCarloAvgPricePutWithGreeks(50,60,0.1,0.4,0,1))	

	print('\n')
	print(ApproxAvgStrikeCall(50,0.1,0.4,0,1))
	print(ApproxAvgStrikeCallWithGreeks(50,0.1,0.4,0,1))	
	print(MonteCarloAvgStrikeCallWithGreeks(50,0.1,0.4,0,1))
	
	print('\n')
	print(ApproxAvgStrikePut(50,0.1,0.4,0,1))
	print(ApproxAvgStrikePutWithGreeks(50,0.1,0.4,0,1))			
	print(MonteCarloAvgStrikePutWithGreeks(50,0.1,0.4,0,1))		

if __name__ == "__main__":
	main()	    									
  		
