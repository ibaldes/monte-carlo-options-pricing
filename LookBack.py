import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from scipy.stats import norm
from multiprocessing import Pool

###

'''
GIVES ANALYTIC AND MONTE-CARLO IMPLEMENTATION OF LOOKBACK OPTIONS WITH BARRIERS
RETURNS THE OPTION PRICE, GREEKS, AND STANDARD ERRORS
'''

StandardBaseSeed = 0


##################################################################################################################################################

def AnalyticFloatingStrikeLookBackCall(S,r,sigma,t,T,Smintodate=None):	
	'''
	Calculates the Floating Strike LookBack Call Price (assuming Black Scholes) using the analytic formula.
	
	Note: The Strike Price is given by the Minimum of S achieved so far [default if note entered: Smintodate = S(t)].
	Note: The Payoff is given by S(T) - Smin, where Smin is the minimum over the option lifetime

	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	Output is the call price in dollars
	'''
	#################################################################
	#################################################################
	
	if Smintodate is None:          
		Smintodate = S		##### Set Smintodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
	
	####### Implement formula from Hull, 11ed., Chapter 26 ########
	
	a1 = ( np.log( S/Smintodate ) + ( r + 0.5*sigma**2 )*(T-t) ) / (sigma*np.sqrt(T-t))
	a2 = a1 - sigma*np.sqrt(T-t)
	a3 = ( np.log( S/Smintodate ) + (-r + 0.5*sigma**2 )*(T-t) ) / (sigma*np.sqrt(T-t))
	Y1 = -2 * ( r - 0.5*sigma**2 )*np.log( S/Smintodate ) / (sigma**2)
	
	FloatingLookBackCallPrice = S*norm.cdf(a1) - S*sigma**2/(2*r)*norm.cdf(-a1) - Smintodate*np.exp(-r*(T-t))*(norm.cdf(a2) - sigma**2/(2*r)*np.exp(Y1)*norm.cdf(-a3) )
	
	return(FloatingLookBackCallPrice)
	

##################################################################################################################################################

def AnalyticFloatingStrikeLookBackPut(S,r,sigma,t,T,Smaxtodate=None):	
	'''
	Calculates the Floating Strike LookBack Put Price (assuming Black Scholes) using the analytic formula.
	
	Note: The Strike Price is given by the Maximum of S achieved so far [default if note entered: Smaxtodate = S(t)].
	Note: The Payoff is given by Smax - S(T), where Smax is the maximum over the option lifetime
	
	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	Output is the put price in dollars
	'''
	#################################################################
	#################################################################
	
	if Smaxtodate is None:          
		Smaxtodate = S 		       ##### Set Smaxtodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
	
	####### Implement formula from Hull, 11ed., Chapter 26 ########
	
	b1 = ( np.log( Smaxtodate/S ) + (-r + 0.5*sigma**2 )*(T-t) ) /  (sigma*np.sqrt(T-t))
	b2 = b1 - sigma*np.sqrt(T-t)
	b3 = ( np.log( Smaxtodate/S ) + ( r - 0.5*sigma**2 )*(T-t) ) /  (sigma*np.sqrt(T-t))
	Y2 = 2 * ( r - 0.5*sigma**2 )*np.log( Smaxtodate/ S ) / (sigma**2)
	
	FloatingLookBackPutPrice = Smaxtodate*np.exp(-r*(T-t))*( norm.cdf(b1) - sigma**2/(2*r)*np.exp(Y2)*norm.cdf(-b3) ) + S*sigma**2/(2*r)*norm.cdf(-b2) - S*norm.cdf(b2)

	return(FloatingLookBackPutPrice)
	

##################################################################################################################################################


def AnalyticFixedStrikeLookBackCall(S,K,r,sigma,t,T,Smaxtodate=None):	
	'''
	Calculates the Fixed Strike LookBack Call Price (assuming Black Scholes) using the analytic formula.
	
	Note: The Payoff is given by maximum( Smax - K , 0), where Smax is the maximum S over the option lifetime

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	
	Output is the call price in dollars
	'''
	#################################################################
	#################################################################
	
	if Smaxtodate is None:          
		Smaxtodate = S 		       ##### Set Smaxtodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
		
	####### Implement formula from Hull, 11ed., Chapter 26 ########
	
	Smaxstar = max(Smaxtodate, K)

	FixedLookBackCallPrice = AnalyticFloatingStrikeLookBackPut(S,r,sigma,t,T,Smaxstar) + S - K*np.exp(-r*(T-t)) #### Priced using parity type relation

	return(FixedLookBackCallPrice)

    
##################################################################################################################################################

def AnalyticFixedStrikeLookBackPut(S,K,r,sigma,t,T,Smintodate=None):	
	'''
	Calculates the Fixed Strike LookBack Put Price (assuming Black Scholes) using the analytic formula.
	
	Note: The Payoff is given by maximum( K - Smin  , 0), where Smin is the minimum S over the option lifetime

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	
	Output is the put price in dollars
	'''
	#################################################################
	#################################################################
	
	if Smintodate is None:          
		Smintodate = S		##### Set Smintodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
	
	####### Implement formula from Hull, 11ed., Chapter 26 ########
	
	Sminstar = min(Smintodate, K)
	
	FixedLookBackPutPrice = AnalyticFloatingStrikeLookBackCall(S,r,sigma,t,T,Sminstar) + K*np.exp(-r*(T-t)) - S  #### Priced using parity type relation
	
	return(FixedLookBackPutPrice)


    
###### ANALYTIC FORMULAS WITH GREEKS ESTIMATED USING FINITE DIFFERENCE METHOD ################

def AnalyticFloatingStrikeLookBackCallWithGreeks(S,r,sigma,t,T,Smintodate=None):	
	'''
	Calculates the Floating Strike LookBack Call Price (assuming Black Scholes) using the analytic formula.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method

	Note: The Strike Price is given by the Minimum of S achieved so far [default if note entered: Smintodate = S(t)].
	Note: The Payoff is given by S(T) - Smin, where Smin is the minimum over the option lifetime

	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
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
	if Smintodate is None:          
		Smintodate = S		##### Set Smintodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
	
	#### Call Price Using Analytic Formula #######
	CallPrice = AnalyticFloatingStrikeLookBackCall(S,r,sigma,t,T,Smintodate)
	
	small_time_step = (T-t)/100

	##### Following is for self-consistency when using the finite element difference for calculating the greeks.	
	if (S-Smintodate) < 0.01:
		Smintodatecorrection = S - 0.01 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smintodatecorrection = Smintodate
	
	### The Greeks ####
	Delta = ( AnalyticFloatingStrikeLookBackCall(S+0.01,r,sigma,t,T,Smintodate) - AnalyticFloatingStrikeLookBackCall(S-0.01,r,sigma,t,T,Smintodatecorrection) ) / (2*0.01)
	Gamma = ( AnalyticFloatingStrikeLookBackCall(S+0.01,r,sigma,t,T,Smintodate) - 2*AnalyticFloatingStrikeLookBackCall(S,r,sigma,t,T,Smintodate) + AnalyticFloatingStrikeLookBackCall(S-0.01,r,sigma,t,T,Smintodatecorrection))/(0.01**2)
	
	Vega = ( AnalyticFloatingStrikeLookBackCall(S,r,sigma+0.01,t,T,Smintodate) - AnalyticFloatingStrikeLookBackCall(S,r,sigma-0.01,t,T,Smintodate) ) / (2*0.01)
	Theta = -( AnalyticFloatingStrikeLookBackCall(S,r,sigma,t+small_time_step,T,Smintodate) - AnalyticFloatingStrikeLookBackCall(S,r,sigma,t-small_time_step,T,Smintodate) ) / (2*small_time_step)
	Rho = ( AnalyticFloatingStrikeLookBackCall(S,r+1e-4,sigma,t,T,Smintodate) - AnalyticFloatingStrikeLookBackCall(S,r-1e-4,sigma,t,T,Smintodate) ) / (2*1e-4)
	
	return(CallPrice, Delta, Gamma, Vega, Theta, Rho)	

#####

def AnalyticFloatingStrikeLookBackPutWithGreeks(S,r,sigma,t,T,Smaxtodate=None):	
	'''
	Calculates the Floating Strike LookBack Put Price (assuming Black Scholes) using the analytic formula.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method

	Note: The Strike Price is given by the Minimum of S achieved so far [default if note entered: Smintodate = S(t)].
	Note: The Payoff is given by S(T) - Smin, where Smin is the minimum over the option lifetime

	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	Output is: 
	The Put price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	##########################
	##########################
	
	if Smaxtodate is None:          
		Smaxtodate = S 		       ##### Set Smaxtodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
	
	#### Put Price Using Analytic Formula #######
	PutPrice = AnalyticFloatingStrikeLookBackPut(S,r,sigma,t,T,Smaxtodate)
	
	small_time_step = (T-t)/100

	##### Following is for self-consistency when using the finite element difference for calculating the greeks.	
	if (Smaxtodate-S) < 0.01:
		Smaxtodatecorrection = S + 0.01 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smaxtodatecorrection = Smaxtodate
	
	### The Greeks ####
	Delta = ( AnalyticFloatingStrikeLookBackPut(S+0.01,r,sigma,t,T,Smaxtodatecorrection) - AnalyticFloatingStrikeLookBackPut(S-0.01,r,sigma,t,T,Smaxtodate) ) / (2*0.01)
	Gamma = ( AnalyticFloatingStrikeLookBackPut(S+0.01,r,sigma,t,T,Smaxtodatecorrection) - 2*AnalyticFloatingStrikeLookBackPut(S,r,sigma,t,T,Smaxtodate) + AnalyticFloatingStrikeLookBackPut(S-0.01,r,sigma,t,T,Smaxtodate))/(0.01**2)
	
	Vega = ( AnalyticFloatingStrikeLookBackPut(S,r,sigma+0.01,t,T,Smaxtodate) - AnalyticFloatingStrikeLookBackPut(S,r,sigma-0.01,t,T,Smaxtodate) ) / (2*0.01)
	Theta = -( AnalyticFloatingStrikeLookBackPut(S,r,sigma,t+small_time_step,T,Smaxtodate) - AnalyticFloatingStrikeLookBackPut(S,r,sigma,t-small_time_step,T,Smaxtodate) ) / (2*small_time_step)
	Rho = ( AnalyticFloatingStrikeLookBackPut(S,r+1e-4,sigma,t,T,Smaxtodate) - AnalyticFloatingStrikeLookBackPut(S,r-1e-4,sigma,t,T,Smaxtodate) ) / (2*1e-4)
	
	return(PutPrice, Delta, Gamma, Vega, Theta, Rho)	

############################################################################

def AnalyticFixedStrikeLookBackCallWithGreeks(S,K,r,sigma,t,T,Smaxtodate=None):	
	'''
	Calculates the Fixed Strike LookBack Call Price (assuming Black Scholes) using the analytic formula.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method
	
	Note: The Payoff is given by maximum( Smax - K , 0), where Smax is the maximum S over the option lifetime

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	#################################################################
	#################################################################

	if Smaxtodate is None:          
		Smaxtodate = S 		       ##### Set Smaxtodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
		
	####### Implement formula from Hull, 11ed., Chapter 26 ########
	
	CallPrice = AnalyticFixedStrikeLookBackCall(S,K,r,sigma,t,T,Smaxtodate)
	
	small_time_step = (T-t)/100

	##### Following is for self-consistency when using the finite element difference for calculating the greeks.	
	if (Smaxtodate-S) < 0.01:
		Smaxtodatecorrection = S + 0.01 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smaxtodatecorrection = Smaxtodate
	
	########### The Greeks using Finite Difference #################

	Delta = ( AnalyticFixedStrikeLookBackCall(S+0.01,K,r,sigma,t,T,Smaxtodatecorrection) - AnalyticFixedStrikeLookBackCall(S-0.01,K,r,sigma,t,T,Smaxtodate) ) / (2*0.01)
	Gamma = ( AnalyticFixedStrikeLookBackCall(S+0.01,K,r,sigma,t,T,Smaxtodatecorrection) - 2*AnalyticFixedStrikeLookBackCall(S,K,r,sigma,t,T,Smaxtodate) + AnalyticFixedStrikeLookBackCall(S-0.01,K,r,sigma,t,T,Smaxtodate))/(0.01**2)
	
	Vega = ( AnalyticFixedStrikeLookBackCall(S,K,r,sigma+0.01,t,T,Smaxtodate) - AnalyticFixedStrikeLookBackCall(S,K,r,sigma-0.01,t,T,Smaxtodate) ) / (2*0.01)
	Theta = -( AnalyticFixedStrikeLookBackCall(S,K,r,sigma,t+small_time_step,T,Smaxtodate) - AnalyticFixedStrikeLookBackCall(S,K,r,sigma,t-small_time_step,T,Smaxtodate) ) / (2*small_time_step)
	Rho = ( AnalyticFixedStrikeLookBackCall(S,K,r+1e-4,sigma,t,T,Smaxtodate) - AnalyticFixedStrikeLookBackCall(S,K,r-1e-4,sigma,t,T,Smaxtodate) ) / (2*1e-4)
	
	return(CallPrice, Delta, Gamma, Vega, Theta, Rho)


############################################################################

def AnalyticFixedStrikeLookBackPutWithGreeks(S,K,r,sigma,t,T,Smintodate=None):	
	'''
	Calculates the Fixed Strike LookBack Call Price (assuming Black Scholes) using the analytic formula.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method
	
	Note: The Payoff is given by maximum( K - Smin  , 0), where Smin is the minimum S over the option lifetime

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	
	
	
	Output is: 
	The Put price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	#################################################################
	#################################################################

	if Smintodate is None:          
		Smintodate = S		##### Set Smintodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
		
	####### Implement formula from Hull, 11ed., Chapter 26 ########
	
	PutPrice = AnalyticFixedStrikeLookBackPut(S,K,r,sigma,t,T,Smintodate)
	
	small_time_step = (T-t)/100

	##### Following is for self-consistency when using the finite element difference for calculating the greeks.	
	if (S-Smintodate) < 0.01:
		Smintodatecorrection = S - 0.01 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smintodatecorrection = Smintodate
	
	########### The Greeks using Finite Difference #################

	Delta = ( AnalyticFixedStrikeLookBackPut(S+0.01,K,r,sigma,t,T,Smintodate) - AnalyticFixedStrikeLookBackPut(S-0.01,K,r,sigma,t,T,Smintodatecorrection) ) / (2*0.01)
	Gamma = ( AnalyticFixedStrikeLookBackPut(S+0.01,K,r,sigma,t,T,Smintodate) - 2*AnalyticFixedStrikeLookBackPut(S,K,r,sigma,t,T,Smintodate) + AnalyticFixedStrikeLookBackPut(S-0.01,K,r,sigma,t,T,Smintodatecorrection))/(0.01**2)
	
	Vega = ( AnalyticFixedStrikeLookBackPut(S,K,r,sigma+0.01,t,T,Smintodate) - AnalyticFixedStrikeLookBackPut(S,K,r,sigma-0.01,t,T,Smintodate) ) / (2*0.01)
	Theta = -( AnalyticFixedStrikeLookBackPut(S,K,r,sigma,t+small_time_step,T,Smintodate) - AnalyticFixedStrikeLookBackPut(S,K,r,sigma,t-small_time_step,T,Smintodate) ) / (2*small_time_step)
	Rho = ( AnalyticFixedStrikeLookBackPut(S,K,r+1e-4,sigma,t,T,Smintodate) - AnalyticFixedStrikeLookBackPut(S,K,r-1e-4,sigma,t,T,Smintodate) ) / (2*1e-4)
	
	return(PutPrice, Delta, Gamma, Vega, Theta, Rho)


############################################################################
############################################################################


def generate_terminal_price_ForGreeks_withMinMaxPrice(S, r, sigma, t, T, n_steps=100, seed=0):

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
	Also outputs the MIN and MAX Prices of the stock over the path, for the central simulation, and the perturbed quantities.
	These MIN and MAX values will be used for the payoff filter of the barrier option.
	'''
	
	### set the random seed #######
	rng = np.random.default_rng(seed)

	### calculate the required time step, unit is in years #####
	time_step = (T-t)/n_steps
	
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
	
	### Minimal and maximum stock prices for barrier options
	
	MinPrice = np.min(stock_price_array)
	MaxPrice = np.max(stock_price_array)

	
	### Finite differences for Delta and Gamma: start with 1 cent greater or smaller price ###
	terminal_price_smaller_S = terminal_price*(S-0.01)/S
	terminal_price_larger_S = terminal_price*(S+0.01)/S 
	
	MinPrice_smaller_S = MinPrice*(S-0.01)/S
	MaxPrice_smaller_S = MaxPrice*(S-0.01)/S
	
	MinPrice_larger_S = MinPrice*(S+0.01)/S
	MaxPrice_larger_S = MaxPrice*(S+0.01)/S

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
	
	MinPrice_smaller_sigma = np.min(stock_price_array_smaller_sigma)
	MaxPrice_smaller_sigma = np.max(stock_price_array_smaller_sigma)
	
	MinPrice_larger_sigma = np.min(stock_price_array_larger_sigma)
	MaxPrice_larger_sigma =	np.max(stock_price_array_larger_sigma)
	
	### Finite differences for t: finish with plus/minus 1 time step differences #############
	
	terminal_price_smaller_t = stock_price_array[-1]*(1+np.sqrt(time_step)*sigma*rng.normal(0, 1)+r*time_step)
	terminal_price_larger_t = stock_price_array[-2]
	
	MinPrice_smaller_t = min(terminal_price_smaller_t, MinPrice)
	MaxPrice_smaller_t = max(terminal_price_smaller_t, MaxPrice)
	
	MinPrice_larger_t = np.min(stock_price_array[0:-1])
	MaxPrice_larger_t = np.max(stock_price_array[0:-1])

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

	MinPrice_smaller_r = np.min(stock_price_array_smaller_r)
	MaxPrice_smaller_r = np.max(stock_price_array_smaller_r)
	
	MinPrice_larger_r = np.min(stock_price_array_larger_r)
	MaxPrice_larger_r = np.max(stock_price_array_larger_r)		
	
	return(terminal_price, terminal_price_smaller_S, terminal_price_larger_S, terminal_price_smaller_sigma, terminal_price_larger_sigma, terminal_price_smaller_t, terminal_price_larger_t, terminal_price_smaller_r, terminal_price_larger_r, MinPrice, MaxPrice, MinPrice_smaller_S, MaxPrice_smaller_S, MinPrice_larger_S, MaxPrice_larger_S, MinPrice_smaller_sigma, MaxPrice_smaller_sigma, MinPrice_larger_sigma, MaxPrice_larger_sigma, MinPrice_smaller_t, MaxPrice_smaller_t, MinPrice_larger_t, MaxPrice_larger_t, MinPrice_smaller_r, MaxPrice_smaller_r,  MinPrice_larger_r, MaxPrice_larger_r)

#####################


def MonteCarloFloatingStrikeLookBackCallWithGreeks(S, r, sigma, t, T, Smintodate=None, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculates the Floating Strike LookBack Call Price (assuming Black Scholes) using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Smintodate is the minimum price of S observed with option active up to time t [default if none entered: Smintodate = S(t), e.g. when t is chosen just as the option comes into existence]
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	BaseSeed is the starting seed for the Seeds input into the Monte-Carlo
	
	Note: The Payoff is given by S(T) - Smin, where Smin is the minimum over the option lifetime
	
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
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	

	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_withMinMaxPrice, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'MinPrice', 'MaxPrice', 'MinPrice_smaller_S', 'MaxPrice_smaller_S', 'MinPrice_larger_S', 'MaxPrice_larger_S', 'MinPrice_smaller_sigma', 'MaxPrice_smaller_sigma', 'MinPrice_larger_sigma', 'MaxPrice_larger_sigma', 'MinPrice_smaller_t', 'MaxPrice_smaller_t', 'MinPrice_larger_t', 'MaxPrice_larger_t', 'MinPrice_smaller_r', 'MaxPrice_smaller_r',  'MinPrice_larger_r', 'MaxPrice_larger_r'])
	
	######
	
	if Smintodate is None:          
		Smintodate = S		##### Set Smintodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
	
	
	##### Following is for self-consistency when using the finite element difference for calculating the greeks.
	if (S-Smintodate) < 0.01:
		Smintodatecorrection = S - 0.01 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smintodatecorrection = Smintodate	
	
	######## Broadie-Glasserman-Kou (BGK) Continuity Correction
	time_step = (T-t)/n_steps
	BGKcorrection = np.exp(-0.5826*sigma*np.sqrt(time_step)) 
	Smintodatecorrection = Smintodatecorrection*BGKcorrection
	Smintodate = Smintodate*BGKcorrection
	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = full_terminal_price_df['terminal_price'].to_numpy()
	MinPrice_array = full_terminal_price_df['MinPrice'].to_numpy()
	MinPrice_array = MinPrice_array*BGKcorrection
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(n_simulations)
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( terminal_price_array - MinPrice_array, terminal_price_array - Smintodate )
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = full_terminal_price_df['terminal_price_smaller_S'].to_numpy()
	terminal_price_array_larger_S = full_terminal_price_df['terminal_price_larger_S'].to_numpy()
	
	MinPrice_array_smaller_S = full_terminal_price_df['MinPrice_smaller_S'].to_numpy()
	MinPrice_array_larger_S = full_terminal_price_df['MinPrice_larger_S'].to_numpy()
	MinPrice_array_smaller_S = MinPrice_array_smaller_S*BGKcorrection
	MinPrice_array_larger_S = MinPrice_array_larger_S*BGKcorrection		
	
	payoff_array_smaller_S = np.maximum( terminal_price_array_smaller_S - MinPrice_array_smaller_S, terminal_price_array_smaller_S - Smintodatecorrection )
	payoff_array_larger_S = np.maximum( terminal_price_array_larger_S - MinPrice_array_larger_S, terminal_price_array_larger_S - Smintodatecorrection )

	payoff_array_withmintodatecorrection = np.maximum( terminal_price_array - MinPrice_array, terminal_price_array - Smintodatecorrection )
	
	option_value_array_withmintodatecorrection = np.exp(-r*(T-t))*payoff_array_withmintodatecorrection		
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	
	delta_value = np.mean(delta_array)
	
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array_withmintodatecorrection+option_value_array_smaller_S)/(0.01**2)
	
	gamma_value = np.mean(gamma_array)
	
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy()
	terminal_price_array_larger_sigma = full_terminal_price_df['terminal_price_larger_sigma'].to_numpy()
	
	MinPrice_array_smaller_sigma = full_terminal_price_df['MinPrice_smaller_sigma'].to_numpy()
	MinPrice_array_larger_sigma = full_terminal_price_df['MinPrice_larger_sigma'].to_numpy()
	MinPrice_array_smaller_sigma = MinPrice_array_smaller_sigma*BGKcorrection
	MinPrice_array_larger_sigma = MinPrice_array_larger_sigma*BGKcorrection
	
	payoff_array_smaller_sigma =  np.maximum( terminal_price_array_smaller_sigma - MinPrice_array_smaller_sigma, terminal_price_array_smaller_sigma - Smintodate )
	payoff_array_larger_sigma = np.maximum( terminal_price_array_larger_sigma - MinPrice_array_larger_sigma, terminal_price_array_larger_sigma - Smintodate )
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	
	vega_value = np.mean(vega_array)
	
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = full_terminal_price_df['terminal_price_smaller_t'].to_numpy()
	terminal_price_array_larger_t = full_terminal_price_df['terminal_price_larger_t'].to_numpy()

	MinPrice_array_smaller_t = full_terminal_price_df['MinPrice_smaller_t'].to_numpy()
	MinPrice_array_larger_t = full_terminal_price_df['MinPrice_larger_t'].to_numpy()
	MinPrice_array_smaller_t = MinPrice_array_smaller_t*BGKcorrection
	MinPrice_array_larger_t = MinPrice_array_larger_t*BGKcorrection

	payoff_array_smaller_t =  np.maximum( terminal_price_array_smaller_t - MinPrice_array_smaller_t, terminal_price_array_smaller_t - Smintodate )
	payoff_array_larger_t = np.maximum( terminal_price_array_larger_t - MinPrice_array_larger_t, terminal_price_array_larger_t - Smintodate )

	
	time_step = (T-t)/n_steps
	
	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	
	theta_value = np.mean(theta_array)
	
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	
	terminal_price_array_smaller_r = full_terminal_price_df['terminal_price_smaller_r'].to_numpy()
	terminal_price_array_larger_r = full_terminal_price_df['terminal_price_larger_r'].to_numpy()
	
	MinPrice_array_smaller_r = full_terminal_price_df['MinPrice_smaller_r'].to_numpy()
	MinPrice_array_larger_r = full_terminal_price_df['MinPrice_larger_r'].to_numpy()
	MinPrice_array_smaller_r = MinPrice_array_smaller_r*BGKcorrection
	MinPrice_array_larger_r = MinPrice_array_larger_r*BGKcorrection

	payoff_array_smaller_r =  np.maximum( terminal_price_array_smaller_r - MinPrice_array_smaller_r, terminal_price_array_smaller_r - Smintodate )
	payoff_array_larger_r = np.maximum( terminal_price_array_larger_r - MinPrice_array_larger_r, terminal_price_array_larger_r - Smintodate )
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	
	rho_value = np.mean(rho_array)
	
	rho_StandardError = stats.sem(rho_array)	
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)

############################################################################


def MonteCarloFloatingStrikeLookBackPutWithGreeks(S, r, sigma, t, T, Smaxtodate=None, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculates the Floating Strike LookBack Put Price (assuming Black Scholes) using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Smaxtodate is the maximum price of S observed with option active up to time t [default if none entered: Smintodate = S(t), e.g. when t is chosen just as the option comes into existence]
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	BaseSeed is the starting seed for the Seeds input into the Monte-Carlo
	
	Note: The Payoff is given by Smax-S(T), where Smax is the maximum over the option lifetime
	
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
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	

	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_withMinMaxPrice, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'MinPrice', 'MaxPrice', 'MinPrice_smaller_S', 'MaxPrice_smaller_S', 'MinPrice_larger_S', 'MaxPrice_larger_S', 'MinPrice_smaller_sigma', 'MaxPrice_smaller_sigma', 'MinPrice_larger_sigma', 'MaxPrice_larger_sigma', 'MinPrice_smaller_t', 'MaxPrice_smaller_t', 'MinPrice_larger_t', 'MaxPrice_larger_t', 'MinPrice_smaller_r', 'MaxPrice_smaller_r',  'MinPrice_larger_r', 'MaxPrice_larger_r'])
	
	######
	
	if Smaxtodate is None:          
		Smaxtodate = S 		       ##### Set Smaxtodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
		
	#######
	
	##### Following is for self-consistency when using the finite element difference for calculating the greeks.	
	if (Smaxtodate-S) < 0.01:
		Smaxtodatecorrection = S + 0.01 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smaxtodatecorrection = Smaxtodate
		
	######## Broadie-Glasserman-Kou (BGK) Continuity Correction
	time_step = (T-t)/n_steps
	BGKcorrection = np.exp(0.5826*sigma*np.sqrt(time_step)) 
	Smaxtodatecorrection = Smaxtodatecorrection*BGKcorrection
	Smaxtodate = Smaxtodate*BGKcorrection
	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = full_terminal_price_df['terminal_price'].to_numpy()
	MaxPrice_array = full_terminal_price_df['MaxPrice'].to_numpy()
	MaxPrice_array = MaxPrice_array*BGKcorrection	
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(n_simulations)
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( MaxPrice_array - terminal_price_array  , Smaxtodate - terminal_price_array  )
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = full_terminal_price_df['terminal_price_smaller_S'].to_numpy()
	terminal_price_array_larger_S = full_terminal_price_df['terminal_price_larger_S'].to_numpy()
	
	MaxPrice_array_smaller_S = full_terminal_price_df['MaxPrice_smaller_S'].to_numpy()
	MaxPrice_array_larger_S = full_terminal_price_df['MaxPrice_larger_S'].to_numpy()
	MaxPrice_array_smaller_S = MaxPrice_array_smaller_S*BGKcorrection
	MaxPrice_array_larger_S = MaxPrice_array_larger_S*BGKcorrection			
	
	payoff_array_smaller_S = np.maximum( MaxPrice_array_smaller_S - terminal_price_array_smaller_S , Smaxtodatecorrection - terminal_price_array_smaller_S  )
	payoff_array_larger_S = np.maximum( MaxPrice_array_larger_S - terminal_price_array_larger_S ,  Smaxtodatecorrection - terminal_price_array_larger_S  )
	payoff_array_withmaxtodatecorrection = np.maximum( MaxPrice_array - terminal_price_array  , Smaxtodatecorrection - terminal_price_array  )
	
	
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	option_value_array_withmaxtodatecorrection = np.exp(-r*(T-t))*payoff_array_withmaxtodatecorrection
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	
	delta_value = np.mean(delta_array)
	
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array_withmaxtodatecorrection+option_value_array_smaller_S)/(0.01**2)
	
	gamma_value = np.mean(gamma_array)
	
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy()
	terminal_price_array_larger_sigma = full_terminal_price_df['terminal_price_larger_sigma'].to_numpy()
	
	MaxPrice_array_smaller_sigma = full_terminal_price_df['MaxPrice_smaller_sigma'].to_numpy()
	MaxPrice_array_larger_sigma = full_terminal_price_df['MaxPrice_larger_sigma'].to_numpy()
	MaxPrice_array_smaller_sigma = MaxPrice_array_smaller_sigma*BGKcorrection
	MaxPrice_array_larger_sigma = MaxPrice_array_larger_sigma*BGKcorrection		
	
	payoff_array_smaller_sigma =  np.maximum(  MaxPrice_array_smaller_sigma - terminal_price_array_smaller_sigma ,  Smaxtodate - terminal_price_array_smaller_sigma  )
	payoff_array_larger_sigma = np.maximum(  MaxPrice_array_larger_sigma - terminal_price_array_larger_sigma ,   Smaxtodate - terminal_price_array_larger_sigma )
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	
	vega_value = np.mean(vega_array)
	
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = full_terminal_price_df['terminal_price_smaller_t'].to_numpy()
	terminal_price_array_larger_t = full_terminal_price_df['terminal_price_larger_t'].to_numpy()

	MaxPrice_array_smaller_t = full_terminal_price_df['MaxPrice_smaller_t'].to_numpy()
	MaxPrice_array_larger_t = full_terminal_price_df['MaxPrice_larger_t'].to_numpy()
	MaxPrice_array_smaller_t = MaxPrice_array_smaller_t*BGKcorrection
	MaxPrice_array_larger_t = MaxPrice_array_larger_t*BGKcorrection	

	payoff_array_smaller_t =  np.maximum( MaxPrice_array_smaller_t - terminal_price_array_smaller_t ,  Smaxtodate - terminal_price_array_smaller_t  )
	payoff_array_larger_t = np.maximum( MaxPrice_array_larger_t - terminal_price_array_larger_t  ,  Smaxtodate - terminal_price_array_larger_t  )

	
	time_step = (T-t)/n_steps
	
	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	
	theta_value = np.mean(theta_array)
	
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	
	terminal_price_array_smaller_r = full_terminal_price_df['terminal_price_smaller_r'].to_numpy()
	terminal_price_array_larger_r = full_terminal_price_df['terminal_price_larger_r'].to_numpy()
	
	MaxPrice_array_smaller_r = full_terminal_price_df['MaxPrice_smaller_r'].to_numpy()
	MaxPrice_array_larger_r = full_terminal_price_df['MaxPrice_larger_r'].to_numpy()
	MaxPrice_array_smaller_r = MaxPrice_array_smaller_r*BGKcorrection
	MaxPrice_array_larger_r = MaxPrice_array_larger_r*BGKcorrection		

	payoff_array_smaller_r =  np.maximum( MaxPrice_array_smaller_r - terminal_price_array_smaller_r , Smaxtodate - terminal_price_array_smaller_r   )
	payoff_array_larger_r = np.maximum( MaxPrice_array_larger_r - terminal_price_array_larger_r ,  Smaxtodate - terminal_price_array_larger_r  )
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	
	rho_value = np.mean(rho_array)
	
	rho_StandardError = stats.sem(rho_array)	

	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)


############################################################################


def MonteCarloFixedStrikeLookBackCallWithGreeks(S, K, r, sigma, t, T, Smaxtodate=None, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):
	'''
	Calculates the Fixed Strike LookBack Call Price (assuming Black Scholes) using Monte-Carlo.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the Monte-Carlo for the price and the finite difference method
	
	Note: The Payoff is given by maximum( Smax - K , 0), where Smax is the maximum S over the option lifetime

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Smaxtodate is the maximum price of S observed with option active up to time t [default if none entered: Smaxtodate = S(t), e.g. when t is chosen just as the option comes into existence]
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	BaseSeed is the starting seed for the Seeds input into the Monte-Carlo
	
	
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
	####################################
	####################################
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	

	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_withMinMaxPrice, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'MinPrice', 'MaxPrice', 'MinPrice_smaller_S', 'MaxPrice_smaller_S', 'MinPrice_larger_S', 'MaxPrice_larger_S', 'MinPrice_smaller_sigma', 'MaxPrice_smaller_sigma', 'MinPrice_larger_sigma', 'MaxPrice_larger_sigma', 'MinPrice_smaller_t', 'MaxPrice_smaller_t', 'MinPrice_larger_t', 'MaxPrice_larger_t', 'MinPrice_smaller_r', 'MaxPrice_smaller_r',  'MinPrice_larger_r', 'MaxPrice_larger_r'])
	
	######
	
	if Smaxtodate is None:          
		Smaxtodate = S 		       ##### Set Smaxtodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
		
	#######
	
	##### Following is for self-consistency when using the finite element difference for calculating the greeks.	
	if (Smaxtodate-S) < 0.01:
		Smaxtodatecorrection = S + 0.01 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smaxtodatecorrection = Smaxtodate
	
	######## Broadie-Glasserman-Kou (BGK) Continuity Correction
	time_step = (T-t)/n_steps
	BGKcorrection = np.exp(0.5826*sigma*np.sqrt(time_step)) 
	Smaxtodatecorrection = Smaxtodatecorrection*BGKcorrection
	Smaxtodate = Smaxtodate*BGKcorrection	
	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = full_terminal_price_df['terminal_price'].to_numpy()
	MaxPrice_array = full_terminal_price_df['MaxPrice'].to_numpy()
	MaxPrice_array = MaxPrice_array*BGKcorrection
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(n_simulations)
	

	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( MaxPrice_array - K  , Smaxtodate - K)
	payoff_array = np.maximum( payoff_array, 0)
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = full_terminal_price_df['terminal_price_smaller_S'].to_numpy()
	terminal_price_array_larger_S = full_terminal_price_df['terminal_price_larger_S'].to_numpy()
	
	MaxPrice_array_smaller_S = full_terminal_price_df['MaxPrice_smaller_S'].to_numpy()
	MaxPrice_array_larger_S = full_terminal_price_df['MaxPrice_larger_S'].to_numpy()
	MaxPrice_array_smaller_S = MaxPrice_array_smaller_S*BGKcorrection 
	MaxPrice_array_larger_S = MaxPrice_array_larger_S*BGKcorrection 		
	
	payoff_array_smaller_S = np.maximum( MaxPrice_array_smaller_S - K , Smaxtodatecorrection - K)
	payoff_array_smaller_S = np.maximum( payoff_array_smaller_S , 0 )
	payoff_array_larger_S = np.maximum( MaxPrice_array_larger_S - K ,  Smaxtodatecorrection - K)
	payoff_array_larger_S = np.maximum( payoff_array_larger_S , 0 )
	payoff_array_withmaxtodatecorrection = np.maximum( MaxPrice_array - K  , Smaxtodatecorrection - K)
	payoff_array_withmaxtodatecorrection = np.maximum( payoff_array_withmaxtodatecorrection, 0)
		
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	option_value_array_withmaxtodatecorrection = np.exp(-r*(T-t))*payoff_array_withmaxtodatecorrection
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	
	delta_value = np.mean(delta_array)
	
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array_withmaxtodatecorrection+option_value_array_smaller_S)/(0.01**2)
	
	gamma_value = np.mean(gamma_array)
	
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy()
	terminal_price_array_larger_sigma = full_terminal_price_df['terminal_price_larger_sigma'].to_numpy()
	
	MaxPrice_array_smaller_sigma = full_terminal_price_df['MaxPrice_smaller_sigma'].to_numpy()
	MaxPrice_array_larger_sigma = full_terminal_price_df['MaxPrice_larger_sigma'].to_numpy()
	MaxPrice_array_smaller_sigma = MaxPrice_array_smaller_sigma*BGKcorrection 
	MaxPrice_array_larger_sigma = MaxPrice_array_larger_sigma*BGKcorrection 
	
	payoff_array_smaller_sigma =  np.maximum(  MaxPrice_array_smaller_sigma - K ,  Smaxtodate - K )
	payoff_array_smaller_sigma =  np.maximum( payoff_array_smaller_sigma, 0 )
	payoff_array_larger_sigma = np.maximum(  MaxPrice_array_larger_sigma - K ,   Smaxtodate - K)
	payoff_array_larger_sigma = np.maximum( payoff_array_larger_sigma, 0)
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	
	vega_value = np.mean(vega_array)
	
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = full_terminal_price_df['terminal_price_smaller_t'].to_numpy()
	terminal_price_array_larger_t = full_terminal_price_df['terminal_price_larger_t'].to_numpy()

	MaxPrice_array_smaller_t = full_terminal_price_df['MaxPrice_smaller_t'].to_numpy()
	MaxPrice_array_larger_t = full_terminal_price_df['MaxPrice_larger_t'].to_numpy()
	MaxPrice_array_smaller_t = MaxPrice_array_smaller_t*BGKcorrection 
	MaxPrice_array_larger_t = MaxPrice_array_larger_t*BGKcorrection 

	payoff_array_smaller_t =  np.maximum( MaxPrice_array_smaller_t - K ,  Smaxtodate - K  )
	payoff_array_smaller_t = np.maximum( payoff_array_smaller_t, 0) 
	payoff_array_larger_t = np.maximum( MaxPrice_array_larger_t - K  ,  Smaxtodate - K  )
	payoff_array_larger_t = np.maximum( payoff_array_larger_t, 0 )
	
	time_step = (T-t)/n_steps
	
	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	
	theta_value = np.mean(theta_array)
	
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	terminal_price_array_smaller_r = full_terminal_price_df['terminal_price_smaller_r'].to_numpy()
	terminal_price_array_larger_r = full_terminal_price_df['terminal_price_larger_r'].to_numpy()
	
	MaxPrice_array_smaller_r = full_terminal_price_df['MaxPrice_smaller_r'].to_numpy()
	MaxPrice_array_larger_r = full_terminal_price_df['MaxPrice_larger_r'].to_numpy()
	MaxPrice_array_smaller_r = MaxPrice_array_smaller_r*BGKcorrection 
	MaxPrice_array_larger_r = MaxPrice_array_larger_r*BGKcorrection 

	payoff_array_smaller_r =  np.maximum( MaxPrice_array_smaller_r - K , Smaxtodate - K  )
	payoff_array_smaller_r = np.maximum( payoff_array_smaller_r , 0 ) 
	payoff_array_larger_r = np.maximum( MaxPrice_array_larger_r - K ,  Smaxtodate - K  )
	payoff_array_larger_r = np.maximum( payoff_array_larger_r , 0 )
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	
	rho_value = np.mean(rho_array)
	
	rho_StandardError = stats.sem(rho_array)	

	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)


############################################################################


def MonteCarloFixedStrikeLookBackPutWithGreeks(S, K, r, sigma, t, T, Smintodate=None, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):
	'''
	Calculates the Fixed Strike LookBack Put Price (assuming Black Scholes) using Monte-Carlo.
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the Monte-Carlo for the price and the finite difference method
	
	Note: The Payoff is given by maximum( K - Smin , 0), where Smin is the minimum S over the option lifetime

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	Smintodate is the minimum price of S observed with option active up to time t [default if none entered: Smintodate = S(t), e.g. when t is chosen just as the option comes into existence]
	n_simulations is the number of simulations to run (defaults to 250000)
	n_steps is the number of steps to use (defaults to 100)
	BaseSeed is the starting seed for the Seeds input into the Monte-Carlo

	
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
	####################################
	####################################
	###########################
	###########################
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	

	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_withMinMaxPrice, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'MinPrice', 'MaxPrice', 'MinPrice_smaller_S', 'MaxPrice_smaller_S', 'MinPrice_larger_S', 'MaxPrice_larger_S', 'MinPrice_smaller_sigma', 'MaxPrice_smaller_sigma', 'MinPrice_larger_sigma', 'MaxPrice_larger_sigma', 'MinPrice_smaller_t', 'MaxPrice_smaller_t', 'MinPrice_larger_t', 'MaxPrice_larger_t', 'MinPrice_smaller_r', 'MaxPrice_smaller_r',  'MinPrice_larger_r', 'MaxPrice_larger_r'])
	
	######
	
	if Smintodate is None:          
		Smintodate = S		##### Set Smintodate to S(t) if no value is entered (say, lookback has just been initiated)
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
	
	
	##### Following is for self-consistency when using the finite element difference for calculating the greeks.
	if (S-Smintodate) < 0.01:
		Smintodatecorrection = S - 0.01 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smintodatecorrection = Smintodate	

	######## Broadie-Glasserman-Kou (BGK) Continuity Correction
	time_step = (T-t)/n_steps
	BGKcorrection = np.exp(-0.5826*sigma*np.sqrt(time_step)) 
	Smintodatecorrection = Smintodatecorrection*BGKcorrection
	Smintodate = Smintodate*BGKcorrection
	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = full_terminal_price_df['terminal_price'].to_numpy()
	MinPrice_array = full_terminal_price_df['MinPrice'].to_numpy()
	MinPrice_array = MinPrice_array*BGKcorrection	
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(n_simulations)
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(  K -  MinPrice_array ,  K - Smintodate )
	payoff_array = np.maximum(payoff_array , 0)
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = full_terminal_price_df['terminal_price_smaller_S'].to_numpy()
	terminal_price_array_larger_S = full_terminal_price_df['terminal_price_larger_S'].to_numpy()
	
	MinPrice_array_smaller_S = full_terminal_price_df['MinPrice_smaller_S'].to_numpy()
	MinPrice_array_larger_S = full_terminal_price_df['MinPrice_larger_S'].to_numpy()
	MinPrice_array_smaller_S = MinPrice_array_smaller_S*BGKcorrection
	MinPrice_array_larger_S = MinPrice_array_larger_S*BGKcorrection	
	
	payoff_array_smaller_S = np.maximum( K - MinPrice_array_smaller_S  ,  K -Smintodatecorrection )
	payoff_array_smaller_S = np.maximum(payoff_array_smaller_S , 0)
	payoff_array_larger_S = np.maximum(  K - MinPrice_array_larger_S ,  K - Smintodatecorrection  )
	payoff_array_larger_S = np.maximum( payoff_array_larger_S , 0)
	payoff_array_withsmintodatecorrection = np.maximum(  K -  MinPrice_array ,  K - Smintodatecorrection )
	payoff_array_withsmintodatecorrection = np.maximum(payoff_array_withsmintodatecorrection , 0)
	
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	option_value_array_withsmintodatecorrection = np.exp(-r*(T-t))*payoff_array_withsmintodatecorrection
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	
	delta_value = np.mean(delta_array)
	
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array_withsmintodatecorrection+option_value_array_smaller_S)/(0.01**2)
	
	gamma_value = np.mean(gamma_array)
	
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy()
	terminal_price_array_larger_sigma = full_terminal_price_df['terminal_price_larger_sigma'].to_numpy()
	
	MinPrice_array_smaller_sigma = full_terminal_price_df['MinPrice_smaller_sigma'].to_numpy()
	MinPrice_array_larger_sigma = full_terminal_price_df['MinPrice_larger_sigma'].to_numpy()
	MinPrice_array_smaller_sigma = MinPrice_array_smaller_sigma*BGKcorrection
	MinPrice_array_larger_sigma = MinPrice_array_larger_sigma*BGKcorrection		
	
	payoff_array_smaller_sigma =  np.maximum( K - MinPrice_array_smaller_sigma   ,   K - Smintodate  )
	payoff_array_smaller_sigma =  np.maximum( payoff_array_smaller_sigma , 0 )
	payoff_array_larger_sigma = np.maximum(  K - MinPrice_array_larger_sigma  ,   K - Smintodate  )
	payoff_array_larger_sigma = np.maximum( payoff_array_larger_sigma , 0 )
	
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	
	vega_value = np.mean(vega_array)
	
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = full_terminal_price_df['terminal_price_smaller_t'].to_numpy()
	terminal_price_array_larger_t = full_terminal_price_df['terminal_price_larger_t'].to_numpy()

	MinPrice_array_smaller_t = full_terminal_price_df['MinPrice_smaller_t'].to_numpy()
	MinPrice_array_larger_t = full_terminal_price_df['MinPrice_larger_t'].to_numpy()
	MinPrice_array_smaller_t = MinPrice_array_smaller_t*BGKcorrection
	MinPrice_array_larger_t = MinPrice_array_larger_t*BGKcorrection	

	payoff_array_smaller_t =  np.maximum(  K - MinPrice_array_smaller_t ,   K  - Smintodate )
	payoff_array_smaller_t =  np.maximum(payoff_array_smaller_t , 0)
	payoff_array_larger_t = np.maximum(  K - MinPrice_array_larger_t ,  K - Smintodate  )
	payoff_array_larger_t = np.maximum( payoff_array_larger_t , 0 )
	
	time_step = (T-t)/n_steps
	
	#### time increment for finite difference is 1 time step ####
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	
	theta_value = np.mean(theta_array)
	
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	
	terminal_price_array_smaller_r = full_terminal_price_df['terminal_price_smaller_r'].to_numpy()
	terminal_price_array_larger_r = full_terminal_price_df['terminal_price_larger_r'].to_numpy()
	
	MinPrice_array_smaller_r = full_terminal_price_df['MinPrice_smaller_r'].to_numpy()
	MinPrice_array_larger_r = full_terminal_price_df['MinPrice_larger_r'].to_numpy()
	MinPrice_array_smaller_r = MinPrice_array_smaller_r*BGKcorrection
	MinPrice_array_larger_r = MinPrice_array_larger_r*BGKcorrection	

	payoff_array_smaller_r =  np.maximum( K - MinPrice_array_smaller_r ,   K - Smintodate  )
	payoff_array_smaller_r =  np.maximum( payoff_array_smaller_r , 0 )
	payoff_array_larger_r = np.maximum(   K - MinPrice_array_larger_r,   K - Smintodate  )
	payoff_array_larger_r = np.maximum( payoff_array_larger_r , 0 )
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	
	rho_value = np.mean(rho_array)
	
	rho_StandardError = stats.sem(rho_array)	
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)


############################################################################
############################################################################


def main():
	print("Number of cpus : ", mp.cpu_count())
	pool = Pool(processes=(mp.cpu_count() - 1))
	
	#######
	
	print("Testing some Lookback Options pricing, using Analytic results, and Monte-Carlo. Check main() in the LookBack.py file for the input values")  

	print(AnalyticFloatingStrikeLookBackCall(80,0.1,0.5,1,1.25))
	print(AnalyticFloatingStrikeLookBackCall(80,0.1,0.5,1,1.25,80))
	print(AnalyticFloatingStrikeLookBackCall(80,0.1,0.5,1,1.25,70))
	
	try:
		AnalyticFloatingStrikeLookBackCall(80,0.1,0.5,1,1.25,85)
	except ValueError as e:
    		print(f"Caught an error: {e}")

	
	print(AnalyticFloatingStrikeLookBackCallWithGreeks(80,0.1,0.5,1,1.25))
	print(AnalyticFloatingStrikeLookBackCallWithGreeks(80,0.1,0.5,1,1.25,80))
	print(AnalyticFloatingStrikeLookBackCallWithGreeks(80,0.1,0.5,1,1.25,70))
	
	try:
		AnalyticFloatingStrikeLookBackCallWithGreeks(80,0.1,0.5,1,1.25,85)
	except ValueError as e:
    		print(f"Caught an error: {e}")
    		
	print(MonteCarloFloatingStrikeLookBackCallWithGreeks(80,0.1,0.5,1,1.25))
	print(MonteCarloFloatingStrikeLookBackCallWithGreeks(80,0.1,0.5,1,1.25,80))
	print(MonteCarloFloatingStrikeLookBackCallWithGreeks(80,0.1,0.5,1,1.25,70))
	
	try:
		MonteCarloFloatingStrikeLookBackCallWithGreeks(80,0.1,0.5,1,1.25,85)
	except ValueError as e:
    		print(f"Caught an error: {e}")
	
	#######

	print(AnalyticFloatingStrikeLookBackPut(80,0.1,0.5,1,1.25))
	print(AnalyticFloatingStrikeLookBackPut(80,0.1,0.5,1,1.25,80))
	print(AnalyticFloatingStrikeLookBackPut(80,0.1,0.5,1,1.25,105))
	try:
    		AnalyticFloatingStrikeLookBackPut(80,0.1,0.5,1,1.25,75)
	except ValueError as e:
    		print(f"Caught an error: {e}")
	
	print(AnalyticFloatingStrikeLookBackPutWithGreeks(80,0.1,0.5,1,1.25))
	print(AnalyticFloatingStrikeLookBackPutWithGreeks(80,0.1,0.5,1,1.25,80))
	print(AnalyticFloatingStrikeLookBackPutWithGreeks(80,0.1,0.5,1,1.25,105))
	try:
    		AnalyticFloatingStrikeLookBackPutWithGreeks(80,0.1,0.5,1,1.25,75)
	except ValueError as e:
    		print(f"Caught an error: {e}")
    		
	print(MonteCarloFloatingStrikeLookBackPutWithGreeks(80,0.1,0.5,1,1.25))
	print(MonteCarloFloatingStrikeLookBackPutWithGreeks(80,0.1,0.5,1,1.25,80))
	print(MonteCarloFloatingStrikeLookBackPutWithGreeks(80,0.1,0.5,1,1.25,105))
	try:
    		MonteCarloFloatingStrikeLookBackPutWithGreeks(80,0.1,0.5,1,1.25,75)
	except ValueError as e:
    		print(f"Caught an error: {e}")
    	
    	#######

	print(AnalyticFixedStrikeLookBackCall(80,85,0.1,0.5,1,1.25))
	print(AnalyticFixedStrikeLookBackCall(80,85,0.1,0.5,1,1.25,80))
	print(AnalyticFixedStrikeLookBackCall(80,85,0.1,0.5,1,1.25,105))
	try:
    		AnalyticFixedStrikeLookBackCall(80,85,0.1,0.5,1,1.25,75)
	except ValueError as e:
    		print(f"Caught an error: {e}")
    		
	print(AnalyticFixedStrikeLookBackCallWithGreeks(80,85,0.1,0.5,1,1.25))
	print(AnalyticFixedStrikeLookBackCallWithGreeks(80,85,0.1,0.5,1,1.25,80))
	print(AnalyticFixedStrikeLookBackCallWithGreeks(80,85,0.1,0.5,1,1.25,105))
	try:
    		AnalyticFixedStrikeLookBackCallWithGreeks(80,85,0.1,0.5,1,1.25,75)
	except ValueError as e:
    		print(f"Caught an error: {e}")

	print(MonteCarloFixedStrikeLookBackCallWithGreeks(80,85,0.1,0.5,1,1.25))
	print(MonteCarloFixedStrikeLookBackCallWithGreeks(80,85,0.1,0.5,1,1.25,80))
	print(MonteCarloFixedStrikeLookBackCallWithGreeks(80,85,0.1,0.5,1,1.25,105))
	try:
    		MonteCarloFixedStrikeLookBackCallWithGreeks(80,85,0.1,0.5,1,1.25,75)
	except ValueError as e:
    		print(f"Caught an error: {e}")

	#######
	
	print(AnalyticFixedStrikeLookBackPut(80,85,0.1,0.5,1,1.25))
	print(AnalyticFixedStrikeLookBackPut(80,85,0.1,0.5,1,1.25,80))
	print(AnalyticFixedStrikeLookBackPut(80,85,0.1,0.5,1,1.25,70))
	try:
		AnalyticFixedStrikeLookBackPut(80,85,0.1,0.5,1,1.25,85)
	except ValueError as e:
		print(f"Caught an error: {e}")	
	
	print(AnalyticFixedStrikeLookBackPutWithGreeks(80,85,0.1,0.5,1,1.25))
	print(AnalyticFixedStrikeLookBackPutWithGreeks(80,85,0.1,0.5,1,1.25,80))
	print(AnalyticFixedStrikeLookBackPutWithGreeks(80,85,0.1,0.5,1,1.25,70))
	try:
		AnalyticFixedStrikeLookBackPutWithGreeks(80,85,0.1,0.5,1,1.25,85)
	except ValueError as e:
		print(f"Caught an error: {e}")

	print(MonteCarloFixedStrikeLookBackPutWithGreeks(80,85,0.1,0.5,1,1.25))
	print(MonteCarloFixedStrikeLookBackPutWithGreeks(80,85,0.1,0.5,1,1.25,80))
	print(MonteCarloFixedStrikeLookBackPutWithGreeks(80,85,0.1,0.5,1,1.25,70))
	try:
		MonteCarloFixedStrikeLookBackPutWithGreeks(80,85,0.1,0.5,1,1.25,85)
	except ValueError as e:
		print(f"Caught an error: {e}")
		
	############################
    	############################
	

if __name__ == "__main__":
	main()	    		
