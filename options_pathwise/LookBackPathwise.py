import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

###

'''
GIVES ANALYTIC AND MONTE-CARLO IMPLEMENTATION OF LOOKBACK OPTIONS 
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
	'''
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
	'''
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
	'''
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
	'''
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
	'''
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
	'''	
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
	'''
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
	'''
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
	'''
	##### Following is for self-consistency when using the finite element difference for calculating the greeks.	
	if (S-Smintodate) < 0.001:
		Smintodatecorrection = S - 0.001 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smintodatecorrection = Smintodate
	'''
	Smintodatecorrection = Smintodate
	
	### The Greeks ####
	Delta = ( AnalyticFloatingStrikeLookBackCall(S+0.001,r,sigma,t,T,Smintodate) - AnalyticFloatingStrikeLookBackCall(S-0.001,r,sigma,t,T,Smintodatecorrection) ) / (2*0.001)
	Gamma = ( AnalyticFloatingStrikeLookBackCall(S+0.001,r,sigma,t,T,Smintodate) - 2*AnalyticFloatingStrikeLookBackCall(S,r,sigma,t,T,Smintodate) + AnalyticFloatingStrikeLookBackCall(S-0.001,r,sigma,t,T,Smintodatecorrection))/(0.001**2)
	
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
	'''
	##### Following is for self-consistency when using the finite element difference for calculating the greeks.	
	if (Smaxtodate-S) < 0.001:
		Smaxtodatecorrection = S + 0.001 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smaxtodatecorrection = Smaxtodate
	'''
	Smaxtodatecorrection = Smaxtodate
	### The Greeks ####
	Delta = ( AnalyticFloatingStrikeLookBackPut(S+0.001,r,sigma,t,T,Smaxtodatecorrection) - AnalyticFloatingStrikeLookBackPut(S-0.001,r,sigma,t,T,Smaxtodate) ) / (2*0.001)
	Gamma = ( AnalyticFloatingStrikeLookBackPut(S+0.001,r,sigma,t,T,Smaxtodatecorrection) - 2*AnalyticFloatingStrikeLookBackPut(S,r,sigma,t,T,Smaxtodate) + AnalyticFloatingStrikeLookBackPut(S-0.001,r,sigma,t,T,Smaxtodate))/(0.001**2)
	
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
	'''
	if (Smaxtodate-S) < 0.001:
		Smaxtodatecorrection = S + 0.001 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smaxtodatecorrection = Smaxtodate
	'''
	Smaxtodatecorrection = Smaxtodate	
	########### The Greeks using Finite Difference #################

	Delta = ( AnalyticFixedStrikeLookBackCall(S+0.001,K,r,sigma,t,T,Smaxtodatecorrection) - AnalyticFixedStrikeLookBackCall(S-0.001,K,r,sigma,t,T,Smaxtodate) ) / (2*0.001)
	Gamma = ( AnalyticFixedStrikeLookBackCall(S+0.001,K,r,sigma,t,T,Smaxtodatecorrection) - 2*AnalyticFixedStrikeLookBackCall(S,K,r,sigma,t,T,Smaxtodate) + AnalyticFixedStrikeLookBackCall(S-0.001,K,r,sigma,t,T,Smaxtodate))/(0.001**2)
	
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
	'''
	if (S-Smintodate) < 0.001:
		Smintodatecorrection = S - 0.001 #### To avoid value error in the case of Delta and Gamma with Smintodate = S
	else:
		Smintodatecorrection = Smintodate
	'''
		
	Smintodatecorrection = Smintodate	
	########### The Greeks using Finite Difference #################

	Delta = ( AnalyticFixedStrikeLookBackPut(S+0.001,K,r,sigma,t,T,Smintodate) - AnalyticFixedStrikeLookBackPut(S-0.001,K,r,sigma,t,T,Smintodatecorrection) ) / (2*0.001)
	Gamma = ( AnalyticFixedStrikeLookBackPut(S+0.001,K,r,sigma,t,T,Smintodate) - 2*AnalyticFixedStrikeLookBackPut(S,K,r,sigma,t,T,Smintodate) + AnalyticFixedStrikeLookBackPut(S-0.001,K,r,sigma,t,T,Smintodatecorrection))/(0.001**2)
	
	Vega = ( AnalyticFixedStrikeLookBackPut(S,K,r,sigma+0.01,t,T,Smintodate) - AnalyticFixedStrikeLookBackPut(S,K,r,sigma-0.01,t,T,Smintodate) ) / (2*0.01)
	Theta = -( AnalyticFixedStrikeLookBackPut(S,K,r,sigma,t+small_time_step,T,Smintodate) - AnalyticFixedStrikeLookBackPut(S,K,r,sigma,t-small_time_step,T,Smintodate) ) / (2*small_time_step)
	Rho = ( AnalyticFixedStrikeLookBackPut(S,K,r+1e-4,sigma,t,T,Smintodate) - AnalyticFixedStrikeLookBackPut(S,K,r-1e-4,sigma,t,T,Smintodate) ) / (2*1e-4)
	
	return(PutPrice, Delta, Gamma, Vega, Theta, Rho)


############################################################################
############################################################################

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


############################################################################
############################# MONTE-CARLO ##################################


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
	
	Note: The Payoff is given by S(T) - Smin, where Smin is the minimum over the option lifetime, and S(T) is the price at maturity.
	
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
		
	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps+1))			### Generate a Brownian motion array - 1 additional step for when we calculate theta
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array = np.exp(log_path_array[:, -2])
	
	############ Max and minimum prices ###########################
	MinPrice_array = np.exp( np.min(log_path_array[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation

	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array	
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	log_path_array_AT = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_AT ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array_AT = np.exp(log_path_array_AT[:, -2])

	############ Max and minimum prices ###########################
	MinPrice_array_AT = np.exp( np.min(log_path_array_AT[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	
	##### perturb in t for theta ##############
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	terminal_price_array_smaller_t_AT = np.exp(log_path_array_AT[:, -1])
	terminal_price_array_larger_t_AT = np.exp(log_path_array_AT[:, -3])

	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	
	MinPrice_array_smaller_t_AT = np.exp( np.min(log_path_array_AT, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t_AT = np.exp( np.min(log_path_array_AT[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter	
	
	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()		
	MinPrice_array = MinPrice_array.flatten()
	
	terminal_price_array_AT = terminal_price_array_AT.flatten()	
	MinPrice_array_AT = MinPrice_array_AT.flatten()
	
	terminal_price_array_smaller_t_AT = terminal_price_array_smaller_t_AT.flatten()
	terminal_price_array_larger_t_AT = terminal_price_array_larger_t_AT.flatten()
	
	MinPrice_array_smaller_t = MinPrice_array_smaller_t.flatten() 
	MinPrice_array_larger_t = MinPrice_array_larger_t.flatten() 
	
	MinPrice_array_smaller_t_AT = MinPrice_array_smaller_t_AT.flatten() 
	MinPrice_array_larger_t_AT = MinPrice_array_larger_t_AT.flatten() 	
	
	#### concatenate ####

	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT) )
	MinPrice_array = np.concatenate( (MinPrice_array, MinPrice_array_AT ) )

	terminal_price_array_smaller_t = np.concatenate(( terminal_price_array_smaller_t, terminal_price_array_smaller_t_AT  ) )
	terminal_price_array_larger_t = np.concatenate(( terminal_price_array_larger_t, terminal_price_array_larger_t_AT  ) )
	   	
	MinPrice_array_smaller_t = np.concatenate( (MinPrice_array_smaller_t, MinPrice_array_smaller_t_AT ) ) 
	MinPrice_array_larger_t = np.concatenate( (MinPrice_array_larger_t, MinPrice_array_larger_t_AT ) ) 
	
	#### special arrays for pathwise derivatives
	log_path_array_full = np.concatenate( (log_path_array, log_path_array_AT))
	
	Smin_index = np.argmin(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	taumin_array = Smin_index*time_step + t 
	
	Z_final_value_array = 1/(sigma*np.sqrt(T-t))*(np.log(terminal_price_array/S) - (r - 0.5*sigma**2)*(T-t)) #### Final Z ~ N(0,1) values needed for some of the pathwise derivatives	
	Z_value_taumin = 1/(sigma)*( np.log(MinPrice_array/S) - (r - 0.5*sigma**2)*(taumin_array-t) )  ### returns the Z value at S_min times sqrt(tau - t) - to avoid divide by zero error 			
	
	######## Broadie-Glasserman-Kou (BGK) Continuity Correction
	time_step = (T-t)/n_steps
	BGKcorrection = np.exp(-0.5826*sigma*np.sqrt(time_step)) 
		
	Smintodateoriginal = Smintodate
	if Smintodate is None:          
		Smintodate = S		##### Set Smintodate to S(t) if no value is entered (say, lookback has just been initiated)
		Smintodate = Smintodate*BGKcorrection
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate

	###### CALCULATE THE OPTION PRICE ##################
	MinPrice_array = MinPrice_array*BGKcorrection
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( terminal_price_array - MinPrice_array, terminal_price_array - Smintodate )
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) - Pathwise Method #########
	if Smintodateoriginal is None:
		delta_array = np.exp(-r*(T-t))*1/S*(terminal_price_array - MinPrice_array)
	else:
		delta_array = np.exp(-r*(T-t))*( terminal_price_array/S - MinPrice_array/S*Heaviside_smoothed( Smintodate - MinPrice_array ) )
	
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial)  - Pathwise Method #####

	if Smintodateoriginal is None:
		Extra_smoothing = 0.5
		gamma_array = -np.exp(-r*(T-t))*( (MinPrice_array/S)**2*(-1)*Heaviside_dx_smoothed( Smintodate - MinPrice_array , Extra_smoothing) ) #### Have to smooth this significantly
	else:
		Extra_smoothing = 0.5
		gamma_array = -np.exp(-r*(T-t))*( (MinPrice_array/S)**2*(-1)*Heaviside_dx_smoothed( Smintodate - MinPrice_array , Extra_smoothing) )
	
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	dSTdsigma_array =   ( -sigma*(T-t)            + Z_final_value_array*np.sqrt(T-t) )*terminal_price_array
	dSmindsigma_array = ( -sigma*(taumin_array-t) + Z_value_taumin)*MinPrice_array
	
	Extra_smoothing = 0.1
	vega_array = np.exp(-r*(T-t))*( dSTdsigma_array*1 + dSmindsigma_array*(-Heaviside_smoothed( Smintodate - MinPrice_array , Extra_smoothing)) )

	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
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
	dSTdr_array = (T-t)*terminal_price_array
	dSmindr_array = (taumin_array-t)*MinPrice_array		

	rho_array = np.exp(-r*(T-t))*( dSTdr_array*1 + dSmindr_array*(-Heaviside_smoothed( Smintodate - MinPrice_array , Extra_smoothing)) )	
		
	rho_value = -(T-t)*option_value + np.mean(rho_array)
	rho_StandardError = np.sqrt( ((T-t)*option_value_StandardError)**2 + stats.sem(rho_array)**2)	
	
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
	Smaxtodate is the maximum price of S observed with option active up to time t [default if none entered: Smaxtodate = S(t), e.g. when t is chosen just as the option comes into existence]
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
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)
	
	time_step = (T-t)/n_steps

	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps+1))			### Generate a Brownian motion array - 1 additional step for when we calculate theta
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array = np.exp(log_path_array[:, -2])
	
	############ Max and minimum prices ###########################
	MaxPrice_array = np.exp( np.max(log_path_array[:, :-1], axis=1, keepdims=True) )

	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array	
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	log_path_array_AT = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_AT ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	#### calculate the terminal prices and store them as a numpy array ####
	terminal_price_array_AT = np.exp(log_path_array_AT[:, -2])


	############ Max and minimum prices ###########################
	MaxPrice_array_AT = np.exp( np.max(log_path_array_AT[:, :-1], axis=1, keepdims=True) )	
	
	##### perturb in t for theta ##############
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	terminal_price_array_smaller_t_AT = np.exp(log_path_array_AT[:, -1])
	terminal_price_array_larger_t_AT = np.exp(log_path_array_AT[:, -3])

	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MaxPrice_array_smaller_t_AT = np.exp( np.max(log_path_array_AT, axis=1, keepdims=True) )
	MaxPrice_array_larger_t_AT = np.exp( np.max(log_path_array_AT[:, :-2], axis=1, keepdims=True) )	
		
	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()		
	MaxPrice_array = MaxPrice_array.flatten()
	
	terminal_price_array_AT = terminal_price_array_AT.flatten()	
	MaxPrice_array_AT = MaxPrice_array_AT.flatten()
	
	terminal_price_array_smaller_t_AT = terminal_price_array_smaller_t_AT.flatten()
	terminal_price_array_larger_t_AT = terminal_price_array_larger_t_AT.flatten()
	
	MaxPrice_array_smaller_t = MaxPrice_array_smaller_t.flatten()
	MaxPrice_array_larger_t = MaxPrice_array_larger_t.flatten()
	
	MaxPrice_array_smaller_t_AT = MaxPrice_array_smaller_t_AT.flatten()
	MaxPrice_array_larger_t_AT = MaxPrice_array_larger_t_AT.flatten() 		
	
	#### concatenate ####

	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT) )
	MaxPrice_array = np.concatenate( (MaxPrice_array, MaxPrice_array_AT ) )

	terminal_price_array_smaller_t = np.concatenate(( terminal_price_array_smaller_t, terminal_price_array_smaller_t_AT  ) )
	terminal_price_array_larger_t = np.concatenate(( terminal_price_array_larger_t, terminal_price_array_larger_t_AT  ) )
	   	
	MaxPrice_array_smaller_t = np.concatenate( (MaxPrice_array_smaller_t, MaxPrice_array_smaller_t_AT ) )
	MaxPrice_array_larger_t = np.concatenate( (MaxPrice_array_larger_t, MaxPrice_array_larger_t_AT ) )	
	
	#### special arrays for pathwise derivatives
	
	log_path_array_full = np.concatenate( (log_path_array, log_path_array_AT))
	
	Smax_index = np.argmax(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	taumax_array = Smax_index*time_step + t 
	
	Z_final_value_array = 1/(sigma*np.sqrt(T-t))*(np.log(terminal_price_array/S) - (r - 0.5*sigma**2)*(T-t)) #### Final Z ~ N(0,1) values needed for some of the pathwise derivatives	
	Z_value_taumax = 1/(sigma)*( np.log(MaxPrice_array/S) - (r - 0.5*sigma**2)*(taumax_array-t) )  ### returns the Z value at S_max times sqrt(tau - t) - to avoid divide by zero error
	
	
	######## Broadie-Glasserman-Kou (BGK) Continuity Correction
	time_step = (T-t)/n_steps
	BGKcorrection = np.exp(0.5826*sigma*np.sqrt(time_step))
	
	Smaxtodateoriginal = Smaxtodate
	if Smaxtodate is None:          
		Smaxtodate = S 		       ##### Set Smaxtodate to S(t) if no value is entered (say, lookback has just been initiated)
		Smaxtodate = Smaxtodate*BGKcorrection
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
	
	###### CALCULATE THE OPTION PRICE ###################
	MaxPrice_array = MaxPrice_array*BGKcorrection	
	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum( MaxPrice_array - terminal_price_array  , Smaxtodate - terminal_price_array  )
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure (apply discounting factor from payoff at T to t)
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)

	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) - Pathwise Method #########
	if Smaxtodateoriginal is None:
		delta_array = np.exp(-r*(T-t))/S*( MaxPrice_array - terminal_price_array)
	else:
		delta_array = np.exp(-r*(T-t))*( MaxPrice_array/S*Heaviside_smoothed( MaxPrice_array - Smaxtodate ) - terminal_price_array/S )
	
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial)  - Pathwise Method #####
	
	if Smaxtodateoriginal is None:
		Extra_smoothing = 0.5
		gamma_array = np.exp(-r*(T-t))*( (MaxPrice_array/S)**2*Heaviside_dx_smoothed( MaxPrice_array - Smaxtodate, Extra_smoothing ) )  #### Significant smoothing 
	else:
		Extra_smoothing = 0.5
		gamma_array = np.exp(-r*(T-t))*( (MaxPrice_array/S)**2*Heaviside_dx_smoothed( MaxPrice_array - Smaxtodate, Extra_smoothing ) )
	
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	dSTdsigma_array =   ( -sigma*(T-t)            + Z_final_value_array*np.sqrt(T-t) )*terminal_price_array
	dSmaxdsigma_array = ( -sigma*(taumax_array-t) + Z_value_taumax)*MaxPrice_array
	
	Extra_smoothing = 0.1
	vega_array = np.exp(-r*(T-t))*( dSTdsigma_array*-1 + dSmaxdsigma_array*Heaviside_smoothed( MaxPrice_array - Smaxtodate, Extra_smoothing ) )

		
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
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
	
	dSTdr_array = (T-t)*terminal_price_array
	dSmaxdr_array = (taumax_array-t)*MaxPrice_array
	
	Extra_smoothing = 0.1
	rho_array = np.exp(-r*(T-t))*( dSTdr_array*-1 + dSmaxdr_array*Heaviside_smoothed( MaxPrice_array - Smaxtodate, Extra_smoothing) )
	
	rho_value = -(T-t)*option_value + np.mean(rho_array)
	rho_StandardError = np.sqrt( ((T-t)*option_value_StandardError)**2 + stats.sem(rho_array)**2)
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
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)
	
	time_step = (T-t)/n_steps
	
	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	rng = np.random.default_rng(BaseSeed)
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps+1))			### Generate a Brownian motion array - 1 additional step for when we calculate theta
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	############ Max and minimum prices ###########################
	MaxPrice_array = np.exp( np.max(log_path_array[:, :-1], axis=1, keepdims=True) )

	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array	
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	log_path_array_AT = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_AT ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	############ Max and minimum prices ###########################
	MaxPrice_array_AT = np.exp( np.max(log_path_array_AT[:, :-1], axis=1, keepdims=True) )	
	
	##### perturb in t for theta ##############
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MaxPrice_array_smaller_t_AT = np.exp( np.max(log_path_array_AT, axis=1, keepdims=True) )
	MaxPrice_array_larger_t_AT = np.exp( np.max(log_path_array_AT[:, :-2], axis=1, keepdims=True) )	
		
	##### flatten arrays #####

	MaxPrice_array = MaxPrice_array.flatten()
	MaxPrice_array_AT = MaxPrice_array_AT.flatten()
	
	MaxPrice_array_smaller_t = MaxPrice_array_smaller_t.flatten()
	MaxPrice_array_larger_t = MaxPrice_array_larger_t.flatten()
	
	MaxPrice_array_smaller_t_AT = MaxPrice_array_smaller_t_AT.flatten()
	MaxPrice_array_larger_t_AT = MaxPrice_array_larger_t_AT.flatten() 		
	
	#### concatenate ####

	MaxPrice_array = np.concatenate( (MaxPrice_array, MaxPrice_array_AT ) )
	MaxPrice_array_smaller_t = np.concatenate( (MaxPrice_array_smaller_t, MaxPrice_array_smaller_t_AT ) )
	MaxPrice_array_larger_t = np.concatenate( (MaxPrice_array_larger_t, MaxPrice_array_larger_t_AT ) )	
	
	#### special arrays for pathwise derivatives
	log_path_array_full = np.concatenate( (log_path_array, log_path_array_AT))
	Smax_index = np.argmax(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	taumax_array = Smax_index*time_step + t 
	
	Z_value_taumax = 1/(sigma)*( np.log(MaxPrice_array/S) - (r - 0.5*sigma**2)*(taumax_array-t) )  ### returns the Z value at S_max times sqrt(tau - t) - to avoid divide by zero error
	
	
	######## Broadie-Glasserman-Kou (BGK) Continuity Correction
	time_step = (T-t)/n_steps
	BGKcorrection = np.exp(0.5826*sigma*np.sqrt(time_step)) 

	######
	
	Smaxtodateoriginal = Smaxtodate
	if Smaxtodate is None:          
		Smaxtodate = S 		       ##### Set Smaxtodate to S(t) if no value is entered (say, lookback has just been initiated)
		Smaxtodate = Smaxtodate*BGKcorrection
	elif S > Smaxtodate:
		raise ValueError("S(t) is larger than entered value of Smaxtodate! This is not possible!") ### error if contradictory value entered
	else:				
		Smaxtodate = Smaxtodate        #####  Otherwise Smaxtodate to entered value 
		
	###### CALCULATE THE OPTION PRICE ##################
	MaxPrice_array = MaxPrice_array*BGKcorrection

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
	#### Calculate Delta = dV/dS (partial) - Pathwise Method #########
	
	if Smaxtodateoriginal is None:
		delta_array = np.exp(-r*(T-t))*( MaxPrice_array/S*Heaviside_smoothed( MaxPrice_array-K ) )
	else:
		delta_array = np.exp(-r*(T-t))*( MaxPrice_array/S*Heaviside_smoothed( MaxPrice_array-K )*Heaviside_smoothed( MaxPrice_array-Smaxtodate ) + MaxPrice_array/S*( (MaxPrice_array-K)*Heaviside_smoothed(MaxPrice_array-K) - (Smaxtodate-K)*Heaviside_smoothed(Smaxtodate-K))*Heaviside_dx_smoothed( MaxPrice_array-Smaxtodate ))
	
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) - Pathwise Method #####	
	
	if Smaxtodateoriginal is None:
#		gamma_array = np.exp(-r*(T-t))*( (MaxPrice_array/S)**2*( Heaviside_dx_smoothed(MaxPrice_array-K) ) )
		
		Extra_smoothing = 1.0   #### Have to check this again ######
		gamma_array = np.exp(-r*(T-t))*( (MaxPrice_array/S)**2*(Heaviside_dx_smoothed(MaxPrice_array-K, Extra_smoothing)*Heaviside_smoothed(MaxPrice_array-Smaxtodate, Extra_smoothing) + 2*Heaviside_dx_smoothed(MaxPrice_array-Smaxtodate, Extra_smoothing)*Heaviside_smoothed(MaxPrice_array - K, Extra_smoothing) + Heaviside_dx2_smoothed(MaxPrice_array-Smaxtodate, Extra_smoothing)*( (MaxPrice_array-K)*Heaviside_smoothed(MaxPrice_array-K, Extra_smoothing) - (Smaxtodate-K)*Heaviside_smoothed(Smaxtodate-K, Extra_smoothing)) ) )
	
	else:
		Extra_smoothing = 1.0	
		gamma_array = np.exp(-r*(T-t))*( (MaxPrice_array/S)**2*(Heaviside_dx_smoothed(MaxPrice_array-K, Extra_smoothing)*Heaviside_smoothed(MaxPrice_array-Smaxtodate, Extra_smoothing) + 2*Heaviside_dx_smoothed(MaxPrice_array-Smaxtodate, Extra_smoothing)*Heaviside_smoothed(MaxPrice_array - K, Extra_smoothing) + Heaviside_dx2_smoothed(MaxPrice_array-Smaxtodate, Extra_smoothing)*( (MaxPrice_array-K)*Heaviside_smoothed(MaxPrice_array-K, Extra_smoothing) - (Smaxtodate-K)*Heaviside_smoothed(Smaxtodate-K, Extra_smoothing)) ) )
	

	gamma_value = np.mean(gamma_array)
	
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	dSmaxdsigma_array = ( -sigma*(taumax_array-t) + Z_value_taumax)*MaxPrice_array					

	Extra_smoothing = 0.1
	vega_array = np.exp(-r*(T-t))*dSmaxdsigma_array*( Heaviside_smoothed( MaxPrice_array - K, Extra_smoothing)*Heaviside_smoothed( MaxPrice_array - Smaxtodate, Extra_smoothing) + Heaviside_dx_smoothed(MaxPrice_array - Smaxtodate, Extra_smoothing )*((MaxPrice_array-K)*Heaviside_smoothed(MaxPrice_array-K, Extra_smoothing) - (Smaxtodate-K)*Heaviside_smoothed(Smaxtodate-K, Extra_smoothing)) )
	
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
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
	
	dSmaxdr_array = (taumax_array-t)*MaxPrice_array
	
	Extra_smoothing = 0.1
	rho_array = np.exp(-r*(T-t))*dSmaxdr_array*( Heaviside_smoothed( MaxPrice_array - K, Extra_smoothing)*Heaviside_smoothed( MaxPrice_array - Smaxtodate, Extra_smoothing) + Heaviside_dx_smoothed(MaxPrice_array - Smaxtodate, Extra_smoothing )*((MaxPrice_array-K)*Heaviside_smoothed(MaxPrice_array-K, Extra_smoothing) - (Smaxtodate-K)*Heaviside_smoothed(Smaxtodate-K, Extra_smoothing)) )
	
	rho_value = -(T-t)*option_value + np.mean(rho_array)
	rho_StandardError = np.sqrt( ((T-t)*option_value_StandardError)**2 + stats.sem(rho_array)**2)	


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
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)
	
	
	time_step = (T-t)/n_steps
	### Calculate the evolution of the stock price - vectorized approach ############
	#################################################################################
	
	rng = np.random.default_rng(BaseSeed)
	brownian_array = rng.normal(0, 1, size=(n_simulations,n_steps+1))			### Generate a Brownian motion array - 1 additional step for when we calculate theta
	log_step_array = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array ### Array of movements in the log of S for each time step
	log_path_array = np.log(S) + np.cumsum( log_step_array, axis = 1 ) 			### Prices at each time step
	log_path_array = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	############ Max and minimum prices ###########################
	MinPrice_array = np.exp( np.min(log_path_array[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation

	#################################################
	######	Antithetic Pairs of the above ###########
	brownian_array_AT = -brownian_array	
	log_step_array_AT = (r - 0.5*sigma**2)*time_step + np.sqrt(time_step)*sigma*brownian_array_AT ### Array of movements in the log of S for each time step
	log_path_array_AT = np.log(S) + np.cumsum( log_step_array_AT, axis = 1 ) 			### Prices at each time step
	log_path_array_AT = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_AT ), axis=1 ) ### add the initial price as the first entry
	
	#################################################################################
	############ Max and minimum prices ###########################
	MinPrice_array_AT = np.exp( np.min(log_path_array_AT[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	
	##### perturb in t for theta ##############
	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	
	MinPrice_array_smaller_t_AT = np.exp( np.min(log_path_array_AT, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t_AT = np.exp( np.min(log_path_array_AT[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter	
	
	##### flatten arrays #####
	MinPrice_array = MinPrice_array.flatten()
	
	MinPrice_array_AT = MinPrice_array_AT.flatten()
		
	MinPrice_array_smaller_t = MinPrice_array_smaller_t.flatten() 
	MinPrice_array_larger_t = MinPrice_array_larger_t.flatten() 
	
	MinPrice_array_smaller_t_AT = MinPrice_array_smaller_t_AT.flatten() 
	MinPrice_array_larger_t_AT = MinPrice_array_larger_t_AT.flatten() 	
	
	#### concatenate ####
	MinPrice_array = np.concatenate( (MinPrice_array, MinPrice_array_AT ) )
	   	
	MinPrice_array_smaller_t = np.concatenate( (MinPrice_array_smaller_t, MinPrice_array_smaller_t_AT ) ) 
	MinPrice_array_larger_t = np.concatenate( (MinPrice_array_larger_t, MinPrice_array_larger_t_AT ) ) 
	
	#### special arrays for pathwise derivatives
	log_path_array_full = np.concatenate( (log_path_array, log_path_array_AT))
	
	Smin_index = np.argmin(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	taumin_array = Smin_index*time_step + t 
	
	Z_value_taumin = 1/(sigma)*( np.log(MinPrice_array/S) - (r - 0.5*sigma**2)*(taumin_array-t) )  ### returns the Z value at S_min times sqrt(tau - t) - to avoid divide by zero error 	

	######## Broadie-Glasserman-Kou (BGK) Continuity Correction
	time_step = (T-t)/n_steps
	BGKcorrection = np.exp(-0.5826*sigma*np.sqrt(time_step)) 
	
	Smintodateoriginal = Smintodate
	if Smintodate is None:          
		Smintodate = S		##### Set Smintodate to S(t) if no value is entered (say, lookback has just been initiated)
		Smintodate = Smintodate*BGKcorrection
	elif Smintodate > S:
		raise ValueError("S(t) is smaller than entered value of Smintodate! This is not possible!") ### error if contradictory value entered
	else:				#####  Otherwise Smintodate set to entered value
		Smintodate = Smintodate
	
	###### CALCULATE THE OPTION PRICE ###################
	MinPrice_array = MinPrice_array*BGKcorrection	
	
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
	#### Calculate Delta = dV/dS (partial) - Pathwise Method #########
	
	if Smintodateoriginal is None:
		delta_array = np.exp(-r*(T-t))*( MinPrice_array/S*(-Heaviside_smoothed( K - MinPrice_array )))
	else:
		delta_array = np.exp(-r*(T-t))*( MinPrice_array/S*(-Heaviside_smoothed( K - MinPrice_array )*Heaviside_smoothed( Smintodate - MinPrice_array ) + Heaviside_dx_smoothed(MinPrice_array - Smintodate)*( (K-MinPrice_array)*Heaviside_smoothed(K-MinPrice_array) - (K-Smintodate)*Heaviside_smoothed(K-Smintodate) )  ) )
	
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) - Pathwise Method #####	
	
	if Smintodateoriginal is None:
#		gamma_array = np.exp(-r*(T-t))*( (MinPrice_array/S)**2*Heaviside_dx_smoothed(K-MinPrice_array) )
		
		Extra_smoothing = 1.0   #### Have to check this again ######
		gamma_array = np.exp(-r*(T-t))*( (MinPrice_array/S)**2*(Heaviside_dx_smoothed(K-MinPrice_array, Extra_smoothing)*Heaviside_smoothed(Smintodate-MinPrice_array, Extra_smoothing) + 2*Heaviside_dx_smoothed(MinPrice_array-Smintodate, Extra_smoothing)*Heaviside_smoothed( K - MinPrice_array, Extra_smoothing) - Heaviside_dx2_smoothed(MinPrice_array-Smintodate, Extra_smoothing)*( (K-MinPrice_array)*Heaviside_smoothed(K-MinPrice_array, Extra_smoothing) - (K-Smintodate)*Heaviside_smoothed(K-Smintodate, Extra_smoothing) ) ) )
	else: 
		Extra_smoothing = 1.0
		gamma_array = np.exp(-r*(T-t))*( (MinPrice_array/S)**2*(Heaviside_dx_smoothed(K-MinPrice_array, Extra_smoothing)*Heaviside_smoothed(Smintodate-MinPrice_array, Extra_smoothing) + 2*Heaviside_dx_smoothed(MinPrice_array-Smintodate, Extra_smoothing)*Heaviside_smoothed( K - MinPrice_array, Extra_smoothing) - Heaviside_dx2_smoothed(MinPrice_array-Smintodate, Extra_smoothing)*( (K-MinPrice_array)*Heaviside_smoothed(K-MinPrice_array, Extra_smoothing) - (K-Smintodate)*Heaviside_smoothed(K-Smintodate, Extra_smoothing) ) ) ) ### check minus signs

	gamma_value = np.mean(gamma_array)	
	gamma_StandardError = stats.sem(gamma_array)	

	#### Calculate Vega = dV/dsigma (partial) ######
	
	dSmindsigma_array = ( -sigma*(taumin_array-t) + Z_value_taumin)*MinPrice_array
	
	Extra_smoothing = 0.3
	vega_array = np.exp(-r*(T-t))*dSmindsigma_array*( -Heaviside_smoothed(K-MinPrice_array,Extra_smoothing)*Heaviside_smoothed(Smintodate-MinPrice_array,Extra_smoothing) + Heaviside_dx_smoothed(MinPrice_array-Smintodate, Extra_smoothing)*(  (K-Smintodate)*Heaviside_smoothed(K-Smintodate, Extra_smoothing) - (K-MinPrice_array)*Heaviside_smoothed(K-MinPrice_array, Extra_smoothing)  )  ) 
	
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
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

	dSmindr_array = (taumin_array-t)*MinPrice_array		

	Extra_smoothing = 0.3
	rho_array = np.exp(-r*(T-t))*dSmindr_array*( -Heaviside_smoothed(K-MinPrice_array,Extra_smoothing)*Heaviside_smoothed(Smintodate-MinPrice_array,Extra_smoothing) + Heaviside_dx_smoothed(MinPrice_array-Smintodate, Extra_smoothing)*(  (K-Smintodate)*Heaviside_smoothed(K-Smintodate, Extra_smoothing) - (K-MinPrice_array)*Heaviside_smoothed(K-MinPrice_array, Extra_smoothing)  )  ) 
	
	rho_value = -(T-t)*option_value + np.mean(rho_array)
	rho_StandardError = np.sqrt( ((T-t)*option_value_StandardError)**2 + stats.sem(rho_array)**2)

	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)


############################################################################
############################################################################


def main():
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
