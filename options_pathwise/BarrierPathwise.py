import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from scipy.stats import norm
from multiprocessing import Pool
from scipy.integrate import quad

###

'''
GIVES ANALYTIC AND MONTE-CARLO IMPLEMENTATION OF EUROPEAN OPTIONS WITH BARRIERS
RETURNS THE OPTION PRICE, GREEKS, AND STANDARD ERRORS
THE MONTE-CARLO USES ANTITHETIC VARIATES THROUGHOUT
THE DELTA AND GAMMA CALCULATION USES THE PATHWISE DERIVATIVE METHOD
'''

StandardBaseSeed = 0

#######################

def AnalyticBlackScholesKnockInCall(S,K,r,sigma,t,T,KnockInBarrier):
	
	'''
	Calculates the Black Scholes Knock In Call price using the analytic formula

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	KnockInBarrier is the Barrier (constant in time)
	
	Output is the call price in dollars
	
	'''
	#################################################################
	#################################################################
	
	
	if S > KnockInBarrier:	### price of down and in call (See Hull Chapter 26)
		
		####################################################
		if KnockInBarrier <= K: 
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockInBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			PriceDownIn = S*( KnockInBarrier/ S )**(2*lambdafactor)*norm.cdf(yfactor) - K*np.exp(-r*(T-t))*( KnockInBarrier/ S )**(2*lambdafactor-2)*norm.cdf(yfactor-sigma*np.sqrt(T-t))
		else:
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			x1factor = np.log( S/KnockInBarrier ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			y1factor = np.log( KnockInBarrier/S ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			PriceDownIn = S*norm.cdf(x1factor) - K*np.exp(-r*(T-t))*norm.cdf(x1factor-sigma*np.sqrt(T-t)) - S*( KnockInBarrier/s )**(2*lambdafactor)*norm.cdf(y1factor)+K*np.exp(-r*(T-t))*( KnockInBarrier/ S )**(2*lambdafactor-2)*norm.cdf(y1factor-sigma*np.sqrt(T-t))
		####################################################
		
		CallPrice = PriceDownIn
		
	############################################################	
	############################################################
	
	else: 	### price of up and in call  (See Hull Chapter 26)

		###########################################
		if KnockInBarrier <= K:
			##### Standard Vanilla Call in this Case ######
			d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			PriceUpIn = S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2)
		else:
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockInBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			y1factor = np.log( KnockInBarrier/S ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			x1factor = np.log( S/KnockInBarrier ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			PriceUpIn = S*norm.cdf(x1factor) - K*np.exp(-r*(T-t))*norm.cdf(x1factor-sigma*np.sqrt(T-t)) - S*( KnockInBarrier/S )**(2*lambdafactor)*(norm.cdf(-yfactor)-norm.cdf(-y1factor))+K*np.exp(-r*(T-t))*( KnockInBarrier/ S )**(2*lambdafactor-2)*(norm.cdf(-yfactor+sigma*np.sqrt(T-t))-norm.cdf(-y1factor+sigma*np.sqrt(T-t)))	
	####################################################
		
		CallPrice = PriceUpIn
	
	##########################################################
	##########################################################		
	
	return(CallPrice)
	
#######################

def AnalyticBlackScholesKnockOutCall(S,K,r,sigma,t,T,KnockOutBarrier):
	
	'''
	Calculates the Black Scholes Knock Out Call price using the analytic formula

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	KnockOutBarrier is the Barrier (constant in time)
	
	Output is the call price in dollars
	
	'''
	#################################################################
	#################################################################
	
	#### Vanilla Call Price	
	d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	VanillaCallPrice = S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2) 
	
	
	#### Knock In Call Price
	KnockInBarrier = KnockOutBarrier
	KnockInCallPrice = AnalyticBlackScholesKnockInCall(S,K,r,sigma,t,T,KnockInBarrier)		

	##### Knock Out Call Price
	
	KnockOutCallPrice = VanillaCallPrice - KnockInCallPrice  #### Using Parity Relation
 		
	return(KnockOutCallPrice)

#######################

def AnalyticBlackScholesKnockInPut(S,K,r,sigma,t,T,KnockInBarrier):
	
	'''
	Calculates the Black Scholes Knock In Put price using the analytic formula

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	KnockInBarrier is the Barrier (constant in time)
	
	Output is the Put price in dollars
	
	'''
	#################################################################
	#################################################################
	
	
	if S > KnockInBarrier:	### price of down and in put 
		
		####################################################
		if KnockInBarrier <= K: 
		
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockInBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			y1factor = np.log( KnockInBarrier/S ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)				
			x1factor = np.log( S/KnockInBarrier ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
#			x2factor = np.log( S/K ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			
			PriceDownIn = -S*norm.cdf(-x1factor) + K*np.exp(-r*(T-t))*norm.cdf(-x1factor+sigma*np.sqrt(T-t)) + S*( KnockInBarrier/S )**(2*lambdafactor)*(norm.cdf(yfactor)-norm.cdf(y1factor))-K*np.exp(-r*(T-t))*(KnockInBarrier/S)**(2*lambdafactor-2)*(norm.cdf(yfactor-sigma*np.sqrt(T-t))-norm.cdf(y1factor-sigma*np.sqrt(T-t)))
								
		else: #### should be vanilla put price 
		
			d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			PriceDownIn = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1) 
		
		####################################################
		
		PutPrice = PriceDownIn
		
	############################################################	
	############################################################
	
	else: 	### price of up and in put 

		###########################################
		if KnockInBarrier <= K:
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockInBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			y1factor = np.log( KnockInBarrier/S ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)				
			x1factor = np.log( S/KnockInBarrier ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			
			PriceUpOut = -S*norm.cdf(-x1factor) + K*norm.cdf(-x1factor+sigma*np.sqrt(T-t)) + S*(KnockInBarrier/S)**(2*lambdafactor)*norm.cdf(-y1factor)-K*np.exp(-r*(T-t))*(KnockInBarrier/S)**(2*lambdafactor-2)*norm.cdf(-y1factor+sigma*np.sqrt(T-t))
			
			d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			VanillaPutprice = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1) 		
			
			PriceUpIn = VanillaPutprice - PriceUpOut
			
		else: #### Hull Formula
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockInBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			PriceUpIn = -S*( KnockInBarrier/S )**(2*lambdafactor)*norm.cdf(-yfactor) + K*np.exp(-r*(T-t))*( KnockInBarrier/S )**(2*lambdafactor-2)*norm.cdf(-yfactor + sigma*np.sqrt(T-t))
		
	####################################################
		
		PutPrice = PriceUpIn
	
	##########################################################
	##########################################################		
	
	return(PutPrice)
	
def AnalyticBlackScholesKnockOutPut(S,K,r,sigma,t,T,KnockOutBarrier):
	
	'''
	Calculates the Black Scholes Knock In Put price using the analytic formula

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	KnockOutBarrier is the Barrier (constant in time)
	
	Output is the Put price in dollars
	
	'''
	#################################################################
	#################################################################
	
	
	if S > KnockOutBarrier:	### price of down and out put 
		
		####################################################
		if KnockOutBarrier <= K: #### Hull
			
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockOutBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			y1factor = np.log( KnockOutBarrier/S ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)				
			x1factor = np.log( S/KnockOutBarrier ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
#			x2factor = np.log( S/K ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			
			##### Price of the Down-and-In Put Using the Same value of the barrier
			PriceDownIn = -S*norm.cdf(-x1factor) + K*np.exp(-r*(T-t))*norm.cdf(-x1factor+sigma*np.sqrt(T-t)) + S*( KnockOutBarrier/S )**(2*lambdafactor)*(norm.cdf(yfactor)-norm.cdf(y1factor))-K*np.exp(-r*(T-t))*(KnockOutBarrier/S)**(2*lambdafactor-2)*(norm.cdf(yfactor-sigma*np.sqrt(T-t))-norm.cdf(y1factor-sigma*np.sqrt(T-t)))
			
			d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			VanillaPutPrice = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1) 
			
			PriceDownOut = VanillaPutPrice - PriceDownIn ### Parity relation
		else:
			PriceDownOut = 0 

		####################################################
		
		PutPrice = PriceDownOut
		
	############################################################	
	############################################################
	
	else: 	### price of up and out put 

		###########################################
		if KnockOutBarrier <= K:
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockOutBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			y1factor = np.log( KnockOutBarrier/S ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)				
			x1factor = np.log( S/KnockOutBarrier ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			
			PriceUpOut = -S*norm.cdf(-x1factor) + K*norm.cdf(-x1factor+sigma*np.sqrt(T-t)) + S*(KnockOutBarrier/S)**(2*lambdafactor)*norm.cdf(-y1factor)-K*np.exp(-r*(T-t))*(KnockOutBarrier/S)**(2*lambdafactor-2)*norm.cdf(-y1factor+sigma*np.sqrt(T-t))
			
		else: ##### Using Hull
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockOutBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			##### Price of the Up-and-In Put Using the Same value of the barrier
			PriceUpIn = -S*( KnockOutBarrier/S )**(2*lambdafactor)*norm.cdf(-yfactor) + K*np.exp(-r*(T-t))*( KnockOutBarrier/S )**(2*lambdafactor-2)*norm.cdf(-yfactor + sigma*np.sqrt(T-t))
			
			d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			VanillaPutPrice = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1) 
			
			PriceUpOut =  VanillaPutPrice - PriceUpIn ### Parity relation
	####################################################
		
		PutPrice = PriceUpOut
	
	##########################################################
	##########################################################		
	
	return(PutPrice)

##########################
##########################

def AnalyticBlackScholesKnockInCallWithGreeks(S,K,r,sigma,t,T,KnockInBarrier):
	
	'''
	Calculates the Black Scholes Knock In Call price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	KnockInBarrier is the Barrier (constant in time)
	
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	
	#### Call Price Using Analytic Formula #######
	CallPrice = AnalyticBlackScholesKnockInCall(S,K,r,sigma,t,T,KnockInBarrier)
	
	small_time_step = (T-t)/100
	
	### The Greeks ####
	Delta = ( AnalyticBlackScholesKnockInCall(S+0.01,K,r,sigma,t,T,KnockInBarrier) - AnalyticBlackScholesKnockInCall(S-0.01,K,r,sigma,t,T,KnockInBarrier) ) / (2*0.01)
	Gamma = ( AnalyticBlackScholesKnockInCall(S+0.01,K,r,sigma,t,T,KnockInBarrier) - 2*AnalyticBlackScholesKnockInCall(S,K,r,sigma,t,T,KnockInBarrier) + AnalyticBlackScholesKnockInCall(S-0.01,K,r,sigma,t,T,KnockInBarrier))/(0.01**2)
	Vega = ( AnalyticBlackScholesKnockInCall(S,K,r,sigma+0.01,t,T,KnockInBarrier) - AnalyticBlackScholesKnockInCall(S,K,r,sigma-0.01,t,T,KnockInBarrier) ) / (2*0.01)
	Theta = -( AnalyticBlackScholesKnockInCall(S,K,r,sigma,t+small_time_step,T,KnockInBarrier) - AnalyticBlackScholesKnockInCall(S,K,r,sigma,t-small_time_step,T,KnockInBarrier) ) / (2*small_time_step)
	Rho = ( AnalyticBlackScholesKnockInCall(S,K,r+1e-4,sigma,t,T,KnockInBarrier) - AnalyticBlackScholesKnockInCall(S,K,r-1e-4,sigma,t,T,KnockInBarrier) ) / (2*1e-4)
	
	return(CallPrice, Delta, Gamma, Vega, Theta, Rho)	

#####

def AnalyticBlackScholesKnockOutCallWithGreeks(S,K,r,sigma,t,T,KnockOutBarrier):
	
	'''
	Calculates the Black Scholes Knock Out Call price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	KnockOutBarrier is the Barrier (constant in time)
	
	Output is: 
	The Call price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	
	#### Call Price Using Analytic Formula #######
	CallPrice = AnalyticBlackScholesKnockOutCall(S,K,r,sigma,t,T,KnockOutBarrier)
	
	small_time_step = (T-t)/100
	
	### The Greeks ####
	Delta = ( AnalyticBlackScholesKnockOutCall(S+0.01,K,r,sigma,t,T,KnockOutBarrier) - AnalyticBlackScholesKnockOutCall(S-0.01,K,r,sigma,t,T,KnockOutBarrier) ) / (2*0.01)
	Gamma = ( AnalyticBlackScholesKnockOutCall(S+0.01,K,r,sigma,t,T,KnockOutBarrier) - 2*AnalyticBlackScholesKnockOutCall(S,K,r,sigma,t,T,KnockOutBarrier) + AnalyticBlackScholesKnockOutCall(S-0.01,K,r,sigma,t,T,KnockOutBarrier))/(0.01**2)
	Vega = ( AnalyticBlackScholesKnockOutCall(S,K,r,sigma+0.01,t,T,KnockOutBarrier) - AnalyticBlackScholesKnockOutCall(S,K,r,sigma-0.01,t,T,KnockOutBarrier) ) / (2*0.01)
	Theta = -( AnalyticBlackScholesKnockOutCall(S,K,r,sigma,t+small_time_step,T,KnockOutBarrier) - AnalyticBlackScholesKnockOutCall(S,K,r,sigma,t-small_time_step,T,KnockOutBarrier) ) / (2*small_time_step)
	Rho = ( AnalyticBlackScholesKnockOutCall(S,K,r+1e-4,sigma,t,T,KnockOutBarrier) - AnalyticBlackScholesKnockOutCall(S,K,r-1e-4,sigma,t,T,KnockOutBarrier) ) / (2*1e-4)
	
	return(CallPrice, Delta, Gamma, Vega, Theta, Rho)
	
#####

def AnalyticBlackScholesKnockInPutWithGreeks(S,K,r,sigma,t,T,KnockInBarrier):
	
	'''
	Calculates the Black Scholes Knock In Put price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	KnockInBarrier is the Barrier (constant in time)
	
	Output is: 
	The Put price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	'''
	#### Put Price Using Analytic Formula #######
	PutPrice = AnalyticBlackScholesKnockInPut(S,K,r,sigma,t,T,KnockInBarrier)
	
	small_time_step = (T-t)/100
	
	### The Greeks ####
	Delta = ( AnalyticBlackScholesKnockInPut(S+0.01,K,r,sigma,t,T,KnockInBarrier) - AnalyticBlackScholesKnockInPut(S-0.01,K,r,sigma,t,T,KnockInBarrier) ) / (2*0.01)
	Gamma = ( AnalyticBlackScholesKnockInPut(S+0.01,K,r,sigma,t,T,KnockInBarrier) - 2*AnalyticBlackScholesKnockInPut(S,K,r,sigma,t,T,KnockInBarrier) + AnalyticBlackScholesKnockInPut(S-0.01,K,r,sigma,t,T,KnockInBarrier))/(0.01**2)
	Vega = ( AnalyticBlackScholesKnockInPut(S,K,r,sigma+0.01,t,T,KnockInBarrier) - AnalyticBlackScholesKnockInPut(S,K,r,sigma-0.01,t,T,KnockInBarrier) ) / (2*0.01)
	Theta = -( AnalyticBlackScholesKnockInPut(S,K,r,sigma,t+small_time_step,T,KnockInBarrier) - AnalyticBlackScholesKnockInPut(S,K,r,sigma,t-small_time_step,T,KnockInBarrier) ) / (2*small_time_step)
	Rho = ( AnalyticBlackScholesKnockInPut(S,K,r+1e-4,sigma,t,T,KnockInBarrier) - AnalyticBlackScholesKnockInPut(S,K,r-1e-4,sigma,t,T,KnockInBarrier) ) / (2*1e-4)
	

	return(PutPrice, Delta, Gamma, Vega, Theta, Rho)
	
	

#####	

def AnalyticBlackScholesKnockOutPutWithGreeks(S,K,r,sigma,t,T,KnockOutBarrier):
	
	'''
	Calculates the Black Scholes Knock Out Put price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formula for the price and the finite difference method

	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
	KnockOutBarrier is the Barrier (constant in time)
	
	Output is:
	The put price in dollars
	Delta
	Gamma
	Vega
	Theta
	Rho
	
	'''
	#### Put Price Using Analytic Formula #######
	PutPrice = AnalyticBlackScholesKnockOutPut(S,K,r,sigma,t,T,KnockOutBarrier)
	
	small_time_step = (T-t)/100
	
	### The Greeks ####
	Delta = ( AnalyticBlackScholesKnockOutPut(S+0.01,K,r,sigma,t,T,KnockOutBarrier) - AnalyticBlackScholesKnockOutPut(S-0.01,K,r,sigma,t,T,KnockOutBarrier) ) / (2*0.01)
	Gamma = ( AnalyticBlackScholesKnockOutPut(S+0.01,K,r,sigma,t,T,KnockOutBarrier) - 2*AnalyticBlackScholesKnockOutPut(S,K,r,sigma,t,T,KnockOutBarrier) + AnalyticBlackScholesKnockOutPut(S-0.01,K,r,sigma,t,T,KnockOutBarrier))/(0.01**2)
	Vega = ( AnalyticBlackScholesKnockOutPut(S,K,r,sigma+0.01,t,T,KnockOutBarrier) - AnalyticBlackScholesKnockOutPut(S,K,r,sigma-0.01,t,T,KnockOutBarrier) ) / (2*0.01)
	Theta = -( AnalyticBlackScholesKnockOutPut(S,K,r,sigma,t+small_time_step,T,KnockOutBarrier) - AnalyticBlackScholesKnockOutPut(S,K,r,sigma,t-small_time_step,T,KnockOutBarrier) ) / (2*small_time_step)
	Rho = ( AnalyticBlackScholesKnockOutPut(S,K,r+1e-4,sigma,t,T,KnockOutBarrier) - AnalyticBlackScholesKnockOutPut(S,K,r-1e-4,sigma,t,T,KnockOutBarrier) ) / (2*1e-4)
	

	return(PutPrice, Delta, Gamma, Vega, Theta, Rho)	
	
#######################
#######################

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

################ MONTE-CARLO BARRIER OPTIONS ####################################
#################################################################################

def MonteCarloKnockInEuropeanCallWithGreeks(S, K, r, sigma, t, T, KnockInBarrier, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculate the Black Scholes European Call WITH A KNOCK-IN BARRIER using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
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
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)
	
	### Barrier Correction Factor - See Hull Chapter 26 ###
	time_step = (T-t)/n_steps
	
	if S <= KnockInBarrier: 
		CorrectionFactor = np.exp(-0.5826*sigma*np.sqrt(time_step))	#### up and in
	else: 
		CorrectionFactor = np.exp(0.5826*sigma*np.sqrt(time_step))	##### down and in
	
	KnockInBarrier = CorrectionFactor*KnockInBarrier
	###
	
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
	MinPrice_array_AT = np.exp( np.min(log_path_array_AT[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	MaxPrice_array_AT = np.exp( np.max(log_path_array_AT[:, :-1], axis=1, keepdims=True) )	
	
	##### perturb in t for theta ##############
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	terminal_price_array_smaller_t_AT = np.exp(log_path_array_AT[:, -1])
	terminal_price_array_larger_t_AT = np.exp(log_path_array_AT[:, -3])

	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_t_AT = np.exp( np.min(log_path_array_AT, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t_AT = np.exp( np.min(log_path_array_AT[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t_AT = np.exp( np.max(log_path_array_AT, axis=1, keepdims=True) )
	MaxPrice_array_larger_t_AT = np.exp( np.max(log_path_array_AT[:, :-2], axis=1, keepdims=True) )	
	
	
	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()		
	MinPrice_array = MinPrice_array.flatten()
	MaxPrice_array = MaxPrice_array.flatten()
	
	terminal_price_array_AT = terminal_price_array_AT.flatten()	
	MinPrice_array_AT = MinPrice_array_AT.flatten()
	MaxPrice_array_AT = MaxPrice_array_AT.flatten()
	
	terminal_price_array_smaller_t_AT = terminal_price_array_smaller_t_AT.flatten()
	terminal_price_array_larger_t_AT = terminal_price_array_larger_t_AT.flatten()
	
		   	
	MinPrice_array_smaller_t = MinPrice_array_smaller_t.flatten() 
	MinPrice_array_larger_t = MinPrice_array_larger_t.flatten() 
	MaxPrice_array_smaller_t = MaxPrice_array_smaller_t.flatten()
	MaxPrice_array_larger_t = MaxPrice_array_larger_t.flatten()
	
	MinPrice_array_smaller_t_AT = MinPrice_array_smaller_t_AT.flatten() 
	MinPrice_array_larger_t_AT = MinPrice_array_larger_t_AT.flatten() 
	MaxPrice_array_smaller_t_AT = MaxPrice_array_smaller_t_AT.flatten()
	MaxPrice_array_larger_t_AT = MaxPrice_array_larger_t_AT.flatten() 		
	
	#### concatenate ####

	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT) )
	MinPrice_array = np.concatenate( (MinPrice_array, MinPrice_array_AT ) )
	MaxPrice_array = np.concatenate( (MaxPrice_array, MaxPrice_array_AT ) )

	terminal_price_array_smaller_t = np.concatenate(( terminal_price_array_smaller_t, terminal_price_array_smaller_t_AT  ) )
	terminal_price_array_larger_t = np.concatenate(( terminal_price_array_larger_t, terminal_price_array_larger_t_AT  ) )
	   	
	MinPrice_array_smaller_t = np.concatenate( (MinPrice_array_smaller_t, MinPrice_array_smaller_t_AT ) ) 
	MinPrice_array_larger_t = np.concatenate( (MinPrice_array_larger_t, MinPrice_array_larger_t_AT ) ) 
	MaxPrice_array_smaller_t = np.concatenate( (MaxPrice_array_smaller_t, MaxPrice_array_smaller_t_AT ) )
	MaxPrice_array_larger_t = np.concatenate( (MaxPrice_array_larger_t, MaxPrice_array_larger_t_AT ) )	
	
	#### special arrays for pathwise derivatives
	
	log_path_array_full = np.concatenate( (log_path_array, log_path_array_AT))
	
	Smax_index = np.argmax(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	Smin_index = np.argmin(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	taumax_array = Smax_index*time_step + t 
	taumin_array = Smin_index*time_step + t 
	
	Z_final_value_array = 1/(sigma*np.sqrt(T-t))*(np.log(terminal_price_array/S) - (r - 0.5*sigma**2)*(T-t)) #### Final Z ~ N(0,1) values needed for some of the pathwise derivatives	
	Z_value_taumax = 1/(sigma)*( np.log(MaxPrice_array/S) - (r - 0.5*sigma**2)*(taumax_array-t) )  ### returns the Z value at S_max times sqrt(tau - t) - to avoid divide by zero error
	Z_value_taumin = 1/(sigma)*( np.log(MinPrice_array/S) - (r - 0.5*sigma**2)*(taumin_array-t) )  ### returns the Z value at S_min times sqrt(tau - t) - to avoid divide by zero error 	

	############################################################### 

	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockInBarrier: 		# down and in
		KnockInFilter_array = np.where(MinPrice_array > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_t = np.where(MinPrice_array_smaller_t > KnockInBarrier, 0, 1)
		KnockInFilter_array_larger_t = np.where(MinPrice_array_larger_t > KnockInBarrier, 0, 1)
	else:				# up and in
		KnockInFilter_array = np.where(MaxPrice_array > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_t = np.where(MaxPrice_array_smaller_t > KnockInBarrier, 1, 0)
		KnockInFilter_array_larger_t = np.where(MaxPrice_array_larger_t > KnockInBarrier, 1, 0)
		
	###### CALCULATE THE OPTION PRICE ###################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(terminal_price_array-K, 0)
	
	#### retain the terminal prices where the MaxPrice over the simulation path exceeded the barrier
	payoff_array = payoff_array*KnockInFilter_array ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) using pathwise derivative #########
	################# Use Papatheodorou MSc thesis Eqs. (3.90) and generalization ################ 
	dSTdSt_array = terminal_price_array/S
	dSmaxdSt_array = MaxPrice_array/S
	dSmindSt_array = MinPrice_array/S
	
	Extra_smoothing = 0.3
	if S > KnockInBarrier:      # down and in 	 
		delta_array = np.exp(-r*(T-t))*( (dSTdSt_array)*Heaviside_smoothed(KnockInBarrier - MinPrice_array, Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) + (dSmindSt_array)*(-1)*Heaviside_dx_smoothed( KnockInBarrier - MinPrice_array, Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) ) 
	else: 			     # up and in
		delta_array = np.exp(-r*(T-t))*( (dSTdSt_array)*Heaviside_smoothed(MaxPrice_array - KnockInBarrier, Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing)    +    (dSmaxdSt_array)*Heaviside_dx_smoothed( MaxPrice_array - KnockInBarrier , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )
	
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####
	################# Use Papatheodorou MSc thesis Eq. (3.95) and generalization ################ 	
	
	Extra_smoothing = 1.0	
	if S > KnockInBarrier:      # down and in 	 
		gamma_array = np.exp(-r*(T-t))*( (dSTdSt_array)**2*Heaviside_smoothed(KnockInBarrier-MinPrice_array,Extra_smoothing)*Heaviside_dx_smoothed(terminal_price_array-K,Extra_smoothing) + 2*(dSTdSt_array*dSmindSt_array)*(-1)*Heaviside_dx_smoothed(KnockInBarrier-MinPrice_array,Extra_smoothing)*Heaviside_smoothed(terminal_price_array-K,Extra_smoothing) + (dSmindSt_array)**2*Heaviside_dx2_smoothed(KnockInBarrier-MinPrice_array,Extra_smoothing)*np.maximum(terminal_price_array-K,0) )
	else: 			     # up and in
		gamma_array = np.exp(-r*(T-t))*( (dSTdSt_array)**2*Heaviside_smoothed(MaxPrice_array-KnockInBarrier,Extra_smoothing)*Heaviside_dx_smoothed(terminal_price_array-K,Extra_smoothing) + 2*(dSTdSt_array*dSmaxdSt_array)*Heaviside_dx_smoothed(MaxPrice_array-KnockInBarrier,Extra_smoothing)*Heaviside_smoothed(terminal_price_array-K,Extra_smoothing) + (dSmaxdSt_array)**2*Heaviside_dx2_smoothed(MaxPrice_array-KnockInBarrier,Extra_smoothing)*np.maximum(terminal_price_array-K,0) )		
		
	gamma_value = np.mean(gamma_array)   	
	gamma_StandardError = stats.sem(gamma_array)		

	#### Calculate Vega = dV/dsigma (partial) ######
	########## Pathwise method #####################	

	dSTdsigma_array =   ( -sigma*(T-t)            + Z_final_value_array*np.sqrt(T-t) )*terminal_price_array
	dSmaxdsigma_array = ( -sigma*(taumax_array-t) + Z_value_taumax)*MaxPrice_array					
	dSmindsigma_array = ( -sigma*(taumin_array-t) + Z_value_taumin)*MinPrice_array
	
	Extra_smoothing = 0.3
	if S > KnockInBarrier:      # down and in 	 
		vega_array = np.exp(-r*(T-t))*( (dSTdsigma_array)*Heaviside_smoothed(KnockInBarrier - MinPrice_array, Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) + (dSmindsigma_array)*(-1)*Heaviside_dx_smoothed( KnockInBarrier - MinPrice_array, Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )
	else: 			     # up and in
		vega_array = np.exp(-r*(T-t))*( (dSTdsigma_array)*Heaviside_smoothed(MaxPrice_array - KnockInBarrier, Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing)    +    (dSmaxdsigma_array)*Heaviside_dx_smoothed( MaxPrice_array - KnockInBarrier , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )

	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  ######
	''' # Pathwise
	dSTdt_array = -( (r-0.5*sigma**2) + sigma*Z_final_value_array/(2*np.sqrt(T-t)) )*terminal_price_array
	
	Z_value_t1 = 1/(sigma)*( np.log(np.exp(log_path_array_full[:,1])/S) - (r - 0.5*sigma**2)*(time_step) )
	print(Z_value_t1[-5:-1]/np.sqrt(time_step))  
	
	dSmaxdt_array = np.where(Smax_index == 0, -( (r-0.5*sigma**2) + sigma*Z_value_t1/(2*time_step) )*S,  -( (r-0.5*sigma**2) + sigma*Z_value_taumax/(2*(taumax_array-t)) )*MaxPrice_array )
	dSmindt_array = np.where(Smin_index == 0, -( (r-0.5*sigma**2) + sigma*Z_value_t1/(2*time_step) )*S,  -( (r-0.5*sigma**2) + sigma*Z_value_taumin/(2*(taumin_array-t)) )*MinPrice_array )	
	
	Extra_smoothing = 0.1
	if S > KnockInBarrier:      # down and in 	 
		theta_array = np.exp(-r*(T-t))*( (dSTdt_array)*Heaviside_smoothed(KnockInBarrier - MinPrice_array, Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) + (dSmindt_array)*(-1)*Heaviside_dx_smoothed( KnockInBarrier - MinPrice_array, Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )
	else: 			     # up and in
		theta_array =  np.exp(-r*(T-t))*( (dSTdt_array)*Heaviside_smoothed(MaxPrice_array - KnockInBarrier, Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) +    (dSmaxdt_array)*Heaviside_dx_smoothed( MaxPrice_array - KnockInBarrier , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )
	
	theta_value = r*option_value + np.mean(theta_array)
	theta_StandardError = np.sqrt( (r*option_value_StandardError)**2  + stats.sem(theta_array)**2 )
	'''
	
	#### Calculate Theta = -dV/dt (partial)  #######
	payoff_array_smaller_t = np.maximum(terminal_price_array_smaller_t-K, 0)
	payoff_array_smaller_t = payoff_array_smaller_t*KnockInFilter_array_smaller_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	payoff_array_larger_t = np.maximum(terminal_price_array_larger_t-K, 0)
	payoff_array_larger_t = payoff_array_larger_t*KnockInFilter_array_larger_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### time increment for finite difference is 1 time step ####
	time_step = (T-t)/n_steps
	
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	dSTdr_array = (T-t)*terminal_price_array
	dSmaxdr_array = (taumax_array-t)*MaxPrice_array
	dSmindr_array = (taumin_array-t)*MinPrice_array		

	Extra_smoothing = 0.3
	if S > KnockInBarrier:      # down and in 	 
		rho_array = np.exp(-r*(T-t))*( (dSTdr_array)*Heaviside_smoothed(KnockInBarrier - MinPrice_array, Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) + (dSmindr_array)*(-1)*Heaviside_dx_smoothed( KnockInBarrier - MinPrice_array, Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )
	else: 			     # up and in
		rho_array = np.exp(-r*(T-t))*( (dSTdr_array)*Heaviside_smoothed(MaxPrice_array - KnockInBarrier, Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing)    +    (dSmaxdr_array)*Heaviside_dx_smoothed( MaxPrice_array - KnockInBarrier , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )
	
	
	rho_value = -(T-t)*option_value + np.mean(rho_array)
	rho_StandardError = np.sqrt( ((T-t)*option_value_StandardError)**2 + stats.sem(rho_array)**2)
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)
	
#################################################################################

def MonteCarloKnockOutEuropeanCallWithGreeks(S, K, r, sigma, t, T, KnockOutBarrier, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculate the Black Scholes European Call WITH A KNOCK-OUT BARRIER using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
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
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)
	
	### Barrier Correction Factor - See Hull Chapter 26 ###
	time_step = (T-t)/n_steps
	
	if S <= KnockOutBarrier: 
		CorrectionFactor = np.exp(-0.5826*sigma*np.sqrt(time_step))	#### up and out
	else: 
		CorrectionFactor = np.exp(0.5826*sigma*np.sqrt(time_step))	##### down and out
	
	KnockOutBarrier = CorrectionFactor*KnockOutBarrier
	###
	
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
	MinPrice_array_AT = np.exp( np.min(log_path_array_AT[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	MaxPrice_array_AT = np.exp( np.max(log_path_array_AT[:, :-1], axis=1, keepdims=True) )	
	
	##### perturb in t for theta ##############
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	terminal_price_array_smaller_t_AT = np.exp(log_path_array_AT[:, -1])
	terminal_price_array_larger_t_AT = np.exp(log_path_array_AT[:, -3])

	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_t_AT = np.exp( np.min(log_path_array_AT, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t_AT = np.exp( np.min(log_path_array_AT[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t_AT = np.exp( np.max(log_path_array_AT, axis=1, keepdims=True) )
	MaxPrice_array_larger_t_AT = np.exp( np.max(log_path_array_AT[:, :-2], axis=1, keepdims=True) )	
	
	
	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()		
	MinPrice_array = MinPrice_array.flatten()
	MaxPrice_array = MaxPrice_array.flatten()
	
	terminal_price_array_AT = terminal_price_array_AT.flatten()	
	MinPrice_array_AT = MinPrice_array_AT.flatten()
	MaxPrice_array_AT = MaxPrice_array_AT.flatten()
	
	terminal_price_array_smaller_t_AT = terminal_price_array_smaller_t_AT.flatten()
	terminal_price_array_larger_t_AT = terminal_price_array_larger_t_AT.flatten()
	
		   	
	MinPrice_array_smaller_t = MinPrice_array_smaller_t.flatten() 
	MinPrice_array_larger_t = MinPrice_array_larger_t.flatten() 
	MaxPrice_array_smaller_t = MaxPrice_array_smaller_t.flatten()
	MaxPrice_array_larger_t = MaxPrice_array_larger_t.flatten()
	
	MinPrice_array_smaller_t_AT = MinPrice_array_smaller_t_AT.flatten() 
	MinPrice_array_larger_t_AT = MinPrice_array_larger_t_AT.flatten() 
	MaxPrice_array_smaller_t_AT = MaxPrice_array_smaller_t_AT.flatten()
	MaxPrice_array_larger_t_AT = MaxPrice_array_larger_t_AT.flatten() 		
	
	#### concatenate ####

	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT) )
	MinPrice_array = np.concatenate( (MinPrice_array, MinPrice_array_AT ) )
	MaxPrice_array = np.concatenate( (MaxPrice_array, MaxPrice_array_AT ) )

	terminal_price_array_smaller_t = np.concatenate(( terminal_price_array_smaller_t, terminal_price_array_smaller_t_AT  ) )
	terminal_price_array_larger_t = np.concatenate(( terminal_price_array_larger_t, terminal_price_array_larger_t_AT  ) )
	   	
	MinPrice_array_smaller_t = np.concatenate( (MinPrice_array_smaller_t, MinPrice_array_smaller_t_AT ) ) 
	MinPrice_array_larger_t = np.concatenate( (MinPrice_array_larger_t, MinPrice_array_larger_t_AT ) ) 
	MaxPrice_array_smaller_t = np.concatenate( (MaxPrice_array_smaller_t, MaxPrice_array_smaller_t_AT ) )
	MaxPrice_array_larger_t = np.concatenate( (MaxPrice_array_larger_t, MaxPrice_array_larger_t_AT ) )	
	
	#### special arrays for pathwise derivatives
	
	log_path_array_full = np.concatenate( (log_path_array, log_path_array_AT))
	
	Smax_index = np.argmax(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	Smin_index = np.argmin(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	taumax_array = Smax_index*time_step + t 
	taumin_array = Smin_index*time_step + t 
	
	Z_final_value_array = 1/(sigma*np.sqrt(T-t))*(np.log(terminal_price_array/S) - (r - 0.5*sigma**2)*(T-t)) #### Final Z ~ N(0,1) values needed for some of the pathwise derivatives	
	Z_value_taumax = 1/(sigma)*( np.log(MaxPrice_array/S) - (r - 0.5*sigma**2)*(taumax_array-t) )  ### returns the Z value at S_max times sqrt(tau - t) - to avoid divide by zero error
	Z_value_taumin = 1/(sigma)*( np.log(MinPrice_array/S) - (r - 0.5*sigma**2)*(taumin_array-t) )  ### returns the Z value at S_min times sqrt(tau - t) - to avoid divide by zero error 	
	
	############################################################### 
	
	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockOutBarrier:
		KnockOutFilter_array = np.where(MinPrice_array > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_t = np.where(MinPrice_array_smaller_t > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_larger_t = np.where(MinPrice_array_larger_t > KnockOutBarrier, 1, 0)	
	else:
		KnockOutFilter_array = np.where(MaxPrice_array > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_t = np.where(MaxPrice_array_smaller_t > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_larger_t = np.where(MaxPrice_array_larger_t > KnockOutBarrier, 0, 1)	

	###### CALCULATE THE OPTION PRICE ###################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(terminal_price_array-K, 0)
	
	#### retain the terminal prices where the MinPrice over the simulation path exceeded the barrier
	payoff_array = payoff_array*KnockOutFilter_array ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) using pathwise derivative #########
	################# Use Papatheodorou MSc thesis Eqs. (3.90) and generalization ################ 
	
	dSTdSt_array = terminal_price_array/S
	dSmaxdSt_array = MaxPrice_array/S
	dSmindSt_array = MinPrice_array/S
	
	Extra_smoothing = 0.3
	if S > KnockOutBarrier:      # down and out 	 
		delta_array = np.exp(-r*(T-t))*( (dSTdSt_array)*Heaviside_smoothed(MinPrice_array - KnockOutBarrier , Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) + (dSmindSt_array)*Heaviside_dx_smoothed( MinPrice_array - KnockOutBarrier , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) ) 
	else: 			     # up and out
		delta_array = np.exp(-r*(T-t))*( (dSTdSt_array)*Heaviside_smoothed(KnockOutBarrier - MaxPrice_array , Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing)    +    (dSmaxdSt_array)*(-1)*Heaviside_dx_smoothed( KnockOutBarrier - MaxPrice_array  , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )
	
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)	
	

	#### Calculate Gamma = d^2V/dS^2 (partial) #####
	################# Use Papatheodorou MSc thesis Eq. (3.95) and generalization ################ 	
	Extra_smoothing = 1.0	
	if S > KnockOutBarrier:      # down and out 	 
		gamma_array = np.exp(-r*(T-t))*( (dSTdSt_array)**2*Heaviside_smoothed( MinPrice_array- KnockOutBarrier,Extra_smoothing)*Heaviside_dx_smoothed(terminal_price_array-K,Extra_smoothing) + 2*(dSTdSt_array*dSmindSt_array)*Heaviside_dx_smoothed(MinPrice_array-KnockOutBarrier,Extra_smoothing)*Heaviside_smoothed(terminal_price_array-K,Extra_smoothing) + (dSmindSt_array)**2*Heaviside_dx2_smoothed(MinPrice_array-KnockOutBarrier,Extra_smoothing)*np.maximum(terminal_price_array-K,0) )
	else: 			     # up and out
		gamma_array = np.exp(-r*(T-t))*( (dSTdSt_array)**2*Heaviside_smoothed(KnockOutBarrier-MaxPrice_array,Extra_smoothing)*Heaviside_dx_smoothed(terminal_price_array-K,Extra_smoothing) + 2*(dSTdSt_array*dSmaxdSt_array)*(-1)*Heaviside_dx_smoothed(KnockOutBarrier-MaxPrice_array,Extra_smoothing)*Heaviside_smoothed(terminal_price_array-K,Extra_smoothing) + (dSmaxdSt_array)**2*Heaviside_dx2_smoothed(KnockOutBarrier-MaxPrice_array,Extra_smoothing)*np.maximum(terminal_price_array-K,0) )		
		
	gamma_value = np.mean(gamma_array)   	
	gamma_StandardError = stats.sem(gamma_array)		

	#### Calculate Vega = dV/dsigma (partial) ######
	
	dSTdsigma_array =   ( -sigma*(T-t)            + Z_final_value_array*np.sqrt(T-t) )*terminal_price_array
	dSmaxdsigma_array = ( -sigma*(taumax_array-t) + Z_value_taumax)*MaxPrice_array					
	dSmindsigma_array = ( -sigma*(taumin_array-t) + Z_value_taumin)*MinPrice_array
	
	Extra_smoothing = 0.3
	if S > KnockOutBarrier:      # down and out 	 
		vega_array = np.exp(-r*(T-t))*( (dSTdsigma_array)*Heaviside_smoothed(MinPrice_array - KnockOutBarrier , Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) + (dSmindsigma_array)*Heaviside_dx_smoothed( MinPrice_array - KnockOutBarrier , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) ) 
	else: 			     # up and out
		vega_array = np.exp(-r*(T-t))*( (dSTdsigma_array)*Heaviside_smoothed(KnockOutBarrier - MaxPrice_array , Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing)    +    (dSmaxdsigma_array)*(-1)*Heaviside_dx_smoothed( KnockOutBarrier - MaxPrice_array  , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )

	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)

	
	#### Calculate Theta = -dV/dt (partial)  #######
	payoff_array_smaller_t = np.maximum(terminal_price_array_smaller_t-K, 0)
	payoff_array_smaller_t = payoff_array_smaller_t*KnockOutFilter_array_smaller_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	payoff_array_larger_t = np.maximum(terminal_price_array_larger_t-K, 0)
	payoff_array_larger_t = payoff_array_larger_t*KnockOutFilter_array_larger_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### time increment for finite difference is 1 time step ####
	time_step = (T-t)/n_steps
	
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	dSTdr_array = (T-t)*terminal_price_array
	dSmaxdr_array = (taumax_array-t)*MaxPrice_array
	dSmindr_array = (taumin_array-t)*MinPrice_array		

	Extra_smoothing = 0.3
	if S > KnockOutBarrier:      # down and out 	 
		rho_array = np.exp(-r*(T-t))*( (dSTdr_array)*Heaviside_smoothed(MinPrice_array - KnockOutBarrier , Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) + (dSmindr_array)*Heaviside_dx_smoothed( MinPrice_array - KnockOutBarrier , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) ) 
	else: 			     # up and out
		rho_array = np.exp(-r*(T-t))*( (dSTdr_array)*Heaviside_smoothed(KnockOutBarrier - MaxPrice_array , Extra_smoothing)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing)    +    (dSmaxdr_array)*(-1)*Heaviside_dx_smoothed( KnockOutBarrier - MaxPrice_array  , Extra_smoothing )*(terminal_price_array - K)*Heaviside_smoothed( terminal_price_array - K, Extra_smoothing) )
	
	rho_value = -(T-t)*option_value + np.mean(rho_array)
	rho_StandardError = np.sqrt( ((T-t)*option_value_StandardError)**2 + stats.sem(rho_array)**2)
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)

#################################################################################
##################################### PUTS  #####################################

def MonteCarloKnockInEuropeanPutWithGreeks(S, K, r, sigma, t, T, KnockInBarrier, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculate the Black Scholes European Put WITH A KNOCK-IN BARRIER using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
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
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)

	### Barrier Correction Factor - See Hull Chapter 26 ###
	time_step = (T-t)/n_steps
	
	if S <= KnockInBarrier: 
		CorrectionFactor = np.exp(-0.5826*sigma*np.sqrt(time_step))	#### up and in
	else: 
		CorrectionFactor = np.exp(0.5826*sigma*np.sqrt(time_step))	##### down and in
	
	KnockInBarrier = CorrectionFactor*KnockInBarrier
	###
	
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
	MinPrice_array_AT = np.exp( np.min(log_path_array_AT[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	MaxPrice_array_AT = np.exp( np.max(log_path_array_AT[:, :-1], axis=1, keepdims=True) )	
	
	##### perturb in t for theta ##############
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	terminal_price_array_smaller_t_AT = np.exp(log_path_array_AT[:, -1])
	terminal_price_array_larger_t_AT = np.exp(log_path_array_AT[:, -3])

	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_t_AT = np.exp( np.min(log_path_array_AT, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t_AT = np.exp( np.min(log_path_array_AT[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t_AT = np.exp( np.max(log_path_array_AT, axis=1, keepdims=True) )
	MaxPrice_array_larger_t_AT = np.exp( np.max(log_path_array_AT[:, :-2], axis=1, keepdims=True) )	
	
	
	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()		
	MinPrice_array = MinPrice_array.flatten()
	MaxPrice_array = MaxPrice_array.flatten()
	
	terminal_price_array_AT = terminal_price_array_AT.flatten()	
	MinPrice_array_AT = MinPrice_array_AT.flatten()
	MaxPrice_array_AT = MaxPrice_array_AT.flatten()
	
	terminal_price_array_smaller_t_AT = terminal_price_array_smaller_t_AT.flatten()
	terminal_price_array_larger_t_AT = terminal_price_array_larger_t_AT.flatten()
	
		   	
	MinPrice_array_smaller_t = MinPrice_array_smaller_t.flatten() 
	MinPrice_array_larger_t = MinPrice_array_larger_t.flatten() 
	MaxPrice_array_smaller_t = MaxPrice_array_smaller_t.flatten()
	MaxPrice_array_larger_t = MaxPrice_array_larger_t.flatten()
	
	MinPrice_array_smaller_t_AT = MinPrice_array_smaller_t_AT.flatten() 
	MinPrice_array_larger_t_AT = MinPrice_array_larger_t_AT.flatten() 
	MaxPrice_array_smaller_t_AT = MaxPrice_array_smaller_t_AT.flatten()
	MaxPrice_array_larger_t_AT = MaxPrice_array_larger_t_AT.flatten() 		
	
	#### concatenate ####

	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT) )
	MinPrice_array = np.concatenate( (MinPrice_array, MinPrice_array_AT ) )
	MaxPrice_array = np.concatenate( (MaxPrice_array, MaxPrice_array_AT ) )

	terminal_price_array_smaller_t = np.concatenate(( terminal_price_array_smaller_t, terminal_price_array_smaller_t_AT  ) )
	terminal_price_array_larger_t = np.concatenate(( terminal_price_array_larger_t, terminal_price_array_larger_t_AT  ) )
	   	
	MinPrice_array_smaller_t = np.concatenate( (MinPrice_array_smaller_t, MinPrice_array_smaller_t_AT ) ) 
	MinPrice_array_larger_t = np.concatenate( (MinPrice_array_larger_t, MinPrice_array_larger_t_AT ) ) 
	MaxPrice_array_smaller_t = np.concatenate( (MaxPrice_array_smaller_t, MaxPrice_array_smaller_t_AT ) )
	MaxPrice_array_larger_t = np.concatenate( (MaxPrice_array_larger_t, MaxPrice_array_larger_t_AT ) )	
	
	#### special arrays for pathwise derivatives
	
	log_path_array_full = np.concatenate( (log_path_array, log_path_array_AT))
	
	Smax_index = np.argmax(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	Smin_index = np.argmin(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	taumax_array = Smax_index*time_step + t 
	taumin_array = Smin_index*time_step + t 
	
	Z_final_value_array = 1/(sigma*np.sqrt(T-t))*(np.log(terminal_price_array/S) - (r - 0.5*sigma**2)*(T-t)) #### Final Z ~ N(0,1) values needed for some of the pathwise derivatives	
	Z_value_taumax = 1/(sigma)*( np.log(MaxPrice_array/S) - (r - 0.5*sigma**2)*(taumax_array-t) )  ### returns the Z value at S_max times sqrt(tau - t) - to avoid divide by zero error
	Z_value_taumin = 1/(sigma)*( np.log(MinPrice_array/S) - (r - 0.5*sigma**2)*(taumin_array-t) )  ### returns the Z value at S_min times sqrt(tau - t) - to avoid divide by zero error 		

	############################################################### 

	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockInBarrier:
		KnockInFilter_array = np.where(MinPrice_array > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_t = np.where(MinPrice_array_smaller_t > KnockInBarrier, 0, 1)
		KnockInFilter_array_larger_t = np.where(MinPrice_array_larger_t > KnockInBarrier, 0, 1)	
	else:
		KnockInFilter_array = np.where(MaxPrice_array > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_t = np.where(MaxPrice_array_smaller_t > KnockInBarrier, 1, 0)
		KnockInFilter_array_larger_t = np.where(MaxPrice_array_larger_t > KnockInBarrier, 1, 0)

	
	###### CALCULATE THE OPTION PRICE ###################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(K-terminal_price_array, 0)
	
	#### retain the terminal prices where the MaxPrice over the simulation path exceeded the barrier
	payoff_array = payoff_array*KnockInFilter_array ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	################# Use Papatheodorou MSc thesis Eqs. (3.90) and generalization ################ 
	
	dSTdSt_array = terminal_price_array/S
	dSmaxdSt_array = MaxPrice_array/S
	dSmindSt_array = MinPrice_array/S
	
	Extra_smoothing = 0.3
	if S > KnockInBarrier:      # down and in 	 
		delta_array = np.exp(-r*(T-t))*( (dSTdSt_array)*-Heaviside_smoothed(KnockInBarrier - MinPrice_array, Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing) + (dSmindSt_array)*(-1)*Heaviside_dx_smoothed( KnockInBarrier - MinPrice_array, Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K-terminal_price_array, Extra_smoothing) ) 
	else: 			     # up and in
		delta_array = np.exp(-r*(T-t))*( (dSTdSt_array)*-Heaviside_smoothed(MaxPrice_array - KnockInBarrier, Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing)    +    (dSmaxdSt_array)*Heaviside_dx_smoothed( MaxPrice_array - KnockInBarrier , Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K-terminal_price_array , Extra_smoothing) )
	
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)


	#### Calculate Gamma = d^2V/dS^2 (partial) #####
	################# Use Papatheodorou MSc thesis Eq. (3.95) and generalization ################ 	

	Extra_smoothing = 1.0	
	if S > KnockInBarrier:      # down and in 	 
		gamma_array = np.exp(-r*(T-t))*( (dSTdSt_array)**2*(-1)**2*Heaviside_smoothed(KnockInBarrier-MinPrice_array,Extra_smoothing)*Heaviside_dx_smoothed(K-terminal_price_array,Extra_smoothing) + 2*(dSTdSt_array*dSmindSt_array)*(-1)**2*Heaviside_dx_smoothed(KnockInBarrier-MinPrice_array,Extra_smoothing)*Heaviside_smoothed(K-terminal_price_array,Extra_smoothing) + (dSmindSt_array)**2*Heaviside_dx2_smoothed(KnockInBarrier-MinPrice_array,Extra_smoothing)*np.maximum(K-terminal_price_array,0) )
	else: 			     # up and in
		gamma_array = np.exp(-r*(T-t))*( (dSTdSt_array)**2*(-1)**2*Heaviside_smoothed(MaxPrice_array-KnockInBarrier,Extra_smoothing)*Heaviside_dx_smoothed(K-terminal_price_array,Extra_smoothing) + 2*(dSTdSt_array*dSmaxdSt_array)*(-1)*Heaviside_dx_smoothed(MaxPrice_array-KnockInBarrier,Extra_smoothing)*Heaviside_smoothed(K-terminal_price_array,Extra_smoothing) + (dSmaxdSt_array)**2*Heaviside_dx2_smoothed(MaxPrice_array-KnockInBarrier,Extra_smoothing)*np.maximum(K-terminal_price_array,0) )		
		
	gamma_value = np.mean(gamma_array)   	
	gamma_StandardError = stats.sem(gamma_array)	
	
	#### Calculate Vega = dV/dsigma (partial) ######
	
	dSTdsigma_array =   ( -sigma*(T-t)            + Z_final_value_array*np.sqrt(T-t) )*terminal_price_array
	dSmaxdsigma_array = ( -sigma*(taumax_array-t) + Z_value_taumax)*MaxPrice_array					
	dSmindsigma_array = ( -sigma*(taumin_array-t) + Z_value_taumin)*MinPrice_array

	Extra_smoothing = 0.3
	if S > KnockInBarrier:      # down and in 	 
		vega_array = np.exp(-r*(T-t))*( (dSTdsigma_array)*-Heaviside_smoothed(KnockInBarrier - MinPrice_array, Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing) + (dSmindsigma_array)*(-1)*Heaviside_dx_smoothed( KnockInBarrier - MinPrice_array, Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K-terminal_price_array, Extra_smoothing) ) 
	else: 			     # up and in
		vega_array = np.exp(-r*(T-t))*( (dSTdsigma_array)*-Heaviside_smoothed(MaxPrice_array - KnockInBarrier, Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing)    +    (dSmaxdsigma_array)*Heaviside_dx_smoothed( MaxPrice_array - KnockInBarrier , Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K-terminal_price_array , Extra_smoothing) )

	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	payoff_array_smaller_t = np.maximum(K-terminal_price_array_smaller_t, 0)
	payoff_array_smaller_t = payoff_array_smaller_t*KnockInFilter_array_smaller_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	payoff_array_larger_t = np.maximum(K-terminal_price_array_larger_t, 0)
	payoff_array_larger_t = payoff_array_larger_t*KnockInFilter_array_larger_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### time increment for finite difference is 1 time step ####
	time_step = (T-t)/n_steps
	
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
	
	dSTdr_array = (T-t)*terminal_price_array
	dSmaxdr_array = (taumax_array-t)*MaxPrice_array
	dSmindr_array = (taumin_array-t)*MinPrice_array		

	Extra_smoothing = 0.3
	if S > KnockInBarrier:      # down and in 	 
		rho_array = np.exp(-r*(T-t))*( (dSTdr_array)*-Heaviside_smoothed(KnockInBarrier - MinPrice_array, Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing) + (dSmindr_array)*(-1)*Heaviside_dx_smoothed( KnockInBarrier - MinPrice_array, Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K-terminal_price_array, Extra_smoothing) ) 
	else: 			     # up and in
		rho_array = np.exp(-r*(T-t))*( (dSTdr_array)*-Heaviside_smoothed(MaxPrice_array - KnockInBarrier, Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing)    +    (dSmaxdr_array)*Heaviside_dx_smoothed( MaxPrice_array - KnockInBarrier , Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K-terminal_price_array , Extra_smoothing) )
	
	rho_value = -(T-t)*option_value + np.mean(rho_array)
	rho_StandardError = np.sqrt( ((T-t)*option_value_StandardError)**2 + stats.sem(rho_array)**2)
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)

##############

def MonteCarloKnockOutEuropeanPutWithGreeks(S, K, r, sigma, t, T, KnockOutBarrier, n_simulations=250000, n_steps=100, BaseSeed=StandardBaseSeed):

	'''
	Calculate the Black Scholes European Put WITH A KNOCK-OUT BARRIER using Monte-Carlo and the following arguments:
	
	S is the stock price at time t
	K is the strike price
	r is the risk-free interest rate
	sigma is the volatility
	t is the time the option price is evaluated
	T is the expiration time in years
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
	
	### force n_steps and n_simulations to integer values ####
	n_steps = int(n_steps)
	n_simulations = int(n_simulations)
	
	#### reduce n_simulations by 2, as we will be using antithetic pairs
	n_simulations = n_simulations/2
	n_simulations = int(n_simulations)
	
	### Barrier Correction Factor - See Hull Chapter 26 ###
	time_step = (T-t)/n_steps
	
	if S <= KnockOutBarrier: 
		CorrectionFactor = np.exp(-0.5826*sigma*np.sqrt(time_step))	#### up and out
	else: 
		CorrectionFactor = np.exp(0.5826*sigma*np.sqrt(time_step))	##### down and out
	
	KnockOutBarrier = CorrectionFactor*KnockOutBarrier
	###
	
		
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
	MinPrice_array_AT = np.exp( np.min(log_path_array_AT[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	MaxPrice_array_AT = np.exp( np.max(log_path_array_AT[:, :-1], axis=1, keepdims=True) )	
	
	##### perturb in t for theta ##############
	terminal_price_array_smaller_t = np.exp(log_path_array[:, -1])
	terminal_price_array_larger_t = np.exp(log_path_array[:, -3])
	terminal_price_array_smaller_t_AT = np.exp(log_path_array_AT[:, -1])
	terminal_price_array_larger_t_AT = np.exp(log_path_array_AT[:, -3])

	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_t_AT = np.exp( np.min(log_path_array_AT, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t_AT = np.exp( np.min(log_path_array_AT[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t_AT = np.exp( np.max(log_path_array_AT, axis=1, keepdims=True) )
	MaxPrice_array_larger_t_AT = np.exp( np.max(log_path_array_AT[:, :-2], axis=1, keepdims=True) )	
	
	
	##### flatten arrays #####
	terminal_price_array = terminal_price_array.flatten()		
	MinPrice_array = MinPrice_array.flatten()
	MaxPrice_array = MaxPrice_array.flatten()
	
	terminal_price_array_AT = terminal_price_array_AT.flatten()	
	MinPrice_array_AT = MinPrice_array_AT.flatten()
	MaxPrice_array_AT = MaxPrice_array_AT.flatten()
	
	terminal_price_array_smaller_t_AT = terminal_price_array_smaller_t_AT.flatten()
	terminal_price_array_larger_t_AT = terminal_price_array_larger_t_AT.flatten()
	
		   	
	MinPrice_array_smaller_t = MinPrice_array_smaller_t.flatten() 
	MinPrice_array_larger_t = MinPrice_array_larger_t.flatten() 
	MaxPrice_array_smaller_t = MaxPrice_array_smaller_t.flatten()
	MaxPrice_array_larger_t = MaxPrice_array_larger_t.flatten()
	
	MinPrice_array_smaller_t_AT = MinPrice_array_smaller_t_AT.flatten() 
	MinPrice_array_larger_t_AT = MinPrice_array_larger_t_AT.flatten() 
	MaxPrice_array_smaller_t_AT = MaxPrice_array_smaller_t_AT.flatten()
	MaxPrice_array_larger_t_AT = MaxPrice_array_larger_t_AT.flatten() 		
	
	#### concatenate ####

	terminal_price_array = np.concatenate( (terminal_price_array, terminal_price_array_AT) )
	MinPrice_array = np.concatenate( (MinPrice_array, MinPrice_array_AT ) )
	MaxPrice_array = np.concatenate( (MaxPrice_array, MaxPrice_array_AT ) )

	terminal_price_array_smaller_t = np.concatenate(( terminal_price_array_smaller_t, terminal_price_array_smaller_t_AT  ) )
	terminal_price_array_larger_t = np.concatenate(( terminal_price_array_larger_t, terminal_price_array_larger_t_AT  ) )
	   	
	MinPrice_array_smaller_t = np.concatenate( (MinPrice_array_smaller_t, MinPrice_array_smaller_t_AT ) ) 
	MinPrice_array_larger_t = np.concatenate( (MinPrice_array_larger_t, MinPrice_array_larger_t_AT ) ) 
	MaxPrice_array_smaller_t = np.concatenate( (MaxPrice_array_smaller_t, MaxPrice_array_smaller_t_AT ) )
	MaxPrice_array_larger_t = np.concatenate( (MaxPrice_array_larger_t, MaxPrice_array_larger_t_AT ) )	
	
	#### special arrays for pathwise derivatives
	
	log_path_array_full = np.concatenate( (log_path_array, log_path_array_AT))
	
	Smax_index = np.argmax(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	Smin_index = np.argmin(log_path_array_full[:, :-1], axis=1) 	### ignore the last column as this is the additional time step for finite difference theta calculation
	taumax_array = Smax_index*time_step + t 
	taumin_array = Smin_index*time_step + t 
	
	Z_final_value_array = 1/(sigma*np.sqrt(T-t))*(np.log(terminal_price_array/S) - (r - 0.5*sigma**2)*(T-t)) #### Final Z ~ N(0,1) values needed for some of the pathwise derivatives	
	Z_value_taumax = 1/(sigma)*( np.log(MaxPrice_array/S) - (r - 0.5*sigma**2)*(taumax_array-t) )  ### returns the Z value at S_max times sqrt(tau - t) - to avoid divide by zero error
	Z_value_taumin = 1/(sigma)*( np.log(MinPrice_array/S) - (r - 0.5*sigma**2)*(taumin_array-t) )  ### returns the Z value at S_min times sqrt(tau - t) - to avoid divide by zero error 	
	
	
	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockOutBarrier:
		KnockOutFilter_array = np.where(MinPrice_array > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_t = np.where(MinPrice_array_smaller_t > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_larger_t = np.where(MinPrice_array_larger_t > KnockOutBarrier, 1, 0)	
	else:
		KnockOutFilter_array = np.where(MaxPrice_array > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_t = np.where(MaxPrice_array_smaller_t > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_larger_t = np.where(MaxPrice_array_larger_t > KnockOutBarrier, 0, 1)	
	
	###### CALCULATE THE OPTION PRICE #######################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(K-terminal_price_array, 0)
	
	#### retain the terminal prices where the MinPrice over the simulation path exceeded the barrier
	payoff_array = payoff_array*KnockOutFilter_array ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	################# Use Papatheodorou MSc thesis Eqs. (3.90) and generalization ################ 

	dSTdSt_array = terminal_price_array/S
	dSmaxdSt_array = MaxPrice_array/S
	dSmindSt_array = MinPrice_array/S
	
	Extra_smoothing = 0.3
	if S > KnockOutBarrier:      # down and out 	 
		delta_array = np.exp(-r*(T-t))*( (dSTdSt_array)*(-1)*Heaviside_smoothed(MinPrice_array - KnockOutBarrier , Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array, Extra_smoothing) + (dSmindSt_array)*Heaviside_dx_smoothed( MinPrice_array - KnockOutBarrier , Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing) ) 
	else: 			     # up and out
		delta_array = np.exp(-r*(T-t))*( (dSTdSt_array)*(-1)*Heaviside_smoothed(KnockOutBarrier - MaxPrice_array , Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing)    +    (dSmaxdSt_array)*(-1)*Heaviside_dx_smoothed( KnockOutBarrier - MaxPrice_array  , Extra_smoothing )*( K-terminal_price_array )*Heaviside_smoothed( K-terminal_price_array , Extra_smoothing) )
	
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)	

	#### Calculate Gamma = d^2V/dS^2 (partial) #####
	################# Use Papatheodorou MSc thesis Eq. (3.95) and generalization ################ 	

	Extra_smoothing = 1.0	
	if S > KnockOutBarrier:      # down and out 	 
		gamma_array = np.exp(-r*(T-t))*( (dSTdSt_array)**2*(-1)**2*Heaviside_smoothed( MinPrice_array- KnockOutBarrier,Extra_smoothing)*Heaviside_dx_smoothed(K-terminal_price_array,Extra_smoothing) + 2*(dSTdSt_array*dSmindSt_array)*(-1)*Heaviside_dx_smoothed(MinPrice_array-KnockOutBarrier,Extra_smoothing)*Heaviside_smoothed(K-terminal_price_array,Extra_smoothing) + (dSmindSt_array)**2*Heaviside_dx2_smoothed(MinPrice_array-KnockOutBarrier,Extra_smoothing)*np.maximum(K-terminal_price_array,0) )
	else: 			     # up and out
		gamma_array = np.exp(-r*(T-t))*( (dSTdSt_array)**2*(-1)**2*Heaviside_smoothed(KnockOutBarrier-MaxPrice_array,Extra_smoothing)*Heaviside_dx_smoothed(K-terminal_price_array,Extra_smoothing) + 2*(dSTdSt_array*dSmaxdSt_array)*(-1)**2*Heaviside_dx_smoothed(KnockOutBarrier-MaxPrice_array,Extra_smoothing)*Heaviside_smoothed(K-terminal_price_array,Extra_smoothing) + (dSmaxdSt_array)**2*Heaviside_dx2_smoothed(KnockOutBarrier-MaxPrice_array,Extra_smoothing)*np.maximum(K-terminal_price_array,0) )		
		
	gamma_value = np.mean(gamma_array)   	
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	dSTdsigma_array =   ( -sigma*(T-t)            + Z_final_value_array*np.sqrt(T-t) )*terminal_price_array
	dSmaxdsigma_array = ( -sigma*(taumax_array-t) + Z_value_taumax)*MaxPrice_array					
	dSmindsigma_array = ( -sigma*(taumin_array-t) + Z_value_taumin)*MinPrice_array
	
	Extra_smoothing = 0.3
	if S > KnockOutBarrier:      # down and out 	 
		vega_array = np.exp(-r*(T-t))*( (dSTdsigma_array)*(-1)*Heaviside_smoothed(MinPrice_array - KnockOutBarrier , Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array, Extra_smoothing) + (dSmindsigma_array)*Heaviside_dx_smoothed( MinPrice_array - KnockOutBarrier , Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing) ) 
	else: 			     # up and out
		vega_array = np.exp(-r*(T-t))*( (dSTdsigma_array)*(-1)*Heaviside_smoothed(KnockOutBarrier - MaxPrice_array , Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing) + (dSmaxdsigma_array)*(-1)*Heaviside_dx_smoothed( KnockOutBarrier - MaxPrice_array  , Extra_smoothing )*( K-terminal_price_array )*Heaviside_smoothed( K-terminal_price_array , Extra_smoothing) )

	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)

	
	
	#### Calculate Theta = -dV/dt (partial)  #######
	payoff_array_smaller_t = np.maximum(K-terminal_price_array_smaller_t, 0)
	payoff_array_smaller_t = payoff_array_smaller_t*KnockOutFilter_array_smaller_t	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	payoff_array_larger_t = np.maximum(K-terminal_price_array_larger_t, 0)
	payoff_array_larger_t = payoff_array_larger_t*KnockOutFilter_array_larger_t	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### time increment for finite difference is 1 time step ####
	time_step = (T-t)/n_steps
	
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########	
	dSTdr_array = (T-t)*terminal_price_array
	dSmaxdr_array = (taumax_array-t)*MaxPrice_array
	dSmindr_array = (taumin_array-t)*MinPrice_array		

	Extra_smoothing = 0.3
	if S > KnockOutBarrier:      # down and out 	 
		rho_array = np.exp(-r*(T-t))*( (dSTdr_array)*(-1)*Heaviside_smoothed(MinPrice_array - KnockOutBarrier , Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array, Extra_smoothing) + (dSmindr_array)*Heaviside_dx_smoothed( MinPrice_array - KnockOutBarrier , Extra_smoothing )*( K - terminal_price_array )*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing) ) 
	else: 			     # up and out
		rho_array = np.exp(-r*(T-t))*( (dSTdr_array)*(-1)*Heaviside_smoothed(KnockOutBarrier - MaxPrice_array , Extra_smoothing)*Heaviside_smoothed( K - terminal_price_array , Extra_smoothing) +  (dSmaxdr_array)*(-1)*Heaviside_dx_smoothed( KnockOutBarrier - MaxPrice_array  , Extra_smoothing )*( K-terminal_price_array )*Heaviside_smoothed( K-terminal_price_array , Extra_smoothing) )
	
	rho_value = -(T-t)*option_value + np.mean(rho_array)
	rho_StandardError = np.sqrt( ((T-t)*option_value_StandardError)**2 + stats.sem(rho_array)**2)	


	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)


##################################################################################
##################################################################################


def main():
	print('\n')
	
	print('Call Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')
	
	print('\nEuropean Call Analytic with Knock In at S=0.01\n', AnalyticBlackScholesKnockInCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 0.01))
	print('\nEuropean Call Monte-Carlo with Knock In at S=0\n', MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 0))
	
	print('\nEuropean Call Analytic with Knock In at S=85\n', AnalyticBlackScholesKnockInCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	print('\nEuropean Call Monte-Carlo with Knock In at S=85\n', MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))

	print('\nEuropean Call Analytic with Knock Out at S=85\n', AnalyticBlackScholesKnockOutCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=85\n', MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	
	print('\nEuropean Call Analytic with Knock Out at S=1000\n', AnalyticBlackScholesKnockOutCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 1000))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=1000\n', MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 1000))

	print('\nEuropean Call Analytic with Knock In at S=75\n', AnalyticBlackScholesKnockInCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	print('\nEuropean Call Monte-Carlo with Knock In at S=75\n', MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))

	print('\nEuropean Call Analytic with Knock Out at S=75\n', AnalyticBlackScholesKnockOutCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=75\n', MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))

	print('\nEuropean Call Analytic with Knock In at S=110\n', AnalyticBlackScholesKnockInCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))
	print('\nEuropean Call Monte-Carlo with Knock In at S=110\n', MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))

	print('\nEuropean Call Analytic with Knock Out at S=110\n', AnalyticBlackScholesKnockOutCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=110\n', MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))

	print('\n')
	
	print('Put Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')

	print('\nEuropean Put Analytic with Knock In at S=0.01\n', AnalyticBlackScholesKnockInPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 0.01))	
	print('\nEuropean Put Monte-Carlo with Knock In at S=0\n', MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 0))
	
	
	print('\nEuropean Put Analytic with Knock In at S=85\n', AnalyticBlackScholesKnockInPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))		
	print('\nEuropean Put Monte-Carlo with Knock In at S=85\n', MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	

	print('\nEuropean Put Analytic with Knock Out at S=85\n', AnalyticBlackScholesKnockOutPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))		
	print('\nEuropean Put Monte-Carlo with Knock Out at S=85\n', MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	
	print('\nEuropean Put Analytic with Knock Out at S=1000\n', AnalyticBlackScholesKnockOutPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 1000))		
	print('\nEuropean Put Monte-Carlo with Knock Out at S=1000\n', MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 1000))
	
	print('\nEuropean Put Analytic with Knock In at S=75\n', AnalyticBlackScholesKnockInPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))			
	print('\nEuropean Put Monte-Carlo with Knock In at S=75\n', MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	
	print('\nEuropean Put Analytic with Knock Out at S=75\n', AnalyticBlackScholesKnockOutPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))			
	print('\nEuropean Put Monte-Carlo with Knock Out at S=75\n', MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	
	print('\nEuropean Put Analytic with Knock In at S=110\n', AnalyticBlackScholesKnockInPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))				
	print('\nEuropean Put Monte-Carlo with Knock In at S=110\n', MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))
	
	print('\nEuropean Put Analytic with Knock Out at S=110\n', AnalyticBlackScholesKnockOutPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))				
	print('\nEuropean Put Monte-Carlo with Knock Out at S=110\n', MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))




if __name__ == "__main__":
	main()	


	
