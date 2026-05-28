import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

###

'''
GIVES ANALYTIC AND MONTE-CARLO IMPLEMENTATION OF EUROPEAN OPTIONS WITH BARRIERS
RETURNS THE OPTION PRICE, GREEKS, AND STANDARD ERRORS
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


################ BARRIER OPTIONS ################################################
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
	
	### Barrier Correction Factor - See Hull Chapter 26 ###
	time_step = (T-t)/n_steps
	
	if S <= KnockInBarrier: 
		CorrectionFactor = np.exp(-0.5826*sigma*np.sqrt(time_step))	#### up and in
	else: 
		CorrectionFactor = np.exp(0.5826*sigma*np.sqrt(time_step))	##### down and in
	
	KnockInBarrier = CorrectionFactor*KnockInBarrier
	
	###############################################################
	
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
	
	##### perturb in S for Delta and Gamma ####
	price_step = 0.01
	terminal_price_array_smaller_S = terminal_price_array*(S-price_step)/S
	terminal_price_array_larger_S = terminal_price_array*(S+price_step)/S 	
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 )
	
	log_path_array_smaller_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_sigma ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_sigma ), axis=1 ) ### add the initial price as the first entry 	 		 	 	
	
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

	log_path_array_smaller_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_r ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_r ), axis=1 ) ### add the initial price as the first entry 		
	
	terminal_price_array_smaller_r = np.exp(log_path_array_smaller_r[:, -2])
	terminal_price_array_larger_r = np.exp(log_path_array_larger_r[:, -2])	

	############ Max and minimum prices ###########################
	
	MinPrice_array = np.exp( np.min(log_path_array[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	MaxPrice_array = np.exp( np.max(log_path_array[:, :-1], axis=1, keepdims=True) )
	
	
	MinPrice_array_smaller_S = MinPrice_array*(S-0.01)/S
	MinPrice_array_larger_S = MinPrice_array*(S+0.01)/S
	MaxPrice_array_smaller_S = MaxPrice_array*(S-0.01)/S
	MaxPrice_array_larger_S = MaxPrice_array*(S+0.01)/S
	
	MinPrice_array_smaller_sigma =  np.exp( np.min(log_path_array_smaller_sigma[:, :-1], axis=1, keepdims=True) )
	MinPrice_array_larger_sigma =   np.exp( np.min(log_path_array_larger_sigma[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_smaller_sigma =	np.exp( np.max(log_path_array_smaller_sigma[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_larger_sigma =   np.exp( np.max(log_path_array_larger_sigma[:, :-1], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_r = np.exp( np.min(log_path_array_smaller_r[:, :-1], axis=1, keepdims=True) )
	MinPrice_array_larger_r = np.exp( np.min(log_path_array_larger_r[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_smaller_r = np.exp( np.max(log_path_array_smaller_r[:, :-1], axis=1, keepdims=True) )	
	MaxPrice_array_larger_r = np.exp( np.max(log_path_array_larger_r[:, :-1], axis=1, keepdims=True) )
		

	############################################################### 

	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockInBarrier:
		KnockInFilter_array = np.where(MinPrice_array > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_S = np.where(MinPrice_array_smaller_S > KnockInBarrier, 0, 1)
		KnockInFilter_array_larger_S =  np.where(MinPrice_array_larger_S > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_sigma = np.where(MinPrice_array_smaller_sigma > KnockInBarrier, 0, 1)
		KnockInFilter_array_larger_sigma = np.where(MinPrice_array_larger_sigma > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_t = np.where(MinPrice_array_smaller_t > KnockInBarrier, 0, 1)
		KnockInFilter_array_larger_t = np.where(MinPrice_array_larger_t > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_r = np.where(MinPrice_array_smaller_r > KnockInBarrier, 0, 1) 
		KnockInFilter_array_larger_r = np.where(MinPrice_array_larger_r > KnockInBarrier, 0, 1)		
	else:
		KnockInFilter_array = np.where(MaxPrice_array > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_S = np.where(MaxPrice_array_smaller_S > KnockInBarrier, 1, 0)
		KnockInFilter_array_larger_S =  np.where(MaxPrice_array_larger_S > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_sigma = np.where(MaxPrice_array_smaller_sigma > KnockInBarrier, 1, 0)
		KnockInFilter_array_larger_sigma = np.where(MaxPrice_array_larger_sigma > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_t = np.where(MaxPrice_array_smaller_t > KnockInBarrier, 1, 0)
		KnockInFilter_array_larger_t = np.where(MaxPrice_array_larger_t > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_r = np.where(MaxPrice_array_smaller_r > KnockInBarrier, 1, 0) 
		KnockInFilter_array_larger_r = np.where(MaxPrice_array_larger_r > KnockInBarrier, 1, 0)	

	
	###### CALCULATE THE OPTION PRICE ###################	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(terminal_price_array-K, 0)
	

	#### retain the terminal prices where the MaxPrice over the simulation path exceeded the barrier
	payoff_array = payoff_array.flatten()*KnockInFilter_array.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	
	payoff_array_smaller_S = np.maximum(terminal_price_array_smaller_S-K, 0)
	payoff_array_smaller_S = payoff_array_smaller_S.flatten()*KnockInFilter_array_smaller_S.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S	
	
	payoff_array_larger_S = np.maximum(terminal_price_array_larger_S-K, 0)
	payoff_array_larger_S = payoff_array_larger_S.flatten()*KnockInFilter_array_larger_S.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*price_step)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(price_step**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	payoff_array_smaller_sigma = np.maximum(terminal_price_array_smaller_sigma-K, 0)
	payoff_array_smaller_sigma = payoff_array_smaller_sigma.flatten()*KnockInFilter_array_smaller_sigma.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	
	payoff_array_larger_sigma = np.maximum(terminal_price_array_larger_sigma-K, 0)
	payoff_array_larger_sigma = payoff_array_larger_sigma.flatten()*KnockInFilter_array_larger_sigma.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	payoff_array_smaller_t = np.maximum(terminal_price_array_smaller_t-K, 0)
	payoff_array_smaller_t = payoff_array_smaller_t.flatten()*KnockInFilter_array_smaller_t.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	
	payoff_array_larger_t = np.maximum(terminal_price_array_larger_t-K, 0)
	payoff_array_larger_t = payoff_array_larger_t.flatten()*KnockInFilter_array_larger_t.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### time increment for finite difference is 1 time step ####
	time_step = (T-t)/n_steps
	
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########

	payoff_array_smaller_r = np.maximum(terminal_price_array_smaller_r-K, 0)	
	payoff_array_smaller_r = payoff_array_smaller_r.flatten()*KnockInFilter_array_smaller_r.flatten() 	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	payoff_array_larger_r = np.maximum(terminal_price_array_larger_r-K, 0)
	payoff_array_larger_r = payoff_array_larger_r.flatten()*KnockInFilter_array_larger_r.flatten()	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	rho_value = np.mean(rho_array)
	rho_StandardError = stats.sem(rho_array)
	
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
	
	##### perturb in S for Delta and Gamma ####
	price_step = 0.01
	terminal_price_array_smaller_S = terminal_price_array*(S-price_step)/S
	terminal_price_array_larger_S = terminal_price_array*(S+price_step)/S 	
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 )
	
	log_path_array_smaller_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_sigma ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_sigma ), axis=1 ) ### add the initial price as the first entry 	 		 	 	
	
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

	log_path_array_smaller_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_r ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_r ), axis=1 ) ### add the initial price as the first entry 		
	
	terminal_price_array_smaller_r = np.exp(log_path_array_smaller_r[:, -2])
	terminal_price_array_larger_r = np.exp(log_path_array_larger_r[:, -2])	

	############ Max and minimum prices ###########################
	
	MinPrice_array = np.exp( np.min(log_path_array[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	MaxPrice_array = np.exp( np.max(log_path_array[:, :-1], axis=1, keepdims=True) )
	
	
	MinPrice_array_smaller_S = MinPrice_array*(S-price_step)/S
	MinPrice_array_larger_S = MinPrice_array*(S+price_step)/S
	MaxPrice_array_smaller_S = MaxPrice_array*(S-price_step)/S
	MaxPrice_array_larger_S = MaxPrice_array*(S+price_step)/S
	
	MinPrice_array_smaller_sigma =  np.exp( np.min(log_path_array_smaller_sigma[:, :-1], axis=1, keepdims=True) )
	MinPrice_array_larger_sigma =   np.exp( np.min(log_path_array_larger_sigma[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_smaller_sigma =	np.exp( np.max(log_path_array_smaller_sigma[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_larger_sigma =   np.exp( np.max(log_path_array_larger_sigma[:, :-1], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_r = np.exp( np.min(log_path_array_smaller_r[:, :-1], axis=1, keepdims=True) )
	MinPrice_array_larger_r = np.exp( np.min(log_path_array_larger_r[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_smaller_r = np.exp( np.max(log_path_array_smaller_r[:, :-1], axis=1, keepdims=True) )	
	MaxPrice_array_larger_r = np.exp( np.max(log_path_array_larger_r[:, :-1], axis=1, keepdims=True) )
		

	############################################################### 
	
	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockOutBarrier:
		KnockOutFilter_array = np.where(MinPrice_array > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_S = np.where(MinPrice_array_smaller_S > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_larger_S =  np.where(MinPrice_array_larger_S > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_sigma = np.where(MinPrice_array_smaller_sigma > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_larger_sigma = np.where(MinPrice_array_larger_sigma > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_t = np.where(MinPrice_array_smaller_t > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_larger_t = np.where(MinPrice_array_larger_t > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_r = np.where(MinPrice_array_smaller_r > KnockOutBarrier, 1, 0) 
		KnockOutFilter_array_larger_r = np.where(MinPrice_array_larger_r > KnockOutBarrier, 1, 0)		
	else:
		KnockOutFilter_array = np.where(MaxPrice_array > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_S = np.where(MaxPrice_array_smaller_S > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_larger_S =  np.where(MaxPrice_array_larger_S > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_sigma = np.where(MaxPrice_array_smaller_sigma > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_larger_sigma = np.where(MaxPrice_array_larger_sigma > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_t = np.where(MaxPrice_array_smaller_t > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_larger_t = np.where(MaxPrice_array_larger_t > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_r = np.where(MaxPrice_array_smaller_r > KnockOutBarrier, 0, 1) 
		KnockOutFilter_array_larger_r = np.where(MaxPrice_array_larger_r > KnockOutBarrier, 0, 1)	

	###### CALCULATE THE OPTION PRICE ###################	
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(terminal_price_array-K, 0)
	
	#### retain the terminal prices where the MinPrice over the simulation path exceeded the barrier
	payoff_array = payoff_array.flatten()*KnockOutFilter_array.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########	
	payoff_array_smaller_S = np.maximum(terminal_price_array_smaller_S-K, 0)
	payoff_array_smaller_S = payoff_array_smaller_S.flatten()*KnockOutFilter_array_smaller_S.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S	
	
	payoff_array_larger_S = np.maximum(terminal_price_array_larger_S-K, 0)
	payoff_array_larger_S = payoff_array_larger_S.flatten()*KnockOutFilter_array_larger_S.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*price_step)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(price_step**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	payoff_array_smaller_sigma = np.maximum(terminal_price_array_smaller_sigma-K, 0)
	payoff_array_smaller_sigma = payoff_array_smaller_sigma.flatten()*KnockOutFilter_array_smaller_sigma.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	
	payoff_array_larger_sigma = np.maximum(terminal_price_array_larger_sigma-K, 0)
	payoff_array_larger_sigma = payoff_array_larger_sigma.flatten()*KnockOutFilter_array_larger_sigma.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	payoff_array_smaller_t = np.maximum(terminal_price_array_smaller_t-K, 0)
	payoff_array_smaller_t = payoff_array_smaller_t.flatten()*KnockOutFilter_array_smaller_t.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	
	payoff_array_larger_t = np.maximum(terminal_price_array_larger_t-K, 0)
	payoff_array_larger_t = payoff_array_larger_t.flatten()*KnockOutFilter_array_larger_t.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### time increment for finite difference is 1 time step ####
	time_step = (T-t)/n_steps
	
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########

	payoff_array_smaller_r = np.maximum(terminal_price_array_smaller_r-K, 0)	
	payoff_array_smaller_r = payoff_array_smaller_r.flatten()*KnockOutFilter_array_smaller_r.flatten() 	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	

	payoff_array_larger_r = np.maximum(terminal_price_array_larger_r-K, 0)
	payoff_array_larger_r = payoff_array_larger_r.flatten()*KnockOutFilter_array_larger_r.flatten()	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	rho_value = np.mean(rho_array)
	rho_StandardError = stats.sem(rho_array)
	
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
	
	##### perturb in S for Delta and Gamma ####
	price_step = 0.01	
	terminal_price_array_smaller_S = terminal_price_array*(S-price_step)/S
	terminal_price_array_larger_S = terminal_price_array*(S+price_step)/S 	
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 )
	
	log_path_array_smaller_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_sigma ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_sigma ), axis=1 ) ### add the initial price as the first entry 	 		 	 	
	
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

	log_path_array_smaller_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_r ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_r ), axis=1 ) ### add the initial price as the first entry 		
	
	terminal_price_array_smaller_r = np.exp(log_path_array_smaller_r[:, -2])
	terminal_price_array_larger_r = np.exp(log_path_array_larger_r[:, -2])	

	############ Max and minimum prices ###########################
	
	MinPrice_array = np.exp( np.min(log_path_array[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	MaxPrice_array = np.exp( np.max(log_path_array[:, :-1], axis=1, keepdims=True) )
	
	
	MinPrice_array_smaller_S = MinPrice_array*(S-price_step)/S
	MinPrice_array_larger_S = MinPrice_array*(S+price_step)/S
	MaxPrice_array_smaller_S = MaxPrice_array*(S-price_step)/S
	MaxPrice_array_larger_S = MaxPrice_array*(S+price_step)/S
	
	MinPrice_array_smaller_sigma =  np.exp( np.min(log_path_array_smaller_sigma[:, :-1], axis=1, keepdims=True) )
	MinPrice_array_larger_sigma =   np.exp( np.min(log_path_array_larger_sigma[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_smaller_sigma =	np.exp( np.max(log_path_array_smaller_sigma[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_larger_sigma =   np.exp( np.max(log_path_array_larger_sigma[:, :-1], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_r = np.exp( np.min(log_path_array_smaller_r[:, :-1], axis=1, keepdims=True) )
	MinPrice_array_larger_r = np.exp( np.min(log_path_array_larger_r[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_smaller_r = np.exp( np.max(log_path_array_smaller_r[:, :-1], axis=1, keepdims=True) )	
	MaxPrice_array_larger_r = np.exp( np.max(log_path_array_larger_r[:, :-1], axis=1, keepdims=True) )

	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockInBarrier:
		KnockInFilter_array = np.where(MinPrice_array > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_S = np.where(MinPrice_array_smaller_S > KnockInBarrier, 0, 1)
		KnockInFilter_array_larger_S =  np.where(MinPrice_array_larger_S > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_sigma = np.where(MinPrice_array_smaller_sigma > KnockInBarrier, 0, 1)
		KnockInFilter_array_larger_sigma = np.where(MinPrice_array_larger_sigma > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_t = np.where(MinPrice_array_smaller_t > KnockInBarrier, 0, 1)
		KnockInFilter_array_larger_t = np.where(MinPrice_array_larger_t > KnockInBarrier, 0, 1)
		KnockInFilter_array_smaller_r = np.where(MinPrice_array_smaller_r > KnockInBarrier, 0, 1) 
		KnockInFilter_array_larger_r = np.where(MinPrice_array_larger_r > KnockInBarrier, 0, 1)		
	else:
		KnockInFilter_array = np.where(MaxPrice_array > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_S = np.where(MaxPrice_array_smaller_S > KnockInBarrier, 1, 0)
		KnockInFilter_array_larger_S =  np.where(MaxPrice_array_larger_S > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_sigma = np.where(MaxPrice_array_smaller_sigma > KnockInBarrier, 1, 0)
		KnockInFilter_array_larger_sigma = np.where(MaxPrice_array_larger_sigma > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_t = np.where(MaxPrice_array_smaller_t > KnockInBarrier, 1, 0)
		KnockInFilter_array_larger_t = np.where(MaxPrice_array_larger_t > KnockInBarrier, 1, 0)
		KnockInFilter_array_smaller_r = np.where(MaxPrice_array_smaller_r > KnockInBarrier, 1, 0) 
		KnockInFilter_array_larger_r = np.where(MaxPrice_array_larger_r > KnockInBarrier, 1, 0)	

	###### CALCULATE THE OPTION PRICE ################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(K-terminal_price_array, 0)
	
	#### retain the terminal prices where the MaxPrice over the simulation path exceeded the barrier
	payoff_array = payoff_array.flatten()*KnockInFilter_array.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	payoff_array_smaller_S = np.maximum(K-terminal_price_array_smaller_S, 0)
	payoff_array_smaller_S = payoff_array_smaller_S.flatten()*KnockInFilter_array_smaller_S.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S	
	

	payoff_array_larger_S = np.maximum(K-terminal_price_array_larger_S, 0)
	payoff_array_larger_S = payoff_array_larger_S.flatten()*KnockInFilter_array_larger_S.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*price_step)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(price_step**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	payoff_array_smaller_sigma = np.maximum(K-terminal_price_array_smaller_sigma, 0)
	payoff_array_smaller_sigma = payoff_array_smaller_sigma.flatten()*KnockInFilter_array_smaller_sigma.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	

	payoff_array_larger_sigma = np.maximum(K-terminal_price_array_larger_sigma, 0)
	payoff_array_larger_sigma = payoff_array_larger_sigma.flatten()*KnockInFilter_array_larger_sigma.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######

	payoff_array_smaller_t = np.maximum(K-terminal_price_array_smaller_t, 0)
	payoff_array_smaller_t = payoff_array_smaller_t.flatten()*KnockInFilter_array_smaller_t.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY


	payoff_array_larger_t = np.maximum(K-terminal_price_array_larger_t, 0)
	payoff_array_larger_t = payoff_array_larger_t.flatten()*KnockInFilter_array_larger_t.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### time increment for finite difference is 1 time step ####
	time_step = (T-t)/n_steps
	
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
		
	payoff_array_smaller_r = np.maximum(K-terminal_price_array_smaller_r, 0)	
	payoff_array_smaller_r = payoff_array_smaller_r.flatten()*KnockInFilter_array_smaller_r.flatten() 	 ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
		
	payoff_array_larger_r = np.maximum(K-terminal_price_array_larger_r, 0)
	payoff_array_larger_r = payoff_array_larger_r.flatten()*KnockInFilter_array_larger_r.flatten()	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	rho_value = np.mean(rho_array)
	rho_StandardError = stats.sem(rho_array)
	
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
	
	### Barrier Correction Factor - See Hull Chapter 26 ###
	time_step = (T-t)/n_steps
	
	if S <= KnockOutBarrier: 
		CorrectionFactor = np.exp(-0.5826*sigma*np.sqrt(time_step))	#### up and in
	else: 
		CorrectionFactor = np.exp(0.5826*sigma*np.sqrt(time_step))	##### down and in
	
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
	
	##### perturb in S for Delta and Gamma ####
	price_step = 0.01
	terminal_price_array_smaller_S = terminal_price_array*(S-price_step)/S
	terminal_price_array_larger_S = terminal_price_array*(S+price_step)/S 	
	
	##### perturb in sigma for vega ###########
	smaller_sigma = (sigma-0.01)
	larger_sigma = (sigma+0.01)
	
	log_step_array_smaller_sigma = (r - 0.5*smaller_sigma**2)*time_step + np.sqrt(time_step)*smaller_sigma*brownian_array
	log_step_array_larger_sigma = (r - 0.5*larger_sigma**2)*time_step + np.sqrt(time_step)*larger_sigma*brownian_array	
	
	log_path_array_smaller_sigma = np.log(S) + np.cumsum( log_step_array_smaller_sigma, axis = 1 ) 
	log_path_array_larger_sigma = np.log(S) + np.cumsum( log_step_array_larger_sigma, axis = 1 )
	
	log_path_array_smaller_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_sigma ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_sigma = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_sigma ), axis=1 ) ### add the initial price as the first entry 	 		 	 	
	
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

	log_path_array_smaller_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_smaller_r ), axis=1 ) ### add the initial price as the first entry
	log_path_array_larger_r = np.concatenate( ( np.full( shape = (n_simulations, 1) , fill_value = np.log(S) ), log_path_array_larger_r ), axis=1 ) ### add the initial price as the first entry 		
	
	terminal_price_array_smaller_r = np.exp(log_path_array_smaller_r[:, -2])
	terminal_price_array_larger_r = np.exp(log_path_array_larger_r[:, -2])	

	############ Max and minimum prices ###########################
	
	MinPrice_array = np.exp( np.min(log_path_array[:, :-1], axis=1, keepdims=True) ) ### ignore the last column as this is the additional time step for theta calculation
	MaxPrice_array = np.exp( np.max(log_path_array[:, :-1], axis=1, keepdims=True) )
	
	
	MinPrice_array_smaller_S = MinPrice_array*(S-price_step)/S
	MinPrice_array_larger_S = MinPrice_array*(S+price_step)/S
	MaxPrice_array_smaller_S = MaxPrice_array*(S-price_step)/S
	MaxPrice_array_larger_S = MaxPrice_array*(S+price_step)/S
	
	MinPrice_array_smaller_sigma =  np.exp( np.min(log_path_array_smaller_sigma[:, :-1], axis=1, keepdims=True) )
	MinPrice_array_larger_sigma =   np.exp( np.min(log_path_array_larger_sigma[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_smaller_sigma =	np.exp( np.max(log_path_array_smaller_sigma[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_larger_sigma =   np.exp( np.max(log_path_array_larger_sigma[:, :-1], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_t = np.exp( np.min(log_path_array, axis=1, keepdims=True) )  ### keep the last column as this is the additional time step for a longer time to maturity
	MinPrice_array_larger_t = np.exp( np.min(log_path_array[:, :-2], axis=1, keepdims=True) )  ### ignore the last two columns as the time to maturity is shorter
	MaxPrice_array_smaller_t = np.exp( np.max(log_path_array, axis=1, keepdims=True) )
	MaxPrice_array_larger_t = np.exp( np.max(log_path_array[:, :-2], axis=1, keepdims=True) )
	
	MinPrice_array_smaller_r = np.exp( np.min(log_path_array_smaller_r[:, :-1], axis=1, keepdims=True) )
	MinPrice_array_larger_r = np.exp( np.min(log_path_array_larger_r[:, :-1], axis=1, keepdims=True) )
	MaxPrice_array_smaller_r = np.exp( np.max(log_path_array_smaller_r[:, :-1], axis=1, keepdims=True) )	
	MaxPrice_array_larger_r = np.exp( np.max(log_path_array_larger_r[:, :-1], axis=1, keepdims=True) )
		

	############################################################### 
	
	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockOutBarrier:
		KnockOutFilter_array = np.where(MinPrice_array > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_S = np.where(MinPrice_array_smaller_S > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_larger_S =  np.where(MinPrice_array_larger_S > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_sigma = np.where(MinPrice_array_smaller_sigma > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_larger_sigma = np.where(MinPrice_array_larger_sigma > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_t = np.where(MinPrice_array_smaller_t > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_larger_t = np.where(MinPrice_array_larger_t > KnockOutBarrier, 1, 0)
		KnockOutFilter_array_smaller_r = np.where(MinPrice_array_smaller_r > KnockOutBarrier, 1, 0) 
		KnockOutFilter_array_larger_r = np.where(MinPrice_array_larger_r > KnockOutBarrier, 1, 0)		
	else:
		KnockOutFilter_array = np.where(MaxPrice_array > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_S = np.where(MaxPrice_array_smaller_S > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_larger_S =  np.where(MaxPrice_array_larger_S > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_sigma = np.where(MaxPrice_array_smaller_sigma > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_larger_sigma = np.where(MaxPrice_array_larger_sigma > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_t = np.where(MaxPrice_array_smaller_t > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_larger_t = np.where(MaxPrice_array_larger_t > KnockOutBarrier, 0, 1)
		KnockOutFilter_array_smaller_r = np.where(MaxPrice_array_smaller_r > KnockOutBarrier, 0, 1) 
		KnockOutFilter_array_larger_r = np.where(MaxPrice_array_larger_r > KnockOutBarrier, 0, 1)	

	###### CALCULATE THE OPTION PRICE ###################
	#### calculate the option payoff for each of the terminal prices #####
	payoff_array = np.maximum(K-terminal_price_array, 0)
	
	#### retain the terminal prices where the MinPrice over the simulation path exceeded the barrier
	payoff_array = payoff_array.flatten()*KnockOutFilter_array.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### option value array, given the interest rate, time to expiry, and payoff under the risk-neutral probability measure 
	option_value_array = np.exp(-r*(T-t))*payoff_array
	
	#### calculate the option value, given the interest rate, time to expiry, and expected payoff under the risk-neutral probability measure 
	option_value = np.mean(option_value_array)
	
	#### Estimates the uncertainty of the option value over the subarrays
	option_value_StandardError = stats.sem(option_value_array)
	
	############## THE GREEKS ######################
	#### Calculate Delta = dV/dS (partial) #########
	payoff_array_smaller_S = np.maximum(K-terminal_price_array_smaller_S, 0)
	payoff_array_smaller_S = payoff_array_smaller_S.flatten()*KnockOutFilter_array_smaller_S.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S	
	
	payoff_array_larger_S = np.maximum(K-terminal_price_array_larger_S, 0)
	payoff_array_larger_S = payoff_array_larger_S.flatten()*KnockOutFilter_array_larger_S.flatten() ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*price_step)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(price_step**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	

	payoff_array_smaller_sigma = np.maximum(K-terminal_price_array_smaller_sigma, 0)
	payoff_array_smaller_sigma = payoff_array_smaller_sigma.flatten()*KnockOutFilter_array_smaller_sigma.flatten()	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	
	payoff_array_larger_sigma = np.maximum(K-terminal_price_array_larger_sigma, 0)
	payoff_array_larger_sigma = payoff_array_larger_sigma.flatten()*KnockOutFilter_array_larger_sigma.flatten()	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	payoff_array_smaller_t = np.maximum(K-terminal_price_array_smaller_t, 0)
	payoff_array_smaller_t = payoff_array_smaller_t.flatten()*KnockOutFilter_array_smaller_t.flatten()	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	
	payoff_array_larger_t = np.maximum(K-terminal_price_array_larger_t, 0)
	payoff_array_larger_t = payoff_array_larger_t.flatten()*KnockOutFilter_array_larger_t.flatten()	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	#### time increment for finite difference is 1 time step ####
	time_step = (T-t)/n_steps
	
	option_value_array_smaller_t = np.exp(-r*(T-(t-time_step)))*payoff_array_smaller_t
	option_value_array_larger_t = np.exp(-r*(T-(t+time_step)))*payoff_array_larger_t
	
	theta_array = -1*(option_value_array_larger_t-option_value_array_smaller_t)/(2*time_step)
	theta_value = np.mean(theta_array)
	theta_StandardError = stats.sem(theta_array)
	
	#### Calculate Rho = dV/dr (partial)   #########
		
	payoff_array_smaller_r = np.maximum(K-terminal_price_array_smaller_r, 0)	
	payoff_array_smaller_r = payoff_array_smaller_r.flatten()*KnockOutFilter_array_smaller_r.flatten() 	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	
	payoff_array_larger_r = np.maximum(K-terminal_price_array_larger_r, 0)
	payoff_array_larger_r = payoff_array_larger_r.flatten()*KnockOutFilter_array_larger_r.flatten()	 ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	option_value_array_smaller_r = np.exp(-(r-1e-4)*(T-t))*payoff_array_smaller_r
	option_value_array_larger_r = np.exp(-(r+1e-4)*(T-t))*payoff_array_larger_r	
	
	rho_array = (option_value_array_larger_r-option_value_array_smaller_r)/(2*1e-4)
	rho_value = np.mean(rho_array)
	rho_StandardError = stats.sem(rho_array)
	
	#### Return the option value the Greeks and standard errors of all the quantities ###
	return(option_value, delta_value, gamma_value, vega_value, theta_value, rho_value, option_value_StandardError, delta_StandardError, gamma_StandardError, vega_StandardError, theta_StandardError, rho_StandardError)



##################################################################################
##################################################################################


def main():
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


	
