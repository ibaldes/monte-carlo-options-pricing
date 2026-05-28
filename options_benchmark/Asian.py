import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from scipy.stats import norm
from multiprocessing import Pool

###

'''
GIVES ANALYTIC AND MONTE-CARLO IMPLEMENTATION OF ASIAN OPTIONS
RETURNS THE OPTION PRICE, GREEKS, AND STANDARD ERRORS
'''

StandardBaseSeed = 0


##################################################################################################################################################

def ApproxAvgPriceCall(S,K,r,sigma,t,T,Savgsofar=None):	
	'''
	Approximate analytic formula for the Asian call.
	
	Note: The Payoff is given by Max[Savg - K,0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
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
	
	Note: The Payoff is given by Max[Savg - K,0], where Savg is the arithmetic average price of the underlying.

	S is the stock price at time t
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

# def ApproxAvgStrikeCall(S,K,r,sigma,t,T,Savgsofar=None):		

##################################################################################################################################################

# def ApproxAvgStrikePut(S,K,r,sigma,t,T,Savgsofar=None):

##################################################################################################################################################

#def ApproxAvgPriceCallWithGreeks(S,K,r,sigma,t,T,Savgsofar=None):		

##################################################################################################################################################

#def ApproxAvgPricePutWithGreeks(S,K,r,sigma,t,T,Savgsofar=None):		

print(ApproxAvgPriceCall(50,50,0.1,0.4,0,1))
print(ApproxAvgPricePut(50,50,0.1,0.4,0,1))				
  		
