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

def ApproxAvgPriceCall(S,K,r,sigma,t,T,Savgsofar=none):	
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
	
	
	M1 = 
	
	M2 = 
	
	sigmanew = 
	
	c =
	
	p =
	
	
	AvgPriceCall = 0

	return(AvgPriceCall)


##################################################################################################################################################

def ApproxAvgPricePut(S,K,r,sigma,t,T,Savgsofar=none):		
 
##################################################################################################################################################

# def ApproxAvgStrikeCall(S,K,r,sigma,t,T,Savgsofar=none):		

##################################################################################################################################################

# def ApproxAvgStrikePut(S,K,r,sigma,t,T,Savgsofar=none):

##################################################################################################################################################

def ApproxAvgPriceCallWithGreeks(S,K,r,sigma,t,T,Savgsofar=none):		

##################################################################################################################################################

def ApproxAvgPricePutWithGreeks(S,K,r,sigma,t,T,Savgsofar=none):		

		
  		
