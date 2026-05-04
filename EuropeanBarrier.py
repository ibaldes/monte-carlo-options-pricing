import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from scipy.stats import norm
from multiprocessing import Pool

###

'''
GIVES ANALYTIC AND MONTE-CARLO IMPLEMENTATION OF EUROPEAN OPTIONS WITH BARRIERS
ReTURNS THE OPTION PRICE, GREEKS, AND STANDARD ERRORS
'''

StandardBaseSeed = 0

if __name__ == '__main__':
	print("Number of cpus : ", mp.cpu_count())
	pool = Pool(processes=(mp.cpu_count() - 1))


#######################

def AnalyticBlackScholesKnockInCall(S,K,r,sigma,t,T,KnockInBarrier):
	
	'''
	Calculates the Black Scholes Knock In Call price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formulas

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

def AnalyticBlackScholesKnockOutCall(S,K,r,sigma,t,T,KnockInBarrier):
	
	'''
	Calculates the Black Scholes Knock Out Call price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formulas

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
	
	#### Vanilla Call Price	
	d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
	VanillaCallPrice = S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2) 
	
	
	#### Knock In Call Price
	KnockInCallPrice = AnalyticBlackScholesKnockInCall(S,K,r,sigma,t,T,KnockInBarrier)		

	##### Knock Out Call Price
	
	KnockOutCallPrice = VanillaCallPrice - KnockInCallPrice
 		
	return(KnockOutCallPrice)

#######################

def AnalyticBlackScholesKnockInPut(S,K,r,sigma,t,T,KnockInBarrier):
	
	'''
	Calculates the Black Scholes Knock In Put price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formulas

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
	
def AnalyticBlackScholesKnockOutPut(S,K,r,sigma,t,T,KnockInBarrier):
	
	'''
	Calculates the Black Scholes Knock In Put price using the analytic formula
	Also returns the Greeks: Delta, Gamma, Vega, Theta, Rho using the analytic formulas

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
	
	
	if S > KnockInBarrier:	### price of down and out put 
		
		####################################################
		if KnockInBarrier <= K: #### Hull
			
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockInBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			y1factor = np.log( KnockInBarrier/S ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)				
			x1factor = np.log( S/KnockInBarrier ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
#			x2factor = np.log( S/K ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			
			PriceDownIn = -S*norm.cdf(-x1factor) + K*np.exp(-r*(T-t))*norm.cdf(-x1factor+sigma*np.sqrt(T-t)) + S*( KnockInBarrier/S )**(2*lambdafactor)*(norm.cdf(yfactor)-norm.cdf(y1factor))-K*np.exp(-r*(T-t))*(KnockInBarrier/S)**(2*lambdafactor-2)*(norm.cdf(yfactor-sigma*np.sqrt(T-t))-norm.cdf(y1factor-sigma*np.sqrt(T-t)))
			
			d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			VanillaPutPrice = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1) 
			
			PriceDownOut = VanillaPutPrice - PriceDownIn
		else:
			PriceDownOut = 0 

		####################################################
		
		PutPrice = PriceDownOut
		
	############################################################	
	############################################################
	
	else: 	### price of up and out put 

		###########################################
		if KnockInBarrier <= K:
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockInBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			y1factor = np.log( KnockInBarrier/S ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)				
			x1factor = np.log( S/KnockInBarrier ) / (sigma * np.sqrt(T-t) ) + lambdafactor*sigma*np.sqrt(T-t)
			
			PriceUpOut = -S*norm.cdf(-x1factor) + K*norm.cdf(-x1factor+sigma*np.sqrt(T-t)) + S*(KnockInBarrier/S)**(2*lambdafactor)*norm.cdf(-y1factor)-K*np.exp(-r*(T-t))*(KnockInBarrier/S)**(2*lambdafactor-2)*norm.cdf(-y1factor+sigma*np.sqrt(T-t))
			
		else: ##### Using Hull
			lambdafactor = (r + 0.5*sigma**2) / (sigma**2)
			yfactor = np.log( KnockInBarrier**2 / (S*K) )/(sigma*np.sqrt(T-t)) + lambdafactor*sigma*np.sqrt(T-t)
			PriceUpIn = -S*( KnockInBarrier/S )**(2*lambdafactor)*norm.cdf(-yfactor) + K*np.exp(-r*(T-t))*( KnockInBarrier/S )**(2*lambdafactor-2)*norm.cdf(-yfactor + sigma*np.sqrt(T-t))
			
			d1 = ( np.log(S/K)+(r+1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			d2 = ( np.log(S/K)+(r-1/2*sigma**2)*(T-t) ) / ( sigma*np.sqrt( T - t ) )
			VanillaPutPrice = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1) 
			
			PriceUpOut =  VanillaPutPrice - PriceUpIn
	####################################################
		
		PutPrice = PriceUpOut
	
	##########################################################
	##########################################################		
	
	return(PutPrice)

#######################

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



################ BARRIER OPTIONS ################################################

#### Filter functions dependent of the Min or Max Price and whether we have a Knock In or Knock Out Barrier
#### Some of these are redundant, but defined and used in the functions below, for clarity of the code.

def KnockInBarrierFilterMaxPrice(MaxPrice, KnockInBarrier):
	if MaxPrice >= KnockInBarrier:
		output = 1
	else:
		output = 0
	return(output)
	
def KnockOutBarrierFilterMaxPrice(MaxPrice, KnockOutBarrier):
	if MaxPrice >= KnockOutBarrier:
		output = 0
	else:
		output = 1
	return(output)

def KnockInBarrierFilterMinPrice(MinPrice, KnockInBarrier):
	if MinPrice <= KnockInBarrier:
		output = 1
	else:
		output = 0
	return(output)
	
def KnockOutBarrierFilterMinPrice(MinPrice, KnockOutBarrier):
	if MinPrice <= KnockOutBarrier:
		output = 0
	else:
		output = 1
	return(output)

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
	###
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	

	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_withMinMaxPrice, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'MinPrice', 'MaxPrice', 'MinPrice_smaller_S', 'MaxPrice_smaller_S', 'MinPrice_larger_S', 'MaxPrice_larger_S', 'MinPrice_smaller_sigma', 'MaxPrice_smaller_sigma', 'MinPrice_larger_sigma', 'MaxPrice_larger_sigma', 'MinPrice_smaller_t', 'MaxPrice_smaller_t', 'MinPrice_larger_t', 'MaxPrice_larger_t', 'MinPrice_smaller_r', 'MaxPrice_smaller_r',  'MinPrice_larger_r', 'MaxPrice_larger_r'])

	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockInBarrier:
		full_terminal_price_df['KnockInFilter'] = full_terminal_price_df['MinPrice'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_S'] = full_terminal_price_df['MinPrice_smaller_S'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_S'] = full_terminal_price_df['MinPrice_larger_S'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_sigma'] = full_terminal_price_df['MinPrice_smaller_sigma'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_sigma'] = full_terminal_price_df['MinPrice_larger_sigma'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_t'] = full_terminal_price_df['MinPrice_smaller_t'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_t'] = full_terminal_price_df['MinPrice_larger_t'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_r'] = full_terminal_price_df['MinPrice_smaller_r'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_r'] = full_terminal_price_df['MinPrice_larger_r'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
	else:
		full_terminal_price_df['KnockInFilter'] = full_terminal_price_df['MaxPrice'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_S'] = full_terminal_price_df['MaxPrice_smaller_S'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_S'] = full_terminal_price_df['MaxPrice_larger_S'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_sigma'] = full_terminal_price_df['MaxPrice_smaller_sigma'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_sigma'] = full_terminal_price_df['MaxPrice_larger_sigma'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_t'] = full_terminal_price_df['MaxPrice_smaller_t'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_t'] = full_terminal_price_df['MaxPrice_larger_t'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_r'] = full_terminal_price_df['MaxPrice_smaller_r'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_r'] = full_terminal_price_df['MaxPrice_larger_r'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)

	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = full_terminal_price_df['terminal_price'].to_numpy()
	KnockInFilter_array = full_terminal_price_df['KnockInFilter'].to_numpy()
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(n_simulations)
	
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
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = full_terminal_price_df['terminal_price_smaller_S'].to_numpy()
	KnockInFilter_array_smaller_S = full_terminal_price_df['KnockInFilter_smaller_S'].to_numpy()	
	payoff_array_smaller_S = np.maximum(terminal_price_array_smaller_S-K, 0)
	payoff_array_smaller_S = payoff_array_smaller_S*KnockInFilter_array_smaller_S ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S	
	
	terminal_price_array_larger_S = full_terminal_price_df['terminal_price_larger_S'].to_numpy()
	KnockInFilter_array_larger_S = full_terminal_price_df['KnockInFilter_smaller_S'].to_numpy()
	payoff_array_larger_S = np.maximum(terminal_price_array_larger_S-K, 0)
	payoff_array_larger_S = payoff_array_larger_S*KnockInFilter_array_larger_S ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(0.01**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy()
	KnockInFilter_array_smaller_sigma = full_terminal_price_df['KnockInFilter_smaller_sigma'].to_numpy()
	payoff_array_smaller_sigma = np.maximum(terminal_price_array_smaller_sigma-K, 0)
	payoff_array_smaller_sigma = payoff_array_smaller_sigma*KnockInFilter_array_smaller_sigma ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	
	terminal_price_array_larger_sigma = full_terminal_price_df['terminal_price_larger_sigma'].to_numpy()
	KnockInFilter_array_larger_sigma = full_terminal_price_df['KnockInFilter_larger_sigma'].to_numpy()
	payoff_array_larger_sigma = np.maximum(terminal_price_array_larger_sigma-K, 0)
	payoff_array_larger_sigma = payoff_array_larger_sigma*KnockInFilter_array_larger_sigma ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = full_terminal_price_df['terminal_price_smaller_t'].to_numpy()
	KnockInFilter_array_smaller_t = full_terminal_price_df['KnockInFilter_smaller_t'].to_numpy()
	payoff_array_smaller_t = np.maximum(terminal_price_array_smaller_t-K, 0)
	payoff_array_smaller_t = payoff_array_smaller_t*KnockInFilter_array_smaller_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	
	terminal_price_array_larger_t = full_terminal_price_df['terminal_price_larger_t'].to_numpy()
	KnockInFilter_array_larger_t = full_terminal_price_df['KnockInFilter_larger_t'].to_numpy()
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
	
	terminal_price_array_smaller_r = full_terminal_price_df['terminal_price_smaller_r'].to_numpy()
	KnockInFilter_array_smaller_r = full_terminal_price_df['KnockInFilter_smaller_r'].to_numpy()	
	payoff_array_smaller_r = np.maximum(terminal_price_array_smaller_r-K, 0)	
	payoff_array_smaller_r = payoff_array_smaller_r*KnockInFilter_array_smaller_r 	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	terminal_price_array_larger_r = full_terminal_price_df['terminal_price_larger_r'].to_numpy()
	KnockInFilter_array_larger_r = full_terminal_price_df['KnockInFilter_larger_r'].to_numpy()	
	payoff_array_larger_r = np.maximum(terminal_price_array_larger_r-K, 0)
	payoff_array_larger_r = payoff_array_larger_r*KnockInFilter_array_larger_r	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
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
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	
	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_withMinMaxPrice, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'MinPrice', 'MaxPrice', 'MinPrice_smaller_S', 'MaxPrice_smaller_S', 'MinPrice_larger_S', 'MaxPrice_larger_S', 'MinPrice_smaller_sigma', 'MaxPrice_smaller_sigma', 'MinPrice_larger_sigma', 'MaxPrice_larger_sigma', 'MinPrice_smaller_t', 'MaxPrice_smaller_t', 'MinPrice_larger_t', 'MaxPrice_larger_t', 'MinPrice_smaller_r', 'MaxPrice_smaller_r',  'MinPrice_larger_r', 'MaxPrice_larger_r'])

	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out			
	if S > KnockOutBarrier:
		full_terminal_price_df['KnockOutFilter'] = full_terminal_price_df['MinPrice'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_S'] = full_terminal_price_df['MinPrice_smaller_S'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_S'] = full_terminal_price_df['MinPrice_larger_S'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_sigma'] = full_terminal_price_df['MinPrice_smaller_sigma'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_sigma'] = full_terminal_price_df['MinPrice_larger_sigma'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_t'] = full_terminal_price_df['MinPrice_smaller_t'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_t'] = full_terminal_price_df['MinPrice_larger_t'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_r'] = full_terminal_price_df['MinPrice_smaller_r'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_r'] = full_terminal_price_df['MinPrice_larger_r'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
	else:
		full_terminal_price_df['KnockOutFilter'] = full_terminal_price_df['MaxPrice'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_S'] = full_terminal_price_df['MaxPrice_smaller_S'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_S'] = full_terminal_price_df['MaxPrice_larger_S'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_sigma'] = full_terminal_price_df['MaxPrice_smaller_sigma'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_sigma'] = full_terminal_price_df['MaxPrice_larger_sigma'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_t'] = full_terminal_price_df['MaxPrice_smaller_t'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_t'] = full_terminal_price_df['MaxPrice_larger_t'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_r'] = full_terminal_price_df['MaxPrice_smaller_r'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_r'] = full_terminal_price_df['MaxPrice_larger_r'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = full_terminal_price_df['terminal_price'].to_numpy()
	KnockOutFilter_array = full_terminal_price_df['KnockOutFilter'].to_numpy()
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(n_simulations)
	
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
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = full_terminal_price_df['terminal_price_smaller_S'].to_numpy()
	KnockOutFilter_array_smaller_S = full_terminal_price_df['KnockOutFilter_smaller_S'].to_numpy()	
	payoff_array_smaller_S = np.maximum(terminal_price_array_smaller_S-K, 0)
	payoff_array_smaller_S = payoff_array_smaller_S*KnockOutFilter_array_smaller_S ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S	
	
	terminal_price_array_larger_S = full_terminal_price_df['terminal_price_larger_S'].to_numpy()
	KnockOutFilter_array_larger_S = full_terminal_price_df['KnockOutFilter_smaller_S'].to_numpy()
	payoff_array_larger_S = np.maximum(terminal_price_array_larger_S-K, 0)
	payoff_array_larger_S = payoff_array_larger_S*KnockOutFilter_array_larger_S ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(0.01**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy()
	KnockOutFilter_array_smaller_sigma = full_terminal_price_df['KnockOutFilter_smaller_sigma'].to_numpy()
	payoff_array_smaller_sigma = np.maximum(terminal_price_array_smaller_sigma-K, 0)
	payoff_array_smaller_sigma = payoff_array_smaller_sigma*KnockOutFilter_array_smaller_sigma ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	
	terminal_price_array_larger_sigma = full_terminal_price_df['terminal_price_larger_sigma'].to_numpy()
	KnockOutFilter_array_larger_sigma = full_terminal_price_df['KnockOutFilter_larger_sigma'].to_numpy()
	payoff_array_larger_sigma = np.maximum(terminal_price_array_larger_sigma-K, 0)
	payoff_array_larger_sigma = payoff_array_larger_sigma*KnockOutFilter_array_larger_sigma ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = full_terminal_price_df['terminal_price_smaller_t'].to_numpy()
	KnockOutFilter_array_smaller_t = full_terminal_price_df['KnockOutFilter_smaller_t'].to_numpy()
	payoff_array_smaller_t = np.maximum(terminal_price_array_smaller_t-K, 0)
	payoff_array_smaller_t = payoff_array_smaller_t*KnockOutFilter_array_smaller_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	
	terminal_price_array_larger_t = full_terminal_price_df['terminal_price_larger_t'].to_numpy()
	KnockOutFilter_array_larger_t = full_terminal_price_df['KnockOutFilter_larger_t'].to_numpy()
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
	
	terminal_price_array_smaller_r = full_terminal_price_df['terminal_price_smaller_r'].to_numpy()
	KnockOutFilter_array_smaller_r = full_terminal_price_df['KnockOutFilter_smaller_r'].to_numpy()	
	payoff_array_smaller_r = np.maximum(terminal_price_array_smaller_r-K, 0)	
	payoff_array_smaller_r = payoff_array_smaller_r*KnockOutFilter_array_smaller_r 	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	terminal_price_array_larger_r = full_terminal_price_df['terminal_price_larger_r'].to_numpy()
	KnockOutFilter_array_larger_r = full_terminal_price_df['KnockOutFilter_larger_r'].to_numpy()	
	payoff_array_larger_r = np.maximum(terminal_price_array_larger_r-K, 0)
	payoff_array_larger_r = payoff_array_larger_r*KnockOutFilter_array_larger_r	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
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
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	
	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_withMinMaxPrice, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'MinPrice', 'MaxPrice', 'MinPrice_smaller_S', 'MaxPrice_smaller_S', 'MinPrice_larger_S', 'MaxPrice_larger_S', 'MinPrice_smaller_sigma', 'MaxPrice_smaller_sigma', 'MinPrice_larger_sigma', 'MaxPrice_larger_sigma', 'MinPrice_smaller_t', 'MaxPrice_smaller_t', 'MinPrice_larger_t', 'MaxPrice_larger_t', 'MinPrice_smaller_r', 'MaxPrice_smaller_r',  'MinPrice_larger_r', 'MaxPrice_larger_r'])

	# Check is starting price is above or below the knock-in barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-in		
	if S > KnockInBarrier:
		full_terminal_price_df['KnockInFilter'] = full_terminal_price_df['MinPrice'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_S'] = full_terminal_price_df['MinPrice_smaller_S'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_S'] = full_terminal_price_df['MinPrice_larger_S'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_sigma'] = full_terminal_price_df['MinPrice_smaller_sigma'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_sigma'] = full_terminal_price_df['MinPrice_larger_sigma'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_t'] = full_terminal_price_df['MinPrice_smaller_t'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_t'] = full_terminal_price_df['MinPrice_larger_t'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_r'] = full_terminal_price_df['MinPrice_smaller_r'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_r'] = full_terminal_price_df['MinPrice_larger_r'].apply(KnockInBarrierFilterMinPrice, KnockInBarrier=KnockInBarrier)
	else:
		full_terminal_price_df['KnockInFilter'] = full_terminal_price_df['MaxPrice'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_S'] = full_terminal_price_df['MaxPrice_smaller_S'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_S'] = full_terminal_price_df['MaxPrice_larger_S'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_sigma'] = full_terminal_price_df['MaxPrice_smaller_sigma'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_sigma'] = full_terminal_price_df['MaxPrice_larger_sigma'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_t'] = full_terminal_price_df['MaxPrice_smaller_t'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_t'] = full_terminal_price_df['MaxPrice_larger_t'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_smaller_r'] = full_terminal_price_df['MaxPrice_smaller_r'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)
		full_terminal_price_df['KnockInFilter_larger_r'] = full_terminal_price_df['MaxPrice_larger_r'].apply(KnockInBarrierFilterMaxPrice, KnockInBarrier=KnockInBarrier)

	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = full_terminal_price_df['terminal_price'].to_numpy()
	KnockInFilter_array = full_terminal_price_df['KnockInFilter'].to_numpy()
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(n_simulations)
	
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
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = full_terminal_price_df['terminal_price_smaller_S'].to_numpy()
	KnockInFilter_array_smaller_S = full_terminal_price_df['KnockInFilter_smaller_S'].to_numpy()	
	payoff_array_smaller_S = np.maximum(K-terminal_price_array_smaller_S, 0)
	payoff_array_smaller_S = payoff_array_smaller_S*KnockInFilter_array_smaller_S ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S	
	
	terminal_price_array_larger_S = full_terminal_price_df['terminal_price_larger_S'].to_numpy()
	KnockInFilter_array_larger_S = full_terminal_price_df['KnockInFilter_smaller_S'].to_numpy()
	payoff_array_larger_S = np.maximum(K-terminal_price_array_larger_S, 0)
	payoff_array_larger_S = payoff_array_larger_S*KnockInFilter_array_larger_S ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(0.01**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy()
	KnockInFilter_array_smaller_sigma = full_terminal_price_df['KnockInFilter_smaller_sigma'].to_numpy()
	payoff_array_smaller_sigma = np.maximum(K-terminal_price_array_smaller_sigma, 0)
	payoff_array_smaller_sigma = payoff_array_smaller_sigma*KnockInFilter_array_smaller_sigma ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	
	terminal_price_array_larger_sigma = full_terminal_price_df['terminal_price_larger_sigma'].to_numpy()
	KnockInFilter_array_larger_sigma = full_terminal_price_df['KnockInFilter_larger_sigma'].to_numpy()
	payoff_array_larger_sigma = np.maximum(K-terminal_price_array_larger_sigma, 0)
	payoff_array_larger_sigma = payoff_array_larger_sigma*KnockInFilter_array_larger_sigma ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = full_terminal_price_df['terminal_price_smaller_t'].to_numpy()
	KnockInFilter_array_smaller_t = full_terminal_price_df['KnockInFilter_smaller_t'].to_numpy()
	payoff_array_smaller_t = np.maximum(K-terminal_price_array_smaller_t, 0)
	payoff_array_smaller_t = payoff_array_smaller_t*KnockInFilter_array_smaller_t ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	
	terminal_price_array_larger_t = full_terminal_price_df['terminal_price_larger_t'].to_numpy()
	KnockInFilter_array_larger_t = full_terminal_price_df['KnockInFilter_larger_t'].to_numpy()
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
	
	terminal_price_array_smaller_r = full_terminal_price_df['terminal_price_smaller_r'].to_numpy()
	KnockInFilter_array_smaller_r = full_terminal_price_df['KnockInFilter_smaller_r'].to_numpy()	
	payoff_array_smaller_r = np.maximum(K-terminal_price_array_smaller_r, 0)	
	payoff_array_smaller_r = payoff_array_smaller_r*KnockInFilter_array_smaller_r 	 ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	terminal_price_array_larger_r = full_terminal_price_df['terminal_price_larger_r'].to_numpy()
	KnockInFilter_array_larger_r = full_terminal_price_df['KnockInFilter_larger_r'].to_numpy()	
	payoff_array_larger_r = np.maximum(K-terminal_price_array_larger_r, 0)
	payoff_array_larger_r = payoff_array_larger_r*KnockInFilter_array_larger_r	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
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
	
	#### calculate the terminal prices and store them as a list ####
	#### use multiple processors to speed up the simulations #######
	seed_number_array = np.arange(n_simulations) + BaseSeed

	args = [(S, r, sigma, t, T, n_steps, seed) for seed in seed_number_array]
	
	with mp.Pool() as pool:
		full_terminal_price_array = pool.starmap(generate_terminal_price_ForGreeks_withMinMaxPrice, args)

	#### 
	full_terminal_price_array = np.array(full_terminal_price_array)
	
	full_terminal_price_df = pd.DataFrame(full_terminal_price_array, columns=['terminal_price', 'terminal_price_smaller_S', 'terminal_price_larger_S', 'terminal_price_smaller_sigma', 'terminal_price_larger_sigma', 'terminal_price_smaller_t', 'terminal_price_larger_t', 'terminal_price_smaller_r', 'terminal_price_larger_r', 'MinPrice', 'MaxPrice', 'MinPrice_smaller_S', 'MaxPrice_smaller_S', 'MinPrice_larger_S', 'MaxPrice_larger_S', 'MinPrice_smaller_sigma', 'MaxPrice_smaller_sigma', 'MinPrice_larger_sigma', 'MaxPrice_larger_sigma', 'MinPrice_smaller_t', 'MaxPrice_smaller_t', 'MinPrice_larger_t', 'MaxPrice_larger_t', 'MinPrice_smaller_r', 'MaxPrice_smaller_r',  'MinPrice_larger_r', 'MaxPrice_larger_r'])

	# Check is starting price is above or below the knock-out barrier. This tells us whether we are to use the MinPrices or MaxPrices over the stock prices to determine whether the option gets knocked-out	
	if S > KnockOutBarrier:
		full_terminal_price_df['KnockOutFilter'] = full_terminal_price_df['MinPrice'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_S'] = full_terminal_price_df['MinPrice_smaller_S'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_S'] = full_terminal_price_df['MinPrice_larger_S'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_sigma'] = full_terminal_price_df['MinPrice_smaller_sigma'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_sigma'] = full_terminal_price_df['MinPrice_larger_sigma'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_t'] = full_terminal_price_df['MinPrice_smaller_t'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_t'] = full_terminal_price_df['MinPrice_larger_t'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_r'] = full_terminal_price_df['MinPrice_smaller_r'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_r'] = full_terminal_price_df['MinPrice_larger_r'].apply(KnockOutBarrierFilterMinPrice, KnockOutBarrier=KnockOutBarrier)
	else:
		full_terminal_price_df['KnockOutFilter'] = full_terminal_price_df['MaxPrice'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_S'] = full_terminal_price_df['MaxPrice_smaller_S'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_S'] = full_terminal_price_df['MaxPrice_larger_S'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_sigma'] = full_terminal_price_df['MaxPrice_smaller_sigma'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_sigma'] = full_terminal_price_df['MaxPrice_larger_sigma'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_t'] = full_terminal_price_df['MaxPrice_smaller_t'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_t'] = full_terminal_price_df['MaxPrice_larger_t'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_smaller_r'] = full_terminal_price_df['MaxPrice_smaller_r'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
		full_terminal_price_df['KnockOutFilter_larger_r'] = full_terminal_price_df['MaxPrice_larger_r'].apply(KnockOutBarrierFilterMaxPrice, KnockOutBarrier=KnockOutBarrier)
	
	###### CALCULATE THE OPTION PRICE ###################
	### extract "central" terminal price array #######
	terminal_price_array = full_terminal_price_df['terminal_price'].to_numpy()
	KnockOutFilter_array = full_terminal_price_df['KnockOutFilter'].to_numpy()
	
	### initialize option payoff array to zero values ####	
	payoff_array = np.zeros(n_simulations)
	
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
	#### Calculate Delta = dV/dS (partial) #########
	terminal_price_array_smaller_S = full_terminal_price_df['terminal_price_smaller_S'].to_numpy()
	KnockOutFilter_array_smaller_S = full_terminal_price_df['KnockOutFilter_smaller_S'].to_numpy()	
	payoff_array_smaller_S = np.maximum(K-terminal_price_array_smaller_S, 0)
	payoff_array_smaller_S = payoff_array_smaller_S*KnockOutFilter_array_smaller_S ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_S = np.exp(-r*(T-t))*payoff_array_smaller_S	
	
	terminal_price_array_larger_S = full_terminal_price_df['terminal_price_larger_S'].to_numpy()
	KnockOutFilter_array_larger_S = full_terminal_price_df['KnockOutFilter_smaller_S'].to_numpy()
	payoff_array_larger_S = np.maximum(K-terminal_price_array_larger_S, 0)
	payoff_array_larger_S = payoff_array_larger_S*KnockOutFilter_array_larger_S ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_S = np.exp(-r*(T-t))*payoff_array_larger_S
	
	delta_array = (option_value_array_larger_S-option_value_array_smaller_S)/(2*0.01)
	delta_value = np.mean(delta_array)
	delta_StandardError = stats.sem(delta_array)

	#### Calculate Gamma = d^2V/dS^2 (partial) #####	
	
	gamma_array = (option_value_array_larger_S-2*option_value_array+option_value_array_smaller_S)/(0.01**2)
	gamma_value = np.mean(gamma_array)
	gamma_StandardError = stats.sem(gamma_array)

	#### Calculate Vega = dV/dsigma (partial) ######
	
	terminal_price_array_smaller_sigma = full_terminal_price_df['terminal_price_smaller_sigma'].to_numpy()
	KnockOutFilter_array_smaller_sigma = full_terminal_price_df['KnockOutFilter_smaller_sigma'].to_numpy()
	payoff_array_smaller_sigma = np.maximum(K-terminal_price_array_smaller_sigma, 0)
	payoff_array_smaller_sigma = payoff_array_smaller_sigma*KnockOutFilter_array_smaller_sigma	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_smaller_sigma = np.exp(-r*(T-t))*payoff_array_smaller_sigma
	
	terminal_price_array_larger_sigma = full_terminal_price_df['terminal_price_larger_sigma'].to_numpy()
	KnockOutFilter_array_larger_sigma = full_terminal_price_df['KnockOutFilter_larger_sigma'].to_numpy()
	payoff_array_larger_sigma = np.maximum(K-terminal_price_array_larger_sigma, 0)
	payoff_array_larger_sigma = payoff_array_larger_sigma*KnockOutFilter_array_larger_sigma	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	option_value_array_larger_sigma = np.exp(-r*(T-t))*payoff_array_larger_sigma
	
	vega_array = (option_value_array_larger_sigma-option_value_array_smaller_sigma)/(2*0.01)
	vega_value = np.mean(vega_array)
	vega_StandardError = stats.sem(vega_array)
	
	#### Calculate Theta = -dV/dt (partial)  #######
	
	terminal_price_array_smaller_t = full_terminal_price_df['terminal_price_smaller_t'].to_numpy()
	KnockOutFilter_array_smaller_t = full_terminal_price_df['KnockOutFilter_smaller_t'].to_numpy()
	payoff_array_smaller_t = np.maximum(K-terminal_price_array_smaller_t, 0)
	payoff_array_smaller_t = payoff_array_smaller_t*KnockOutFilter_array_smaller_t	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY

	
	terminal_price_array_larger_t = full_terminal_price_df['terminal_price_larger_t'].to_numpy()
	KnockOutFilter_array_larger_t = full_terminal_price_df['KnockOutFilter_larger_t'].to_numpy()
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
	
	terminal_price_array_smaller_r = full_terminal_price_df['terminal_price_smaller_r'].to_numpy()
	KnockOutFilter_array_smaller_r = full_terminal_price_df['KnockOutFilter_smaller_r'].to_numpy()	
	payoff_array_smaller_r = np.maximum(K-terminal_price_array_smaller_r, 0)	
	payoff_array_smaller_r = payoff_array_smaller_r*KnockOutFilter_array_smaller_r 	### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
	terminal_price_array_larger_r = full_terminal_price_df['terminal_price_larger_r'].to_numpy()
	KnockOutFilter_array_larger_r = full_terminal_price_df['KnockOutFilter_larger_r'].to_numpy()	
	payoff_array_larger_r = np.maximum(K-terminal_price_array_larger_r, 0)
	payoff_array_larger_r = payoff_array_larger_r*KnockOutFilter_array_larger_r	 ### APPLIES THE BARRIER FILTER TO THE PAYOFF ARRAY
	
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
	print('\n')
	
	print('Call Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')
	
	print('\nEuropean Call Analytic with Knock In at S=0.01\n', AnalyticBlackScholesKnockInCall(80, 85, 0.05, 0.4, 1, 1.25, 0.01))
	print('\nEuropean Call Monte-Carlo with Knock In at S=0\n', MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 0))
	
	print('\nEuropean Call Analytic with Knock In at S=85\n', AnalyticBlackScholesKnockInCall(80, 85, 0.05, 0.4, 1, 1.25, 85))
	print('\nEuropean Call Monte-Carlo with Knock In at S=85\n', MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))

	print('\nEuropean Call Analytic with Knock Out at S=85\n', AnalyticBlackScholesKnockOutCall(80, 85, 0.05, 0.4, 1, 1.25, 85))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=85\n', MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	
	print('\nEuropean Call Analytic with Knock Out at S=1000\n', AnalyticBlackScholesKnockOutCall(80, 85, 0.05, 0.4, 1, 1.25, 1000))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=1000\n', MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 1000))

	print('\nEuropean Call Analytic with Knock In at S=75\n', AnalyticBlackScholesKnockInCall(80, 85, 0.05, 0.4, 1, 1.25, 75))
	print('\nEuropean Call Monte-Carlo with Knock In at S=75\n', MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))

	print('\nEuropean Call Analytic with Knock Out at S=75\n', AnalyticBlackScholesKnockOutCall(80, 85, 0.05, 0.4, 1, 1.25, 75))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=75\n', MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))

	print('\nEuropean Call Analytic with Knock In at S=110\n', AnalyticBlackScholesKnockInCall(80, 85, 0.05, 0.4, 1, 1.25, 110))
	print('\nEuropean Call Monte-Carlo with Knock In at S=110\n', MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))

	print('\nEuropean Call Analytic with Knock Out at S=110\n', AnalyticBlackScholesKnockOutCall(80, 85, 0.05, 0.4, 1, 1.25, 110))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=110\n', MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))

	print('\n')
	
	print('Put Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')

	print('\nEuropean Put Analytic with Knock In at S=0.01\n', AnalyticBlackScholesKnockInPut(80, 85, 0.05, 0.4, 1, 1.25, 0.01))	
	print('\nEuropean Put Monte-Carlo with Knock In at S=0\n', MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 0))
	
	
	print('\nEuropean Put Analytic with Knock In at S=85\n', AnalyticBlackScholesKnockInPut(80, 85, 0.05, 0.4, 1, 1.25, 85))		
	print('\nEuropean Put Monte-Carlo with Knock In at S=85\n', MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	

	print('\nEuropean Put Analytic with Knock Out at S=85\n', AnalyticBlackScholesKnockOutPut(80, 85, 0.05, 0.4, 1, 1.25, 85))		
	print('\nEuropean Put Monte-Carlo with Knock Out at S=85\n', MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	
	print('\nEuropean Put Analytic with Knock Out at S=1000\n', AnalyticBlackScholesKnockOutPut(80, 85, 0.05, 0.4, 1, 1.25, 1000))		
	print('\nEuropean Put Monte-Carlo with Knock Out at S=1000\n', MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 1000))
	
	print('\nEuropean Put Analytic with Knock In at S=75\n', AnalyticBlackScholesKnockInPut(80, 85, 0.05, 0.4, 1, 1.25, 75))			
	print('\nEuropean Put Monte-Carlo with Knock In at S=75\n', MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	
	print('\nEuropean Put Analytic with Knock Out at S=75\n', AnalyticBlackScholesKnockOutPut(80, 85, 0.05, 0.4, 1, 1.25, 75))			
	print('\nEuropean Put Monte-Carlo with Knock Out at S=75\n', MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	
	print('\nEuropean Put Analytic with Knock In at S=110\n', AnalyticBlackScholesKnockInPut(80, 85, 0.05, 0.4, 1, 1.25, 110))				
	print('\nEuropean Put Monte-Carlo with Knock In at S=110\n', MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))
	
	print('\nEuropean Put Analytic with Knock Out at S=110\n', AnalyticBlackScholesKnockOutPut(80, 85, 0.05, 0.4, 1, 1.25, 110))				
	print('\nEuropean Put Monte-Carlo with Knock Out at S=110\n', MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))


if __name__ == "__main__":
	main()	


	
