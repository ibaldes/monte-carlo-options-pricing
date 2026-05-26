import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

###

'''
CREATES A VISUALISATION OF THE MONTE-CARLO SHARE PRICE WITH COMPARISON OF OBJECTIVE AND RISK-NEUTRAL MEASURE
UNDER THE OBJECTIVE MEASURE THE STOCK PRICE DRIFT IS mu
UNDER THE RISK NEUTRAL MEASURE, THE DISCOUNTED STOCK PRICE IS A MARTINGALE, SO THE STOCK PRICE DRIFT IS r 
'''

##################################################################################
##################################################################################


def main():

	StandardBaseSeed = 100
	rng = np.random.default_rng(StandardBaseSeed)
	
	S = 100		# Initial stock price
	mu = 0.50	# Objective stock price drift
	sigma = 0.25	# Yealy volatility
	r = 0.05	# Yearly risk-free rate
	t = 0		# Initial time
	T = 1		# Final time
	
	n_steps = 252
	n_simulations = 6
	time_step = (T-t)/n_steps
	
	

	#### initialize the stock price array, the first entry is the price at t, the rest are initialized to zero. There are 2*n_simulation rows as we will use antithetic sampling ####
	stock_price_array_rnp = np.zeros((2*n_simulations,n_steps+1))  #### rnp = risk neutral pricing
	stock_price_array_obj = np.zeros((2*n_simulations,n_steps+1))  #### obj = risk neutral pricing	
	
	stock_price_array_rnp[:,0] = S
	stock_price_array_obj[:,0] = S

	#### loop over the n_simulations, using antithetic sampling, generating brownian motions for the stock price motions
	for i in range(0,n_simulations):
	
		### generate an array of brownian motions, one for each time step, given the time step and volatility	
		brownian_array = np.sqrt(time_step)*sigma*rng.normal(0, 1, size=n_steps)
	
		#### fill in the stock price array for simulation i, given the stock price on the previous day, the brownian motion and the SDE #####
		#### also fill in the stock price array for the antithetic simulation i (bottom half of the array), given the stock price on the previous day, and the brownian motion #####		
		
		for j in range(1,n_steps+1):
			
			stock_price_array_rnp[i,j] = stock_price_array_rnp[i,j-1]*(1+brownian_array[j-1]+r*time_step)  							### risk-neutral
			stock_price_array_rnp[i+n_simulations,j] = stock_price_array_rnp[i+n_simulations,j-1]*(1-brownian_array[j-1]+r*time_step) 			### risk-neutral antithetic
			
			stock_price_array_obj[i,j] = stock_price_array_obj[i,j-1]*(1+brownian_array[j-1]+(mu+0.5*sigma**2)*time_step)  					### objective
			stock_price_array_obj[i+n_simulations,j] = stock_price_array_obj[i+n_simulations,j-1]*(1-brownian_array[j-1]+(mu+0.5*sigma**2)*time_step) 	### objective antithetic

	
	time_array = np.linspace(start=t, stop=T, num=n_steps+1)
	

	##### make plot #########

	plt.plot(time_array, stock_price_array_obj[0, :], '-', c='C0', label='Objective Measure')
	plt.plot(time_array, stock_price_array_rnp[0, :], '--', c='C1', label='Risk-neutral Measure')		
	for i in range(1,2*n_simulations):
		plt.plot(time_array, stock_price_array_obj[i, :], '-', c='C0')
		plt.plot(time_array, stock_price_array_rnp[i, :], '--', c='C1')
	plt.title(rf'Stock Price Path ($\mu$={mu}, r={r}, $\sigma$={sigma}, $\mathrm{{ n_{{ steps}} }}$={n_steps})')
	plt.ylabel('Price S(t) [$]')
	plt.xlabel('t [years]')	
	plt.legend(loc='upper left')
	plt.grid(True)
	plt.xlim(0, 1)
	plt.savefig("Path_Visualisation.jpg")
	plt.savefig("../plots/PathVisualisation/Path_Visualisation.jpg")
	plt.clf()
	
	####################################################################
	##### Redo for comparison, using S(t) directly, rather than SDE ####
	####################################################################
	
	StandardBaseSeed = 100
	rng = np.random.default_rng(StandardBaseSeed)
	
	#### initialize the stock price array, the first entry is the price at t, the rest are initialized to zero. There are 2*n_simulation rows as we will use antithetic sampling ####
	stock_price_array_rnp = np.zeros((2*n_simulations,n_steps+1))  #### rnp = risk neutral pricing
	stock_price_array_obj = np.zeros((2*n_simulations,n_steps+1))  #### obj = risk neutral pricing	
	
	stock_price_array_rnp[:,0] = S
	stock_price_array_obj[:,0] = S

	#### loop over the n_simulations, using antithetic sampling, generating brownian motions for the stock price motions
	for i in range(0,n_simulations):
	
		### generate an array of brownian motions, one for each time step, given the time step and volatility	
		brownian_array = np.sqrt(time_step)*sigma*rng.normal(0, 1, size=n_steps)
	
		#### fill in the stock price array for simulation i, given the stock price on the previous day, the brownian motion and the SDE #####
		#### also fill in the stock price array for the antithetic simulation i (bottom half of the array), given the stock price on the previous day, and the brownian motion #####		
		
		for j in range(1,n_steps+1):
			
			stock_price_array_rnp[i,j] = stock_price_array_rnp[i,j-1]*np.exp(brownian_array[j-1]+(r-0.5*sigma**2)*time_step)  				### risk-neutral
			stock_price_array_rnp[i+n_simulations,j] = stock_price_array_rnp[i+n_simulations,j-1]*np.exp(-brownian_array[j-1]+(r-0.5*sigma**2)*time_step) 	### risk-neutral antithetic
			
			stock_price_array_obj[i,j] = stock_price_array_obj[i,j-1]*np.exp(brownian_array[j-1]+mu*time_step)  						### objective
			stock_price_array_obj[i+n_simulations,j] = stock_price_array_obj[i+n_simulations,j-1]*np.exp(-brownian_array[j-1]+mu*time_step) 		### objective antithetic

	
	time_array = np.linspace(start=t, stop=T, num=n_steps+1)
	
	##### make plot 2 #########

	plt.plot(time_array, stock_price_array_obj[0, :], '-', c='C0', label='Objective Measure')
	plt.plot(time_array, stock_price_array_rnp[0, :], '--', c='C1', label='Risk-neutral Measure')		
	for i in range(1,2*n_simulations):
		plt.plot(time_array, stock_price_array_obj[i, :], '-', c='C0')
		plt.plot(time_array, stock_price_array_rnp[i, :], '--', c='C1')
	plt.title(rf'Stock Price Path ($\mu$={mu}, r={r}, $\sigma$={sigma}, $\mathrm{{ n_{{ steps}} }}$={n_steps})')
	plt.ylabel('Price S(t) [$]')
	plt.xlabel('t [years]')	
	plt.legend(loc='upper left')
	plt.grid(True)
	plt.xlim(0, 1)
	plt.savefig("Path_Visualisation_2.jpg")	
	plt.savefig("../plots/PathVisualisation/Path_Visualisation_2.jpg")
	plt.clf()		



if __name__ == "__main__":
	main()	


	
