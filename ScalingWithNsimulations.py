import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from scipy.stats import norm
from multiprocessing import Pool

import EuropeanVanilla as ev

### Compares the Monte-Carlo of the Vanilla European with the Analytic Black-Scholes Formula ####
### Generates a plot, showing the convergence and error estimate ################################
### To do: perform similar exercise with the Greeks #############################################

##################################################################################
##################################################################################

def main():
	print("Number of cpus : ", mp.cpu_count())
	pool = Pool(processes=(mp.cpu_count() - 1))
	StandardBaseSeed = 0
	
	
	print('\n')	
	print ("Starting simulations...\n")

	NSim_array = np.array([10, 20, 40, 70, 100, 200, 400, 700, 1e3, 2e3, 4e3, 7e3, 1e4, 2e4, 4e4, 7e4, 1e5, 2e5, 4e5, 7e5, 1e6]) ##### array of n_simulations values to scan over
	NSim_array = np.array([10, 20, 40, 70, 100, 200, 400, 700, 1e3, 2e3, 4e3, 7e3, 1e4, 2e4, 4e4, 7e4, 1e5, 2e5, 4e5, 7e5, 1e6, 2e6, 4e6, 7e6, 1e7]) ##### array of n_simulations values to scan over
	n_examples = len(NSim_array)
	Output_array = np.zeros((n_examples, 12)) # Output array: BlackScholesVanillaEuropeanCallWithGreeks function has an output of length 12

	### Choose some example values for our Option
	Stockprice = 80
	Strikeprice = 85
	interest = 0.05
	volatility = 0.4
	timenow = 0
	timeatmaturity = 0.25 

	### Generate the Monte-Carlo prices and Greeks. Note we can increase n_steps to get a better theta estimate (current implementation using plus/minus one step to calculate derivative).

	for i in range(0,n_examples):
		print(f'Starting example with {NSim_array[i]} simulations\n')
		Output_array[i, :] = ev.MonteCarloVanillaEuropeanCallWithGreeks(Stockprice, Strikeprice, interest, volatility, timenow, timeatmaturity, n_steps=10, n_simulations=NSim_array[i])

	print('Prices and Greeks using Monte-Carlo for the different n_simulations are:\n', Output_array)

	Analytic_array = np.array(ev.BlackScholesVanillaEuropeanCallWithGreeks(Stockprice, Strikeprice, interest, volatility, timenow, timeatmaturity))
	print('\nPrices and Greeks using the analytic formula are:\n', Analytic_array)


	AnalyticPrice = Analytic_array[0]
	MonteCarloPrice_array = Output_array[:, 0]
	MonteCarloPrice_StdErr_array = Output_array[:, 6]
	
	#### Range of CI to use. Use multiplification factor of 1 for 68%, 1.645 for 90%, 1.96 for 95%, or 2.58 for 99%. Update plot label if change is made.
	MonteCarloPrice_array_upper = MonteCarloPrice_array+1.645*MonteCarloPrice_StdErr_array
	MonteCarloPrice_array_lower = MonteCarloPrice_array-1.645*MonteCarloPrice_StdErr_array

	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloPrice_array, '-o', label='Monte Carlo Price', c='C0')
	plt.fill_between(NSim_array, MonteCarloPrice_array_lower, MonteCarloPrice_array_upper, alpha=0.3, label='(90% CI)')
	plt.axhline(y=AnalyticPrice, color='r', linestyle='--', label='Analytic Price')
	plt.xscale('log') 
	plt.title(f'Price of Call Option (S={Stockprice}, K={Strikeprice}, r={interest}, sigma={volatility}, t={timenow}, T={timeatmaturity})')
	plt.xlabel('Number of Simulations')
	plt.ylabel('Price ($)')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.xlim(1e1, 1e7)
	plt.ylim(1, 7)
	plt.savefig("./plots/MonteCarloPriceConvergence.jpg")
	plt.clf()

	print('Generated MonteCarloPriceConvergence.jpg and saved in plots folder.')


if __name__ == "__main__":
	main()



