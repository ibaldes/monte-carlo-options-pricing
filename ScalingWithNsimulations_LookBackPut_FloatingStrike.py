import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from scipy.stats import norm
from multiprocessing import Pool

import LookBack as lb

### Compares the Monte-Carlo of the Look Back Floating Strike Put with the Analytic  Formula ####
### Generates a plot, showing the convergence and error estimate ################################

##################################################################################
##################################################################################

def main():
	print("Number of cpus : ", mp.cpu_count())
	pool = Pool(processes=(mp.cpu_count() - 1))
	StandardBaseSeed = 0
	
	
	print('\n')	
	print ("Starting simulations...\n")

	NSim_array = np.array([10, 20, 40, 70, 100, 200, 400, 700, 1e3, 2e3, 4e3, 7e3, 1e4, 2e4, 4e4, 7e4, 1e5, 2e5, 4e5, 7e5, 1e6]) ##### array of n_simulations values to scan over
	n_examples = len(NSim_array)
	Output_array = np.zeros((n_examples, 12)) # Output array: MonteCarloFloatingStrikeLookBackPutWithGreeks function has an output of length 12

	### Choose some example values for our Option
	Stockprice = 80
	interest = 0.05
	volatility = 0.4
	timenow = 0
	timeatmaturity = 0.25 

	### Generate the Monte-Carlo prices and Greeks. Note we can increase n_steps to get a better theta estimate (current implementation using plus/minus one step to calculate derivative).

	for i in range(0,n_examples):
		print(f'Starting example with {NSim_array[i]} simulations\n')
		Output_array[i, :] = lb.MonteCarloFloatingStrikeLookBackPutWithGreeks(Stockprice, interest, volatility, timenow, timeatmaturity, Smaxtodate=None, n_steps=100, n_simulations=NSim_array[i])

	print('Prices and Greeks using Monte-Carlo for the different n_simulations are:\n', Output_array)

	Analytic_array = np.array(lb.AnalyticFloatingStrikeLookBackPutWithGreeks(Stockprice, interest, volatility, timenow, timeatmaturity, Smaxtodate=None))
	print('\nPrices and Greeks using the analytic formula are:\n', Analytic_array)


	AnalyticPrice = Analytic_array[0]
	MonteCarloPrice_array = Output_array[:, 0]
	MonteCarloPrice_StdErr_array = Output_array[:, 6]
	
	AnalyticDelta = Analytic_array[1]
	MonteCarloDelta_array = Output_array[:, 1]
	MonteCarloDelta_StdErr_array = Output_array[:, 7]
	
	AnalyticGamma = Analytic_array[2]
	MonteCarloGamma_array = Output_array[:, 2]
	MonteCarloGamma_StdErr_array = Output_array[:, 8]	

	AnalyticVega = Analytic_array[3]
	MonteCarloVega_array = Output_array[:, 3]
	MonteCarloVega_StdErr_array = Output_array[:, 9]

	AnalyticTheta = Analytic_array[4]
	MonteCarloTheta_array = Output_array[:, 4]
	MonteCarloTheta_StdErr_array = Output_array[:, 10]
	
	AnalyticRho = Analytic_array[5]
	MonteCarloRho_array = Output_array[:, 5]
	MonteCarloRho_StdErr_array = Output_array[:, 11]
	
	#### Range of CI to use. Use multiplification factor of 1 for 68%, 1.645 for 90%, 1.96 for 95%, or 2.58 for 99%. Update plot label if change is made.
	multiplcation_factor = 1.645

	if multiplcation_factor == 1:
		CIpercentage = 68
	elif multiplcation_factor == 1.645:
		CIpercentage = 90
	elif multiplcation_factor == 1.96:
		CIpercentage = 95
	elif multiplcation_factor == 2.58:
		CIpercentage = 99
	else:
		CIpercentage = 0

	MonteCarloPrice_array_upper = MonteCarloPrice_array+multiplcation_factor*MonteCarloPrice_StdErr_array
	MonteCarloPrice_array_lower = MonteCarloPrice_array-multiplcation_factor*MonteCarloPrice_StdErr_array

	MonteCarloDelta_array_upper = MonteCarloDelta_array+multiplcation_factor*MonteCarloDelta_StdErr_array
	MonteCarloDelta_array_lower = MonteCarloDelta_array-multiplcation_factor*MonteCarloDelta_StdErr_array

	MonteCarloGamma_array_upper = MonteCarloGamma_array+multiplcation_factor*MonteCarloGamma_StdErr_array
	MonteCarloGamma_array_lower = MonteCarloGamma_array-multiplcation_factor*MonteCarloGamma_StdErr_array

	MonteCarloVega_array_upper = MonteCarloVega_array+multiplcation_factor*MonteCarloVega_StdErr_array
	MonteCarloVega_array_lower = MonteCarloVega_array-multiplcation_factor*MonteCarloVega_StdErr_array	

	MonteCarloTheta_array_upper = MonteCarloTheta_array+multiplcation_factor*MonteCarloTheta_StdErr_array
	MonteCarloTheta_array_lower = MonteCarloTheta_array-multiplcation_factor*MonteCarloTheta_StdErr_array

	MonteCarloRho_array_upper = MonteCarloRho_array+multiplcation_factor*MonteCarloRho_StdErr_array
	MonteCarloRho_array_lower = MonteCarloRho_array-multiplcation_factor*MonteCarloRho_StdErr_array
	
	
	
	##### make plots #########
		
	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloPrice_array, '-o', label='Monte Carlo Price', c='C0')
	plt.fill_between(NSim_array, MonteCarloPrice_array_lower, MonteCarloPrice_array_upper, alpha=0.3, label=f'Naive {CIpercentage}% CI')
	plt.axhline(y=AnalyticPrice, color='r', linestyle='--', label='Analytic Price')
	plt.xscale('log') 
	plt.title(f'Price of Floating Strike Look Back Put Option (S={Stockprice}, r={interest}, sigma={volatility}, t={timenow}, T={timeatmaturity})')
	plt.xlabel('Number of Simulations')
	plt.ylabel('Price ($)')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.xlim(1e1, 1e7)
	plt.savefig("./plots/LookBackPutFloatingStrike/MonteCarloPriceConvergence_LookBackPut_FloatingStrike.jpg")
	plt.clf()

	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloDelta_array, '-o', label=r'Monte Carlo $\Delta$', c='C0')
	plt.fill_between(NSim_array, MonteCarloDelta_array_lower, MonteCarloDelta_array_upper, alpha=0.3, label=f'Naive {CIpercentage}% CI')
	plt.axhline(y=AnalyticDelta, color='r', linestyle='--', label=r'Analytic $\Delta$')
	plt.xscale('log') 
	plt.title(f'Delta of Floating Strike Look Back Put Option (S={Stockprice}, r={interest}, sigma={volatility}, t={timenow}, T={timeatmaturity})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\Delta$')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.xlim(1e1, 1e7)
	plt.savefig("./plots/LookBackPutFloatingStrike/MonteCarloDeltaConvergence_LookBackPut_FloatingStrike.jpg")
	plt.clf()

	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloGamma_array, '-o', label=r'Monte Carlo $\Gamma$', c='C0')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower, MonteCarloGamma_array_upper, alpha=0.3, label=f'Naive {CIpercentage}% CI')
	plt.axhline(y=AnalyticGamma, color='r', linestyle='--', label=r'Analytic $\Gamma$')
	plt.xscale('log') 
	plt.title(f'Gamma of Floating Strike Look Back Put Option (S={Stockprice}, r={interest}, sigma={volatility}, t={timenow}, T={timeatmaturity})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\Gamma$ $(\$)^{-1}$')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.xlim(1e1, 1e7)
	plt.savefig("./plots/LookBackPutFloatingStrike/MonteCarloGammaConvergence_LookBackPut_FloatingStrike.jpg")
	plt.clf()		
	
	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloVega_array, '-o', label=r'Monte Carlo Vega', c='C0')
	plt.fill_between(NSim_array, MonteCarloVega_array_lower, MonteCarloVega_array_upper, alpha=0.3, label=f'Naive {CIpercentage}% CI')
	plt.axhline(y=AnalyticVega, color='r', linestyle='--', label=r'Analytic Vega')
	plt.xscale('log') 
	plt.title(f'Vega of Floating Strike Look Back Put Option (S={Stockprice}, r={interest}, sigma={volatility}, t={timenow}, T={timeatmaturity})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'Vega $( \$ \cdot \sqrt{\mathrm{year}} )$')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.xlim(1e1, 1e7)
	plt.savefig("./plots/LookBackPutFloatingStrike/MonteCarloVegaConvergence_LookBackPut_FloatingStrike.jpg")
	plt.clf()
	
	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloTheta_array, '-o', label=r'Monte Carlo $\Theta$', c='C0')
	plt.fill_between(NSim_array, MonteCarloTheta_array_lower, MonteCarloTheta_array_upper, alpha=0.3, label=f'Naive {CIpercentage}% CI')
	plt.axhline(y=AnalyticTheta, color='r', linestyle='--', label=r'Analytic $\Theta$')
	plt.xscale('log') 
	plt.title(f'Theta of Floating Strike Look Back Put Option (S={Stockprice}, r={interest}, sigma={volatility}, t={timenow}, T={timeatmaturity})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\Theta$ $( \$ / \mathrm{year} )$')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.xlim(1e1, 1e7)
	plt.savefig("./plots/LookBackPutFloatingStrike/MonteCarloThetaConvergence_LookBackPut_FloatingStrike.jpg")
	plt.clf()

	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloRho_array, '-o', label=r'Monte Carlo $\rho$', c='C0')
	plt.fill_between(NSim_array, MonteCarloRho_array_lower, MonteCarloRho_array_upper, alpha=0.3, label=f'Naive {CIpercentage}% CI')
	plt.axhline(y=AnalyticRho, color='r', linestyle='--', label=r'Analytic $\rho$')
	plt.xscale('log') 
	plt.title(f'Rho of Floating Strike Look Back Put Option (S={Stockprice}, r={interest}, sigma={volatility}, t={timenow}, T={timeatmaturity})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\rho$ $( \$ \cdot \mathrm{year} )$')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.xlim(1e1, 1e7)
	plt.savefig("./plots/LookBackPutFloatingStrike/MonteCarloRhoConvergence_LookBackPut_FloatingStrike.jpg")
	plt.clf()

	print('Generated plots saved in plots folder.')


if __name__ == "__main__":
	main()



