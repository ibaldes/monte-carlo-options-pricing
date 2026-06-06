import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Get current directory as a Path object
current_dir = Path.cwd()
grandparent_dir = current_dir.parent.parent
# Add grandparent directory to the system path
if str(grandparent_dir) not in sys.path:
    sys.path.insert(0, str(grandparent_dir))

from options_benchmark import LookBack as lb
from options_antithetic import LookBackAntithetic as at
from options_pathwise import LookBackPathwise as pw

### Compares the Monte-Carlo of the Look Back Fixed Strike Call with the Analytic  Formula ####
### Generates a plot, showing the convergence and error estimate ################################

##################################################################################
##################################################################################

def main():
	print('\n')	
	print ("Starting simulations...\n")

	NSim_array = np.array([10, 20, 40, 70, 100, 200, 400, 700, 1e3, 2e3, 4e3, 7e3, 1e4, 2e4, 4e4, 7e4, 1e5, 2e5, 4e5, 7e5, 1e6]) ##### array of n_simulations values to scan over
	NSim_array = np.array([10, 20, 40, 70, 100, 200, 400, 700, 1e3, 2e3, 4e3, 7e3, 1e4, 2e4, 4e4, 7e4, 1e5, 2e5, 4e5]) ##### array of n_simulations values to scan over
	n_examples = len(NSim_array)
	Output_array = np.zeros((n_examples, 12)) # Output array: MonteCarloFixedStrikeLookBackCallWithGreeks function has an output of length 12
	Output_array_2 = np.zeros((n_examples, 12)) # Output array: MonteCarloFixedStrikeLookBackCallWithGreeks function has an output of length 12	
	Output_array_3 = np.zeros((n_examples, 12)) # Output array: MonteCarloFixedStrikeLookBackCallWithGreeks function has an output of length 12
	Output_array_4 = np.zeros((n_examples, 12)) # Output array: MonteCarloFixedStrikeLookBackCallWithGreeks function has an output of length 12					

	### Choose some example values for our Option
	Stockprice = 80
	Strikeprice = 85
	interest = 0.05
	volatility = 0.4
	timenow = 0.25
	timeatmaturity = 0.5
	Smaxtodate = 110
	n_steps_1 = 100
	n_steps_2 = 500

	### Generate the Monte-Carlo prices and Greeks. Note we can increase n_steps to get a better theta estimate (current implementation using plus/minus one step to calculate derivative).

	for i in range(0,n_examples):
		print(f'Starting example with {int(NSim_array[i])} simulations and n_steps = {n_steps_1}\n')
		Output_array[i, :] = lb.MonteCarloFixedStrikeLookBackCallWithGreeks(Stockprice, Strikeprice, interest, volatility, timenow, timeatmaturity, Smaxtodate=Smaxtodate, n_steps=n_steps_1, n_simulations=NSim_array[i])

	print(f'\nPrices and Greeks using Monte-Carlo for the different n_simulations with n_steps = {n_steps_1} are:\n', Output_array)
	print('\n')

	for i in range(0,n_examples):
		print(f'Starting example with {int(NSim_array[i])} simulations and n_steps = {n_steps_2}\n')
		Output_array_2[i, :] = lb.MonteCarloFixedStrikeLookBackCallWithGreeks(Stockprice, Strikeprice, interest, volatility, timenow, timeatmaturity, Smaxtodate=Smaxtodate, n_steps=n_steps_2, n_simulations=NSim_array[i])

	print(f'\nPrices and Greeks using Monte-Carlo for the different n_simulations with n_steps = {n_steps_2} are:\n', Output_array_2)
	print('\n')
	
	for i in range(0,n_examples):
		print(f'Starting example with {int(NSim_array[i])} simulations, n_steps = {n_steps_2}, and Antithetic Variates\n')
		Output_array_3[i, :] = at.MonteCarloFixedStrikeLookBackCallWithGreeks(Stockprice, Strikeprice, interest, volatility, timenow, timeatmaturity, Smaxtodate=Smaxtodate, n_steps=n_steps_2, n_simulations=NSim_array[i])

	print(f'\nPrices and Greeks using Monte-Carlo for the different n_simulations with n_steps = {n_steps_2} and Antithetic Variates are:\n', Output_array_3)
	print('\n')
	
	for i in range(0,n_examples):
		print(f'Starting example with {int(NSim_array[i])} simulations, n_steps = {n_steps_2}, Antithetic Variates, and Pathwise Method\n')
		Output_array_4[i, :] = pw.MonteCarloFixedStrikeLookBackCallWithGreeks(Stockprice, Strikeprice, interest, volatility, timenow, timeatmaturity, Smaxtodate=Smaxtodate, n_steps=n_steps_2, n_simulations=NSim_array[i])

	print(f'\nPrices and Greeks using Monte-Carlo for the different n_simulations with n_steps = {n_steps_2}, Antithetic Variates, and Pathwise Method are:\n', Output_array_4)
	print('\n')			

	Analytic_array = np.array(lb.AnalyticFixedStrikeLookBackCallWithGreeks(Stockprice, Strikeprice, interest, volatility, timenow, timeatmaturity, Smaxtodate=Smaxtodate))
	print('\nPrices and Greeks using the analytic formula are:\n', Analytic_array)
	print('\n')

	AnalyticPrice = Analytic_array[0]
	MonteCarloPrice_array = Output_array[:, 0]
	MonteCarloPrice_StdErr_array = Output_array[:, 6]
	MonteCarloPrice_array_2 = Output_array_2[:, 0]
	MonteCarloPrice_StdErr_array_2 = Output_array_2[:, 6]
	MonteCarloPrice_array_3 = Output_array_3[:, 0]
	MonteCarloPrice_StdErr_array_3 = Output_array_3[:, 6]
	MonteCarloPrice_array_4 = Output_array_4[:, 0]
	MonteCarloPrice_StdErr_array_4 = Output_array_4[:, 6]			
	
	AnalyticDelta = Analytic_array[1]
	MonteCarloDelta_array = Output_array[:, 1]
	MonteCarloDelta_StdErr_array = Output_array[:, 7]
	MonteCarloDelta_array_2 = Output_array_2[:, 1]
	MonteCarloDelta_StdErr_array_2 = Output_array_2[:, 7]
	MonteCarloDelta_array_3 = Output_array_3[:, 1]
	MonteCarloDelta_StdErr_array_3 = Output_array_3[:, 7]
	MonteCarloDelta_array_4 = Output_array_4[:, 1]
	MonteCarloDelta_StdErr_array_4 = Output_array_4[:, 7]		
	
	AnalyticGamma = Analytic_array[2]
	MonteCarloGamma_array = Output_array[:, 2]
	MonteCarloGamma_StdErr_array = Output_array[:, 8]
	MonteCarloGamma_array_2 = Output_array_2[:, 2]
	MonteCarloGamma_StdErr_array_2 = Output_array_2[:, 8]
	MonteCarloGamma_array_3 = Output_array_3[:, 2]
	MonteCarloGamma_StdErr_array_3 = Output_array_3[:, 8]	
	MonteCarloGamma_array_4 = Output_array_4[:, 2]
	MonteCarloGamma_StdErr_array_4 = Output_array_4[:, 8]			

	AnalyticVega = Analytic_array[3]
	MonteCarloVega_array = Output_array[:, 3]
	MonteCarloVega_StdErr_array = Output_array[:, 9]
	MonteCarloVega_array_2 = Output_array_2[:, 3]
	MonteCarloVega_StdErr_array_2 = Output_array_2[:, 9]
	MonteCarloVega_array_3 = Output_array_3[:, 3]
	MonteCarloVega_StdErr_array_3 = Output_array_3[:, 9]
	MonteCarloVega_array_4 = Output_array_4[:, 3]
	MonteCarloVega_StdErr_array_4 = Output_array_4[:, 9]			

	AnalyticTheta = Analytic_array[4]
	MonteCarloTheta_array = Output_array[:, 4]
	MonteCarloTheta_StdErr_array = Output_array[:, 10]
	MonteCarloTheta_array_2 = Output_array_2[:, 4]
	MonteCarloTheta_StdErr_array_2 = Output_array_2[:, 10]
	MonteCarloTheta_array_3 = Output_array_3[:, 4]
	MonteCarloTheta_StdErr_array_3 = Output_array_3[:, 10]
	MonteCarloTheta_array_4 = Output_array_4[:, 4]
	MonteCarloTheta_StdErr_array_4 = Output_array_4[:, 10]	
	
	AnalyticRho = Analytic_array[5]
	MonteCarloRho_array = Output_array[:, 5]
	MonteCarloRho_StdErr_array = Output_array[:, 11]
	MonteCarloRho_array_2 = Output_array_2[:, 5]
	MonteCarloRho_StdErr_array_2 = Output_array_2[:, 11]
	MonteCarloRho_array_3 = Output_array_3[:, 5]
	MonteCarloRho_StdErr_array_3 = Output_array_3[:, 11]
	MonteCarloRho_array_4 = Output_array_4[:, 5]
	MonteCarloRho_StdErr_array_4 = Output_array_4[:, 11]		
	
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
	MonteCarloPrice_array_upper_2 = MonteCarloPrice_array_2+multiplcation_factor*MonteCarloPrice_StdErr_array_2
	MonteCarloPrice_array_lower_2 = MonteCarloPrice_array_2-multiplcation_factor*MonteCarloPrice_StdErr_array_2	
	MonteCarloPrice_array_upper_3 = MonteCarloPrice_array_3+multiplcation_factor*MonteCarloPrice_StdErr_array_3
	MonteCarloPrice_array_lower_3 = MonteCarloPrice_array_3-multiplcation_factor*MonteCarloPrice_StdErr_array_3
	MonteCarloPrice_array_upper_4 = MonteCarloPrice_array_4+multiplcation_factor*MonteCarloPrice_StdErr_array_4
	MonteCarloPrice_array_lower_4 = MonteCarloPrice_array_4-multiplcation_factor*MonteCarloPrice_StdErr_array_4			

	MonteCarloDelta_array_upper = MonteCarloDelta_array+multiplcation_factor*MonteCarloDelta_StdErr_array
	MonteCarloDelta_array_lower = MonteCarloDelta_array-multiplcation_factor*MonteCarloDelta_StdErr_array
	MonteCarloDelta_array_upper_2 = MonteCarloDelta_array_2+multiplcation_factor*MonteCarloDelta_StdErr_array_2
	MonteCarloDelta_array_lower_2 = MonteCarloDelta_array_2-multiplcation_factor*MonteCarloDelta_StdErr_array_2
	MonteCarloDelta_array_upper_3 = MonteCarloDelta_array_3+multiplcation_factor*MonteCarloDelta_StdErr_array_3
	MonteCarloDelta_array_lower_3 = MonteCarloDelta_array_3-multiplcation_factor*MonteCarloDelta_StdErr_array_3
	MonteCarloDelta_array_upper_4 = MonteCarloDelta_array_4+multiplcation_factor*MonteCarloDelta_StdErr_array_4
	MonteCarloDelta_array_lower_4 = MonteCarloDelta_array_4-multiplcation_factor*MonteCarloDelta_StdErr_array_4
	

	MonteCarloGamma_array_upper = MonteCarloGamma_array+multiplcation_factor*MonteCarloGamma_StdErr_array
	MonteCarloGamma_array_lower = MonteCarloGamma_array-multiplcation_factor*MonteCarloGamma_StdErr_array
	MonteCarloGamma_array_upper_2 = MonteCarloGamma_array_2+multiplcation_factor*MonteCarloGamma_StdErr_array_2
	MonteCarloGamma_array_lower_2 = MonteCarloGamma_array_2-multiplcation_factor*MonteCarloGamma_StdErr_array_2
	MonteCarloGamma_array_upper_3 = MonteCarloGamma_array_3+multiplcation_factor*MonteCarloGamma_StdErr_array_3
	MonteCarloGamma_array_lower_3 = MonteCarloGamma_array_3-multiplcation_factor*MonteCarloGamma_StdErr_array_3
	MonteCarloGamma_array_upper_4 = MonteCarloGamma_array_4+multiplcation_factor*MonteCarloGamma_StdErr_array_4
	MonteCarloGamma_array_lower_4 = MonteCarloGamma_array_4-multiplcation_factor*MonteCarloGamma_StdErr_array_4	
	
	MonteCarloVega_array_upper = MonteCarloVega_array+multiplcation_factor*MonteCarloVega_StdErr_array
	MonteCarloVega_array_lower = MonteCarloVega_array-multiplcation_factor*MonteCarloVega_StdErr_array	
	MonteCarloVega_array_upper_2 = MonteCarloVega_array_2+multiplcation_factor*MonteCarloVega_StdErr_array_2
	MonteCarloVega_array_lower_2 = MonteCarloVega_array_2-multiplcation_factor*MonteCarloVega_StdErr_array_2
	MonteCarloVega_array_upper_3 = MonteCarloVega_array_3+multiplcation_factor*MonteCarloVega_StdErr_array_3
	MonteCarloVega_array_lower_3 = MonteCarloVega_array_3-multiplcation_factor*MonteCarloVega_StdErr_array_3
	MonteCarloVega_array_upper_4 = MonteCarloVega_array_4+multiplcation_factor*MonteCarloVega_StdErr_array_4
	MonteCarloVega_array_lower_4 = MonteCarloVega_array_4-multiplcation_factor*MonteCarloVega_StdErr_array_4		


	MonteCarloTheta_array_upper = MonteCarloTheta_array+multiplcation_factor*MonteCarloTheta_StdErr_array
	MonteCarloTheta_array_lower = MonteCarloTheta_array-multiplcation_factor*MonteCarloTheta_StdErr_array
	MonteCarloTheta_array_upper_2 = MonteCarloTheta_array_2+multiplcation_factor*MonteCarloTheta_StdErr_array_2
	MonteCarloTheta_array_lower_2 = MonteCarloTheta_array_2-multiplcation_factor*MonteCarloTheta_StdErr_array_2
	MonteCarloTheta_array_upper_3 = MonteCarloTheta_array_3+multiplcation_factor*MonteCarloTheta_StdErr_array_3
	MonteCarloTheta_array_lower_3 = MonteCarloTheta_array_3-multiplcation_factor*MonteCarloTheta_StdErr_array_3
	MonteCarloTheta_array_upper_4 = MonteCarloTheta_array_4+multiplcation_factor*MonteCarloTheta_StdErr_array_4
	MonteCarloTheta_array_lower_4 = MonteCarloTheta_array_4-multiplcation_factor*MonteCarloTheta_StdErr_array_4	


	MonteCarloRho_array_upper = MonteCarloRho_array+multiplcation_factor*MonteCarloRho_StdErr_array
	MonteCarloRho_array_lower = MonteCarloRho_array-multiplcation_factor*MonteCarloRho_StdErr_array
	MonteCarloRho_array_upper_2 = MonteCarloRho_array_2+multiplcation_factor*MonteCarloRho_StdErr_array_2
	MonteCarloRho_array_lower_2 = MonteCarloRho_array_2-multiplcation_factor*MonteCarloRho_StdErr_array_2
	MonteCarloRho_array_upper_3 = MonteCarloRho_array_3+multiplcation_factor*MonteCarloRho_StdErr_array_3
	MonteCarloRho_array_lower_3 = MonteCarloRho_array_3-multiplcation_factor*MonteCarloRho_StdErr_array_3	
	MonteCarloRho_array_upper_4 = MonteCarloRho_array_4+multiplcation_factor*MonteCarloRho_StdErr_array_4
	MonteCarloRho_array_lower_4 = MonteCarloRho_array_4-multiplcation_factor*MonteCarloRho_StdErr_array_4
	
	##### make plots #########
		
	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloPrice_array, '-o', label=f'Monte Carlo Price n_steps = {n_steps_1}', c='C0')
	plt.fill_between(NSim_array, MonteCarloPrice_array_lower, MonteCarloPrice_array_upper, alpha=0.3, label=f' {CIpercentage}% CI n_steps = {n_steps_1}')
	plt.plot(NSim_array, MonteCarloPrice_array_2, '-o', label=f'Monte Carlo Price n_steps = {n_steps_2}', c='C1')	
	plt.fill_between(NSim_array, MonteCarloPrice_array_lower_2, MonteCarloPrice_array_upper_2, alpha=0.3, label=f' {CIpercentage}% CI  n_steps = {n_steps_2}')
	plt.plot(NSim_array, MonteCarloPrice_array_3, '-o', label=f'Monte Carlo Price Antithetic n_steps = {n_steps_2}', c='C2')	
	plt.fill_between(NSim_array, MonteCarloPrice_array_lower_3, MonteCarloPrice_array_upper_3, alpha=0.3, label=f' {CIpercentage}% CI Antithetic n_steps = {n_steps_2}')	
	plt.axhline(y=AnalyticPrice, color='r', linestyle='--', label='Analytic Price')
	plt.xscale('log') 
	plt.title(rf'Price of Fixed Strike Look Back Call Option (S={Stockprice}, K={Strikeprice}, r={interest}, $\sigma$={volatility}, t={timenow}, T={timeatmaturity}, $\bar{{S}}$={Smaxtodate})')
	plt.xlabel('Number of Simulations')
	plt.ylabel('Price [$]')
	plt.legend()
	plt.grid(True)
	plt.xlim(1e1, 1e6)
	plt.savefig("../../plots/LookBack/LookBackCallFixedStrike_PrevSmax/MonteCarloPriceConvergence_LookBackCall_FixedStrike_PrevSmax.jpg")
	plt.clf()

	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloDelta_array, '-o', label=fr'Monte Carlo $\Delta$ n_steps = {n_steps_1}', c='C0')
	plt.fill_between(NSim_array, MonteCarloDelta_array_lower, MonteCarloDelta_array_upper, alpha=0.3, label=f' {CIpercentage}% CI n_steps = {n_steps_1}')
	plt.plot(NSim_array, MonteCarloDelta_array_2, '-o', label=fr'Monte Carlo $\Delta$ n_steps = {n_steps_2}', c='C1')
	plt.fill_between(NSim_array, MonteCarloDelta_array_lower_2, MonteCarloDelta_array_upper_2, alpha=0.3, label=f' {CIpercentage}% CI n_steps = {n_steps_2}')	
	plt.plot(NSim_array, MonteCarloDelta_array_3, '-o', label=fr'Monte Carlo $\Delta$ Antithetic n_steps = {n_steps_2}', c='C2')
	plt.fill_between(NSim_array, MonteCarloDelta_array_lower_3, MonteCarloDelta_array_upper_3, alpha=0.3, label=f' {CIpercentage}% CI Antithetic n_steps = {n_steps_2}')
	plt.plot(NSim_array, MonteCarloDelta_array_4, '-o', label=fr'Monte Carlo $\Delta$ Antithetic Pathwise n_steps = {n_steps_2}', c='C3')
	plt.fill_between(NSim_array, MonteCarloDelta_array_lower_4, MonteCarloDelta_array_upper_4, alpha=0.3, label=f' {CIpercentage}% CI Antithetic Pathwise n_steps = {n_steps_2}')					
	plt.axhline(y=AnalyticDelta, color='r', linestyle='--', label=r'Analytic $\Delta$')
	plt.xscale('log') 
	plt.title(rf'Delta of Fixed Strike Look Back Call Option (S={Stockprice}, K={Strikeprice}, r={interest}, $\sigma$={volatility}, t={timenow}, T={timeatmaturity}, $\bar{{S}}$={Smaxtodate})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\Delta$')
	plt.legend()
	plt.grid(True)
	plt.xlim(1e1, 1e6)
	plt.savefig("../../plots/LookBack/LookBackCallFixedStrike_PrevSmax/MonteCarloDeltaConvergence_LookBackCall_FixedStrike_PrevSmax.jpg")
	plt.clf()
	
	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloGamma_array, '-o', label=fr'Monte Carlo $\Gamma$ n_steps = {n_steps_1}', c='C0')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower, MonteCarloGamma_array_upper, alpha=0.3, label=f' {CIpercentage}% CI n_steps ={n_steps_1}')
	plt.plot(NSim_array, MonteCarloGamma_array_2, '-o', label=fr'Monte Carlo $\Gamma$ n_steps = {n_steps_2}', c='C1')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower_2, MonteCarloGamma_array_upper_2, alpha=0.3, label=f' {CIpercentage}% CI n_steps = {n_steps_2}')
	plt.plot(NSim_array, MonteCarloGamma_array_3, '-o', label=fr'Monte Carlo $\Gamma$ Antithetic n_steps = {n_steps_2}', c='C2')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower_3, MonteCarloGamma_array_upper_3, alpha=0.3, label=f' {CIpercentage}% CI Antithetic n_steps = {n_steps_2}')
	plt.plot(NSim_array, MonteCarloGamma_array_4, '-o', label=fr'Monte Carlo $\Gamma$ Antithetic Pathwise n_steps = {n_steps_2}', c='C3')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower_4, MonteCarloGamma_array_upper_4, alpha=0.3, label=f' {CIpercentage}% CI Antithetic Pathwise n_steps = {n_steps_2}')			
	plt.axhline(y=AnalyticGamma, color='r', linestyle='--', label=r'Analytic $\Gamma$')
	plt.xscale('log') 
	plt.title(rf'Gamma of Fixed Strike Look Back Call Option (S={Stockprice}, K={Strikeprice}, r={interest}, $\sigma$={volatility}, t={timenow}, T={timeatmaturity}, $\bar{{S}}$={Smaxtodate})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\Gamma$ [$ 1/\$ $]')
	plt.legend()
	plt.grid(True)
	plt.xlim(1e1, 1e6)
	plt.savefig("../../plots/LookBack/LookBackCallFixedStrike_PrevSmax/MonteCarloGammaConvergence_LookBackCall_FixedStrike_PrevSmax.jpg")
	plt.clf()

	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloGamma_array, '-o', label=fr'Monte Carlo $\Gamma$ n_steps = {n_steps_1}', c='C0')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower, MonteCarloGamma_array_upper, alpha=0.3, label=f' {CIpercentage}% CI n_steps ={n_steps_1}')
	plt.plot(NSim_array, MonteCarloGamma_array_2, '-o', label=fr'Monte Carlo $\Gamma$ n_steps = {n_steps_2}', c='C1')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower_2, MonteCarloGamma_array_upper_2, alpha=0.3, label=f' {CIpercentage}% CI n_steps = {n_steps_2}')
	plt.plot(NSim_array, MonteCarloGamma_array_3, '-o', label=fr'Monte Carlo $\Gamma$ Antithetic n_steps = {n_steps_2}', c='C2')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower_3, MonteCarloGamma_array_upper_3, alpha=0.3, label=f' {CIpercentage}% CI Antithetic n_steps = {n_steps_2}')
	plt.plot(NSim_array, MonteCarloGamma_array_4, '-o', label=fr'Monte Carlo $\Gamma$ Antithetic Pathwise n_steps = {n_steps_2}', c='C3')
	plt.fill_between(NSim_array, MonteCarloGamma_array_lower_4, MonteCarloGamma_array_upper_4, alpha=0.3, label=f' {CIpercentage}% CI Antithetic Pathwise n_steps = {n_steps_2}')					
	plt.axhline(y=AnalyticGamma, color='r', linestyle='--', label=r'Analytic $\Gamma$')
	plt.xscale('log') 
	plt.title(rf'Gamma of Fixed Strike Look Back Call Option (S={Stockprice}, K={Strikeprice}, r={interest}, $\sigma$={volatility}, t={timenow}, T={timeatmaturity}, $\bar{{S}}$={Smaxtodate})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\Gamma$ [$ 1/\$ $]')
	plt.legend()
	plt.grid(True)
	plt.xlim(1e1, 1e6)
	plt.ylim(0, 0.2)	
	plt.savefig("../../plots/LookBack/LookBackCallFixedStrike_PrevSmax/MonteCarloGammaConvergence_LookBackCall_FixedStrike_PrevSmax_ZoomedIn.jpg")
	plt.clf()			
	
	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloVega_array, '-o', label=fr'Monte Carlo Vega n_steps = {n_steps_1}', c='C0')
	plt.fill_between(NSim_array, MonteCarloVega_array_lower, MonteCarloVega_array_upper, alpha=0.3, label=f' {CIpercentage}% CI n_steps ={n_steps_1}')
	plt.plot(NSim_array, MonteCarloVega_array_2, '-o', label=fr'Monte Carlo Vega n_steps = {n_steps_2}', c='C1')
	plt.fill_between(NSim_array, MonteCarloVega_array_lower_2, MonteCarloVega_array_upper_2, alpha=0.3, label=f' {CIpercentage}% CI n_steps ={n_steps_2}')
	plt.plot(NSim_array, MonteCarloVega_array_3, '-o', label=fr'Monte Carlo Vega Antithetic n_steps = {n_steps_2}', c='C2')
	plt.fill_between(NSim_array, MonteCarloVega_array_lower_3, MonteCarloVega_array_upper_3, alpha=0.3, label=f' {CIpercentage}% CI Antithetic n_steps ={n_steps_2}')			
	plt.axhline(y=AnalyticVega, color='r', linestyle='--', label=r'Analytic Vega')
	plt.xscale('log') 
	plt.title(rf'Vega of Fixed Strike Look Back Call Option (S={Stockprice}, K={Strikeprice}, r={interest}, $\sigma$={volatility}, t={timenow}, T={timeatmaturity}, $\bar{{S}}$={Smaxtodate})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'Vega [$ \$ \cdot \sqrt{\mathrm{year}}$ ]')
	plt.legend()
	plt.grid(True)
	plt.xlim(1e1, 1e6)
	plt.savefig("../../plots/LookBack/LookBackCallFixedStrike_PrevSmax/MonteCarloVegaConvergence_LookBackCall_FixedStrike_PrevSmax.jpg")
	plt.clf()
	
	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloTheta_array, '-o', label=fr'Monte Carlo $\Theta$ n_steps ={n_steps_1}', c='C0')
	plt.fill_between(NSim_array, MonteCarloTheta_array_lower, MonteCarloTheta_array_upper, alpha=0.3, label=f' {CIpercentage}% CI n_steps ={n_steps_1}')
	plt.plot(NSim_array, MonteCarloTheta_array_2, '-o', label=fr'Monte Carlo $\Theta$ n_steps ={n_steps_2}', c='C1')
	plt.fill_between(NSim_array, MonteCarloTheta_array_lower_2, MonteCarloTheta_array_upper_2, alpha=0.3, label=f' {CIpercentage}% CI n_steps ={n_steps_2}')
	plt.plot(NSim_array, MonteCarloTheta_array_3, '-o', label=fr'Monte Carlo $\Theta$ Antithetic n_steps ={n_steps_2}', c='C2')
	plt.fill_between(NSim_array, MonteCarloTheta_array_lower_3, MonteCarloTheta_array_upper_3, alpha=0.3, label=f' {CIpercentage}% CI Antithetic n_steps ={n_steps_2}')		
	plt.axhline(y=AnalyticTheta, color='r', linestyle='--', label=r'Analytic $\Theta$')
	plt.xscale('log') 
	plt.title(rf'Theta of Fixed Strike Look Back Call Option (S={Stockprice}, K={Strikeprice}, r={interest}, $\sigma$={volatility}, t={timenow}, T={timeatmaturity}, $\bar{{S}}$={Smaxtodate})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\Theta$ [$ \$ / \mathrm{year} $]')
	plt.legend()
	plt.grid(True)
	plt.xlim(1e1, 1e6)
	plt.savefig("../../plots/LookBack/LookBackCallFixedStrike_PrevSmax/MonteCarloThetaConvergence_LookBackCall_FixedStrike_PrevSmax.jpg")
	plt.clf()

	plt.figure(figsize=(8, 6))
	plt.plot(NSim_array, MonteCarloRho_array, '-o', label=fr'Monte Carlo $\rho$ n_steps ={n_steps_1}', c='C0')
	plt.fill_between(NSim_array, MonteCarloRho_array_lower, MonteCarloRho_array_upper, alpha=0.3, label=f' {CIpercentage}% CI n_steps ={n_steps_1}')
	plt.plot(NSim_array, MonteCarloRho_array_2, '-o', label=fr'Monte Carlo $\rho$ n_steps ={n_steps_2}', c='C1')
	plt.fill_between(NSim_array, MonteCarloRho_array_lower_2, MonteCarloRho_array_upper_2, alpha=0.3, label=f' {CIpercentage}% CI n_steps ={n_steps_2}')
	plt.plot(NSim_array, MonteCarloRho_array_3, '-o', label=fr'Monte Carlo $\rho$ Antithetic n_steps ={n_steps_2}', c='C2')
	plt.fill_between(NSim_array, MonteCarloRho_array_lower_3, MonteCarloRho_array_upper_3, alpha=0.3, label=f' {CIpercentage}% CI Antithetic n_steps ={n_steps_2}')		
	plt.axhline(y=AnalyticRho, color='r', linestyle='--', label=r'Analytic $\rho$')
	plt.xscale('log') 
	plt.title(rf'Rho of Fixed Strike Look Back Call Option (S={Stockprice}, K={Strikeprice}, r={interest}, $\sigma$={volatility}, t={timenow}, T={timeatmaturity}, $\bar{{S}}$={Smaxtodate})')
	plt.xlabel('Number of Simulations')
	plt.ylabel(r'$\rho$ [$ \$ \cdot \mathrm{year} $]')
	plt.legend()
	plt.grid(True)
	plt.xlim(1e1, 1e6)
	plt.savefig("../../plots/LookBack/LookBackCallFixedStrike_PrevSmax/MonteCarloRhoConvergence_LookBackCall_FixedStrike_PrevSmax.jpg")
	plt.clf()

	print('Generated plots saved in plots folder.')


if __name__ == "__main__":
	main()



