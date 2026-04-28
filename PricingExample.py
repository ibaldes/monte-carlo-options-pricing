import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
from scipy.stats import norm
from multiprocessing import Pool

import EuropeanVanilla as ev
import EuropeanBarrier as eb

### PROVIDES EXAMPLES OF HOW TO USE THE ANALYTIC AND MONTE-CARLO PRICING FUNCTIONS 
### WE CAN CHECK THE VALUES ARE CONSISTENT (UP TO RANDOM ERROS FROM THE MONTE-CARLO) IN THE RELEVANT LIMITS

print("Number of cpus : ", mp.cpu_count())
pool = Pool(processes=(mp.cpu_count() - 1))

StandardBaseSeed = 0

##################################################################################
##################################################################################

def main():
	print('\n')
	print('**********************************************************************************************************************')
	print('Call Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')
	print('\n')	
	
	print('\nEuropean Call Analytic\n', ev.BlackScholesVanillaEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))
	print('\nEuropean Call Monte-Carlo\n', ev.MonteCarloVanillaEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))		

	print('\nEuropean Call Monte-Carlo with Knock In at S=0\n', eb.MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 0))
	print('\nEuropean Call Monte-Carlo with Knock In at S=85\n', eb.MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=85\n', eb.MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=1000\n', eb.MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 1000))
	print('\nEuropean Call Monte-Carlo with Knock In at S=75\n', eb.MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=75\n', eb.MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	print('\nEuropean Call Monte-Carlo with Knock In at S=110\n', eb.MonteCarloKnockInEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))
	print('\nEuropean Call Monte-Carlo with Knock Out at S=110\n', eb.MonteCarloKnockOutEuropeanCallWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))

	print('\n')
	print('**********************************************************************************************************************')
	print('**********************************************************************************************************************')
	print('\n')

	print('Put Option Prices with S=80, K=85, r=0.05, sigma=0.4, t=1, T=1.25')
	print('\n')
	
	print('\nEuropean Put Analytic\n', ev.BlackScholesVanillaEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))
	print('\nEuropean Put Monte-Carlo\n', ev.MonteCarloVanillaEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25))

	print('\nEuropean Put Monte-Carlo with Knock In at S=0\n', eb.MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 0))
	print('\nEuropean Put Monte-Carlo with Knock In at S=85\n', eb.MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	print('\nEuropean Put Monte-Carlo with Knock Out at S=85\n', eb.MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 85))
	print('\nEuropean Put Monte-Carlo with Knock Out at S=1000\n', eb.MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 1000))
	print('\nEuropean Put Monte-Carlo with Knock In at S=75\n', eb.MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	print('\nEuropean Put Monte-Carlo with Knock Out at S=75\n', eb.MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 75))
	print('\nEuropean Put Monte-Carlo with Knock In at S=110\n', eb.MonteCarloKnockInEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))
	print('\nEuropean Put Monte-Carlo with Knock Out at S=110\n', eb.MonteCarloKnockOutEuropeanPutWithGreeks(80, 85, 0.05, 0.4, 1, 1.25, 110))

	print('\n')
	print('**********************************************************************************************************************')
	print('\n')

if __name__ == '__main__':
	main()
	
