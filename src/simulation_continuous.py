#!/bin/python

import sys
import numpy as np
from utils import *
from scipy.stats import mvn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import tqdm
import pickle
import scipy.spatial
plt.rcParams.update({'font.size': 13})


def generate_continuous_prob(lo = None, hi = None):
	'''
		simulate Gaussian distribution for a gene pair
	'''
	mu = np.random.uniform(low=0.1, high=10, size=2)
	std_x = np.random.uniform(low=1e-2, high=mu[0]/4, size=1)[0]
	std_y = np.random.uniform(low=1e-2, high=mu[1]/4, size=1)[0]
	if (lo is None) or (hi is None):
		max_cov = std_x * std_y
		cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
	else:
		rho_min = 1 - np.exp(-2 * lo)
		rho_max = 1 - np.exp(-2 * hi)
		cov = (np.random.uniform(low=-1, high=1) > 0) * np.random.uniform(low=rho_min * std_x * std_y, high=rho_max * std_x * std_y, size=1)[0]
	covariance = np.array([[std_x*std_x, cov], [cov, std_y*std_y]])
	return mu, covariance


def generate_bootstrap_covariance(covariance):
	boot_var_x = np.random.uniform(low=0.01*covariance[0,0], high=0.5 * covariance[0,0], size=1)[0]
	boot_var_y = np.random.uniform(low=0.01*covariance[1,1], high=0.5 * covariance[1,1], size=1)[0]
	max_cov = np.sqrt(boot_var_x * boot_var_y)
	boot_cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
	boot_covariance = np.array([[boot_var_x, boot_cov], [boot_cov, boot_var_y]])
	return boot_covariance


def basic_sample_observation(mu, covariance, boot_covariance, n_sample=50, n_bootstrap=5):
	'''
		output: real-valued points of dimension n_bootstrap * n_sample * 2
		The observation are real valued.
		First sample the hidden true value from multivariate Gaussian distribution, 
		then sample the measurement error from another multivatiate Gaussian distribution,
		finally add the measurement error to the true value to get the observation.
	'''
	hidden_value = np.random.multivariate_normal(mean=mu, cov=covariance, size=n_sample)
	measure_error = np.random.multivariate_normal(mean=np.zeros(2), cov=boot_covariance, size=(n_bootstrap, n_sample))
	observation = hidden_value + measure_error
	return hidden_value, observation


def generate_mixture_gaussian(k):
	'''
		generate a mixture of k Gaussian distribution, each has a mean mu, a covariance, and a weight
		output:
			multi_mu: mean of size k * 2
			multi_covariance: covariance matrix of k * 2 * 2
			weights: weight of size k
	'''
	# mean
	multi_mu = np.random.uniform(low=0.1, high=10, size=(k, 2))
	# covariance
	std_x = np.random.uniform(low=1e-2, high=multi_mu[:, 0]/4)
	std_y = np.random.uniform(low=1e-2, high=multi_mu[:, 1]/4)
	max_cov = std_x * std_y
	cov = np.random.uniform(low=-max_cov, high=max_cov)
	multi_covariance = np.zeros( (k, 2, 2) )
	for i in range(k):
		multi_covariance[i,:,:] = np.array([[std_x[i]*std_x[i], cov[i]], [cov[i], std_y[i]*std_y[i]]])
	# weight
	weights = np.random.dirichlet(np.ones(k))
	assert( np.abs(np.sum(weights) - 1) < 1e-4 )
	return multi_mu, multi_covariance, weights


def generate_bootstrap_covariance_mixture(multi_covariance):
	multi_boot_covariance = np.zeros( multi_covariance.shape )
	for i in range(multi_covariance.shape[0]):
		multi_boot_covariance[i,:,:] = generate_bootstrap_covariance(multi_covariance[i,:,:])
	return multi_boot_covariance


def generate_bootstrap_covariance_mixture_two(multi_covariance):
	multi_boot_covariance1 = np.zeros( multi_covariance.shape )
	multi_boot_covariance2 = np.zeros( multi_covariance.shape )
	for i in range(multi_covariance.shape[0]):
		multi_boot_covariance1[i,:,:] = generate_bootstrap_covariance(multi_covariance[i,:,:])
		multi_boot_covariance2[i,:,:] = generate_bootstrap_covariance(multi_covariance[i,:,:])
	# weights
	error_weights = np.random.dirichlet(np.ones(2))
	return multi_boot_covariance1, multi_boot_covariance2, error_weights


def sample_observation_mixture(multi_mu, multi_covariance, multi_boot_covariance, weights, n_sample, n_bootstrap):
	'''
		output: real-valued points of dimension n_bootstrap * n_sample * 2
		procedure: sample n_sample * weight observations for each mixture component and concatenate them
	'''
	k = multi_mu.shape[0]
	hidden = None
	observation = None
	for i in range(k):
		ni = int(np.round(n_sample * weights[i]))
		hid, obs = basic_sample_observation(multi_mu[i,:], multi_covariance[i,:,:], multi_boot_covariance[i,:,:], ni, n_bootstrap)
		if hidden is None:
			hidden = hid
			observation = obs
		else:
			hidden = np.vstack( (hidden, hid) )
			observation = np.concatenate( (observation, obs) , axis=1)
	return hidden, observation


def basic_sample_observation_two(mu, covariance, boot_covariance0, boot_covariance1, error_weights, n_sample=50, n_bootstrap=5):
	'''
		output: real-valued points of dimension n_bootstrap * n_sample * 2
		First sample the hidden true value from multivariate Gaussian distribution, 
		then sample the measurement error from a mixture of multivatiate Gaussian distributions (the number of each error is n_bootstrap * error_weights[i]),
		finally add the measurement error to the true value to get the observation.
	'''
	hidden_value = np.random.multivariate_normal(mean=mu, cov=covariance, size=n_sample)
	measure_error0 = np.random.multivariate_normal(mean=np.zeros(2), cov=boot_covariance0, size=(int(np.round(n_bootstrap * error_weights[0])), n_sample))
	measure_error1 = np.random.multivariate_normal(mean=np.zeros(2), cov=boot_covariance1, size=(int(np.round(n_bootstrap * error_weights[1])), n_sample))
	observation = np.concatenate( (hidden_value + measure_error0, hidden_value + measure_error1), axis=0)
	return hidden_value, observation


def sample_observation_mixture_two(multi_mu, multi_covariance, weights, multi_boot_covariance1, multi_boot_covariance2, error_weights, n_sample, n_bootstrap):
	'''
		output: real-valued points of dimension n_bootstrap * n_sample * 2
		procedure: sample n_sample * weight observations for each mixture component and concatenate them
	'''
	k = multi_mu.shape[0]
	hidden = None
	observation = None
	for i in range(k):
		ni = int(np.round(n_sample * weights[i]))
		hid, obs = basic_sample_observation_two(multi_mu[i,:], multi_covariance[i,:,:], multi_boot_covariance1[i,:,:], multi_boot_covariance2[i,:,:], error_weights, ni, n_bootstrap)
		if hidden is None:
			hidden = hid
			observation = obs
		else:
			hidden = np.vstack( (hidden, hid) )
			observation = np.concatenate( (observation, obs) , axis=1)
	return hidden, observation


def sample_observation_linear(n_sample, n_bootstrap):
	# slope between x and y
	slope = np.random.uniform(low=0, high=100)
	intercept = np.random.uniform(low=0, high = 10)
	# sample x and calculate y based on the slope and intercept
	x = np.random.uniform(low=0, high=10, size=n_sample)
	y = x * slope + intercept
	hidden_value = np.vstack( (x,y) ).transpose()
	# sample error boot_covariance
	boot_var_x = np.random.uniform(low=0.01*np.cov(x), high=0.5*np.cov(x))
	boot_var_y = np.random.uniform(low=0.01*np.cov(y), high=0.5*np.cov(y))
	max_cov = np.sqrt(boot_var_x * boot_var_y)
	boot_cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
	boot_covariance = np.array([[boot_var_x, boot_cov], [boot_cov, boot_var_y]])
	measure_error = np.random.multivariate_normal(mean=np.zeros(2), cov=boot_covariance, size=(n_bootstrap, n_sample))
	observation = hidden_value + measure_error
	return boot_covariance, hidden_value, observation


def sample_observation_linear_enlargingcov(n_sample, n_bootstrap, n_types=10):
	# slope between x and y
	slope = np.random.uniform(low=0, high=100)
	intercept = np.random.uniform(low=0, high = 10)
	# sample x and calculate y based on the slope and intercept
	x = np.random.uniform(low=0, high=10, size=n_sample)
	y = x * slope + intercept
	hidden_value = np.vstack( (x,y) ).transpose()
	# global covariance without considering the largeness of x and y
	boot_var_x = np.random.uniform(low=0.01*np.cov(x), high=np.cov(x))
	boot_var_y = np.random.uniform(low=0.01*np.cov(y), high=np.cov(y))
	max_cov = np.sqrt(boot_var_x * boot_var_y)
	boot_cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
	boot_covariance = np.array([[boot_var_x, boot_cov], [boot_cov, boot_var_y]])
	# for a partition of n_types on the x axis, multiply the global boot_covariance by (partition mean / global mean)**2
	observation = np.zeros( (n_bootstrap, n_sample, 2) )
	for i in range(n_types):
		lb = max(0, 10 / n_types * i)
		ub = 10 / n_types * (i+1)
		indexes = np.where(np.logical_and(x >= lb, x < ub))[0]
		mu = np.mean(x[indexes])
		multiplier = (mu / np.mean(x)) ** 2
		measure_error = np.random.multivariate_normal(mean=np.zeros(2), cov=multiplier*boot_covariance, size=(n_bootstrap, len(indexes)))
		observation[:, indexes, :] = hidden_value[indexes, :] + measure_error
	return boot_covariance, hidden_value, observation


def sample_observation_sine(n_sample, n_bootstrap):
	# amplification
	amplification = np.random.uniform(low=0.01, high=5)
	# sample x and calculate y based on the sine function of x
	x = np.random.uniform(low=0, high=10, size=n_sample)
	y = amplification * np.sin(x)
	hidden_value = np.vstack( (x,y) ).transpose()
	# sample error boot_covariance
	boot_var_x = np.random.uniform(low=0.01*np.cov(x), high=0.5*np.cov(x))
	boot_var_y = np.random.uniform(low=0.01*np.cov(y), high=0.5*np.cov(y))
	max_cov = np.sqrt(boot_var_x * boot_var_y)
	boot_cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
	boot_covariance = np.array([[boot_var_x, boot_cov], [boot_cov, boot_var_y]])
	measure_error = np.random.multivariate_normal(mean=np.zeros(2), cov=boot_covariance, size=(n_bootstrap, n_sample))
	observation = hidden_value + measure_error
	return boot_covariance, hidden_value, observation


def sample_observation_independent(n_sample, n_bootstrap):
	# sample x and calculate y based on the sine function of x
	x = np.random.uniform(low=0, high=10, size=n_sample)
	y = np.random.uniform(low=0, high=10, size=n_sample)
	hidden_value = np.vstack( (x,y) ).transpose()
	# sample error boot_covariance
	boot_var_x = np.random.uniform(low=0.01*np.cov(x), high=0.5*np.cov(x))
	boot_var_y = np.random.uniform(low=0.01*np.cov(y), high=0.5*np.cov(y))
	max_cov = np.sqrt(boot_var_x * boot_var_y)
	boot_cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
	boot_covariance = np.array([[boot_var_x, boot_cov], [boot_cov, boot_var_y]])
	measure_error = np.random.multivariate_normal(mean=np.zeros(2), cov=boot_covariance, size=(n_bootstrap, n_sample))
	observation = hidden_value + measure_error
	return boot_covariance, hidden_value, observation



def run_1(script_dir):
	'''
		single Gaussian distribution with known small MI (independent), and show that the correction is not helpful.
	'''
	sample_sizes = [5000]
	bootstrap_sizes = [20]
	n_simulation = 100
	truth = [0.1, 0.2, 0.3]
	# initial result vector
	result_mi_real = {}
	result_mi_raw = {}
	result_mi_corr = {}
	for t in truth:
		for n_sample in sample_sizes:
			for n_bootstrap in bootstrap_sizes:
				result_mi_real[(t, n_sample, n_bootstrap)] = []
				result_mi_raw[(t, n_sample, n_bootstrap)] = []
				result_mi_corr[(t, n_sample, n_bootstrap)] = []
	# run simulation
	np.random.seed(0)
	with tqdm.tqdm(total=n_simulation) as pbar:
		for i in range(n_simulation):
			for t in truth:
				mu, covariance = generate_continuous_prob(t - 0.1, t)
				bootstrap_covariance = generate_bootstrap_covariance(covariance)
				mi_real = MI_singleGaussian(covariance)
				for n_sample in sample_sizes:
					for n_bootstrap in bootstrap_sizes:
						hidden, observation = basic_sample_observation(mu, covariance, bootstrap_covariance, n_sample, n_bootstrap)
						# for MI using hidden and raw mean observed
						x = np.mean(observation[:,:,0], axis=0)
						y = np.mean(observation[:,:,1], axis=0)
						h_x = np.exp(0.2 * np.log(4 / 3 / n_sample)) * np.std(x)
						h_y = np.exp(0.2 * np.log(4 / 3 / n_sample)) * np.std(y)
						_,_,_,mi_raw = MI_KDE(x, y, h_x, h_y)
						# for MI using correction
						covariance_est =  BootstrapCovariance(observation[:, :, 0], observation[:, :, 1])
						tmp_xsi, tmp_P = np.linalg.eig(covariance_est * 2)
						tmp_P = tmp_P.transpose()
						Sigma_1_2 = tmp_P.transpose().dot(np.diag(np.sqrt(tmp_xsi))).dot(tmp_P)
						xsi,_ = np.linalg.eig( Sigma_1_2.dot(np.diag(1 / 2 / np.array([h_x**2, h_y**2]))).dot(Sigma_1_2) )
						if np.max(xsi * 2) > 0.5:
							covariance_est /= (np.max(xsi * 2) / 0.5)
						mi_corr = MI_KDE_correction(x, y, h_x, h_y, covariance_est)
						# update result vector
						result_mi_real[(t, n_sample, n_bootstrap)].append(mi_real)
						result_mi_raw[(t, n_sample, n_bootstrap)].append(mi_raw)
						result_mi_corr[(t, n_sample, n_bootstrap)].append(mi_corr)
			pbar.update(1)
	for t in truth:
		for n_sample in sample_sizes:
			for n_bootstrap in bootstrap_sizes:
				result_mi_real[(t, n_sample, n_bootstrap)] = np.array(result_mi_real[(t, n_sample, n_bootstrap)])
				result_mi_raw[(t, n_sample, n_bootstrap)] = np.array(result_mi_raw[(t, n_sample, n_bootstrap)])
				result_mi_corr[(t, n_sample, n_bootstrap)] = np.array(result_mi_corr[(t, n_sample, n_bootstrap)])
				print( (np.mean(np.abs( result_mi_real[(t, n_sample, n_bootstrap)] - result_mi_raw[(t, n_sample, n_bootstrap)] )), np.mean(np.abs( result_mi_real[(t, n_sample, n_bootstrap)] - result_mi_corr[(t, n_sample, n_bootstrap)] ))) )
	# save result
	pickle.dump( result_mi_real, open( script_dir + "/results/simulation_independence_real.pkl", "wb" ) )
	pickle.dump( result_mi_raw, open( script_dir + "/results/simulation_independence_raw.pkl", "wb" ) )
	pickle.dump( result_mi_corr, open( script_dir + "/results/simulation_independence_corr.pkl", "wb" ) )


def run_3(script_dir):
	# mixture of Gaussian, the error distribution is mixture of two Gaussian distributions for each mixture
	mixtures = [2, 5, 10, 20]
	sample_sizes = [500, 1000, 5000, 10000]
	bootstrap_sizes = [20, 50, 100]
	n_simulation = 100
	# initial result vector
	result_mi_real = {}
	result_mi_raw = {}
	result_mi_corr = {}
	for k in mixtures:
		for n_sample in sample_sizes:
			for n_bootstrap in bootstrap_sizes:
				result_mi_real[(k, n_sample, n_bootstrap)] = []
				result_mi_raw[(k, n_sample, n_bootstrap)] = []
				result_mi_corr[(k, n_sample, n_bootstrap)] = []
	# start simulation
	np.random.seed(0)
	with tqdm.tqdm(total=n_simulation) as pbar:
		for i in range(n_simulation):
			for k in mixtures:
				multi_mu, multi_covariance, weights = generate_mixture_gaussian(k)
				multi_boot_covariance1, multi_boot_covariance2, error_weights = generate_bootstrap_covariance_mixture_two(multi_covariance)
				for n_sample in sample_sizes:
					for n_bootstrap in bootstrap_sizes:
						hidden, observation = sample_observation_mixture_two(multi_mu, multi_covariance, weights, multi_boot_covariance1, multi_boot_covariance2, error_weights, n_sample, n_bootstrap)
						# for MI using hidden and raw mean observed
						x = np.mean(observation[:,:,0], axis=0)
						y = np.mean(observation[:,:,1], axis=0)
						h_x = np.exp(0.2 * np.log(4 / 3 / n_sample)) * np.std(x)
						h_y = np.exp(0.2 * np.log(4 / 3 / n_sample)) * np.std(y)
						_,_,_,mi_real = MI_KDE(hidden[:,0], hidden[:,1], h_x, h_y)
						_,_,_,mi_raw = MI_KDE(x, y, h_x, h_y)
						# for MI using correction
						multi_covariance_est, label =  BootstrapCovariance_cluster(observation[:, :, 0], observation[:, :, 1], k)
						for j in range(multi_covariance_est.shape[0]):
							tmp_xsi, tmp_P = np.linalg.eig(multi_covariance_est[j,:,:] * 2)
							tmp_P = tmp_P.transpose()
							Sigma_1_2 = tmp_P.transpose().dot(np.diag(np.sqrt(tmp_xsi))).dot(tmp_P)
							xsi,_ = np.linalg.eig( Sigma_1_2.dot(np.diag(1 / 2 / np.array([h_x**2, h_y**2]))).dot(Sigma_1_2) )
							if np.max(xsi * 2) > 0.5:
								multi_covariance_est[j,:,:] /= (np.max(xsi * 2) / 0.5)
						_,_,_,mi_corr = MI_KDE_correction_cluster(x, y, h_x, h_y, multi_covariance_est, label)
						# update result vector
						result_mi_real[(k, n_sample, n_bootstrap)].append(mi_real)
						result_mi_raw[(k, n_sample, n_bootstrap)].append(mi_raw)
						result_mi_corr[(k, n_sample, n_bootstrap)].append(mi_corr)
						# print( (mi_real, mi_raw, mi_corr) )
			pbar.update(1)
	# convert to numpy vector
	for k in mixtures:
		for n_sample in sample_sizes:
			for n_bootstrap in bootstrap_sizes:
				result_mi_real[(k, n_sample, n_bootstrap)] = np.array(result_mi_real[(k, n_sample, n_bootstrap)])
				result_mi_raw[(k, n_sample, n_bootstrap)] = np.array(result_mi_raw[(k, n_sample, n_bootstrap)])
				result_mi_corr[(k, n_sample, n_bootstrap)] = np.array(result_mi_corr[(k, n_sample, n_bootstrap)])
				print( (np.mean(np.abs( result_mi_real[(k, n_sample, n_bootstrap)] - result_mi_raw[(k, n_sample, n_bootstrap)] )), np.mean(np.abs( result_mi_real[(k, n_sample, n_bootstrap)] - result_mi_corr[(k, n_sample, n_bootstrap)] ))) )
	# save result
	pickle.dump( result_mi_real, open( script_dir + "/results/simulation_double_mixture_real_05.pkl", "wb" ) )
	pickle.dump( result_mi_raw, open( script_dir + "/results/simulation_double_mixture_raw_05.pkl", "wb" ) )
	pickle.dump( result_mi_corr, open( script_dir + "/results/simulation_double_mixture_corr_05.pkl", "wb" ) )


def plot_combined(script_dir):
	result_mi_real = pickle.load( open( script_dir + "/results/simulation_double_mixture_real_05.pkl", 'rb') )
	result_mi_raw = pickle.load( open( script_dir + "/results/simulation_double_mixture_raw_05.pkl", 'rb') )
	result_mi_corr = pickle.load( open( script_dir + "/results/simulation_double_mixture_corr_05.pkl", 'rb') )
	result_inde_mi_real = pickle.load( open( script_dir + "/results/simulation_independence_real.pkl", 'rb') )
	result_inde_mi_raw = pickle.load( open( script_dir + "/results/simulation_independence_raw.pkl", 'rb') )
	result_inde_mi_corr = pickle.load( open( script_dir + "/results/simulation_independence_corr.pkl", 'rb') )
	# plot
	fig, axes = plt.subplots(2, 2, figsize=(9.69, 7))
	# mixture subplot 1
	df1 = None
	for k,v in result_mi_real.items():
		if k[0] != 10 or k[2] != 20:
			continue
		tmp = pd.DataFrame( {"sample size" : k[1], "abs diff between MI" : np.abs(v - result_mi_raw[k]), "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"sample size" : k[1], "abs diff between MI" : np.abs(v - result_mi_corr[k]), "label" : "corrected"} )
		if df1 is None:
			df1 = tmp
			df1 = df1.append(tmp2)
		else:
			df1 = df1.append(tmp, ignore_index = True)
			df1 = df1.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df1, x="sample size", y="abs diff between MI", hue="label", cut=0, ax=axes[0,0], palette=seaborn.color_palette("pastel"))
	axes[0,0].set_xticklabels(axes[0].get_xticklabels())
	axes[0,0].set_title("mixture = 10, measurement = 20")
	handles, labels = axes[0,0].get_legend_handles_labels()
	axes[0,0].legend(handles=handles, labels=labels)
	axes[0,0].text(-1.2, 0.018, "A", fontsize=14, fontweight='bold')
	# mixture subplot 2
	df2 = None
	for k,v in result_mi_real.items():
		if k[0] != 10 or k[1] != 5000:
			continue
		tmp = pd.DataFrame( {"measurement size" : k[2], "abs diff between MI" : np.abs(v - result_mi_raw[k]), "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"measurement size" : k[2], "abs diff between MI" : np.abs(v - result_mi_corr[k]), "label" : "corrected"} )
		if df2 is None:
			df2 = tmp
			df2 = df2.append(tmp2)
		else:
			df2 = df2.append(tmp, ignore_index = True)
			df2 = df2.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df2, x="measurement size", y="abs diff between MI", hue="label", cut=0, ax=axes[0, 1], palette=seaborn.color_palette("pastel"))
	axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels())
	axes[0, 1].set_title("mixture = 10, sample = 5000")
	handles, labels = axes[0, 1].get_legend_handles_labels()
	axes[0, 1].legend(handles=handles, labels=labels)
	axes[0, 1].text(-1.2, 0.0145, "B", fontsize=14, fontweight='bold')
	# mixture subplot 3
	df3 = None
	for k,v in result_mi_real.items():
		if k[1] != 5000 or k[2] != 20:
			continue
		tmp = pd.DataFrame( {"number of mixtures" : k[0], "abs diff between MI" : np.abs(v - result_mi_raw[k]), "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"number of mixtures" : k[0], "abs diff between MI" : np.abs(v - result_mi_corr[k]), "label" : "corrected"} )
		if df3 is None:
			df3 = tmp
			df3 = df3.append(tmp2)
		else:
			df3 = df3.append(tmp, ignore_index = True)
			df3 = df3.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df3, x="number of mixtures", y="abs diff between MI", hue="label", cut=0, ax=axes[1, 0], palette=seaborn.color_palette("pastel"))
	axes[1, 0].set_xticklabels(axes[2].get_xticklabels())
	axes[1, 0].set_title("sample = 5000, measurement = 20")
	handles, labels = axes[1, 0].get_legend_handles_labels()
	axes[1, 0].legend(handles=handles, labels=labels)
	axes[1, 0].text(-1.2, 0.050, "C", fontsize=14, fontweight='bold')
	# independent
	df4 = None
	for k,v in result_inde_mi_real.items():
		tmp = pd.DataFrame( {"range of true MI" : "({}, {})".format(np.around(k[0]-0.1,2), np.around(k[0],2)), "abs diff between MI" : np.abs(v - result_inde_mi_raw[k]), "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"range of true MI" : "({}, {})".format(np.around(k[0]-0.1,2), np.around(k[0],2)), "abs diff between MI" : np.abs(v - result_inde_mi_corr[k]), "label" : "corrected"} )
		if df4 is None:
			df4 = tmp
			df4 = df4.append(tmp2)
		else:
			df4 = df4.append(tmp, ignore_index = True)
			df4 = df4.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df4, x="range of true MI", y="abs diff between MI", hue="label", cut=0, ax=axes[1, 1], palette=seaborn.color_palette("pastel"))
	axes[1, 1].set_xticklabels(ax[1, 1].get_xticklabels())
	axes[1, 1].set_title("simulation of weak independence")
	axes[1, 1].set(ylim=(0, 0.055))
	handles, labels = axes[1, 1].get_legend_handles_labels()
	axes[1, 1].legend(handles=handles, labels=labels)
	axes[1, 1].text(-1.2, 0.061, "D", fontsize=14, fontweight='bold')
	fig.subplots_adjust(hspace=0.4, wspace = 0.4)
	fig.savefig( script_dir + "/results/simulation_continuous_all.pdf", transparent = True, bbox_inches='tight')


if __name__ == "__main__":
	# get the code directory
	script_dir = sys.argv[0]
	strs = script_dir.split("/")
	if len(strs) > 2:
		script_dir = "/".join(strs[:-2])
	elif len(strs) == 2:
		script_dir = "./"
	else:
		script_dir = "../"

	if (not Path(script_dir + "/results/simulation_independence_real.pkl").exists()) or \
	   (not Path(script_dir + "/results/simulation_independence_raw.pkl").exists()) or \
	   (not Path(script_dir + "/results/simulation_independence_corr.pkl").exists()):
		run_1(script_dir)

	if (not Path(script_dir + "/results/simulation_double_mixture_real_05.pkl").exists()) or \
	   (not Path(script_dir + "/results/simulation_double_mixture_raw_05.pkl").exists()) or \
	   (not Path(script_dir + "/results/simulation_double_mixture_corr_05.pkl").exists()):
		run_3(script_dir)

	plot_combined(script_dir)