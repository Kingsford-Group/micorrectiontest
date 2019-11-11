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


nbin_x = 5
nbin_y = 5
boundary_x = list(range(1, nbin_x + 1))
boundary_y = list(range(1, nbin_y + 1))
mean_x = [x - 0.5 for x in boundary_x]
mean_y = [x - 0.5 for x in boundary_y]


def generate_multinomial_prob(nbin_x, nbin_y):
	prob = np.random.dirichlet(np.ones(nbin_x * nbin_y)).reshape( (nbin_x,nbin_y) )
	return prob


def generate_covariance(boundary_x, boundary_y):
	binsize_x = np.min([boundary_x[i+1] - boundary_x[i] for i in range(len(boundary_x) - 1)])
	binsize_y = np.min([boundary_y[i+1] - boundary_y[i] for i in range(len(boundary_y) - 1)])
	var_x = np.random.uniform(low=0.1*binsize_x, high=4*binsize_x, size=1)[0]
	var_y = np.random.uniform(low=0.1*binsize_y, high=4*binsize_y, size=1)[0]
	max_cov = np.sqrt(var_x * var_y)
	# cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
	cov = (np.random.uniform(low=-1, high=1, size=1)[0] > 0) * max_cov
	covariance = np.array([[var_x, cov], [cov, var_y]])
	return covariance


def generate_covariance_multi(boundary_x, boundary_y):
	default_binsize_x = np.min([boundary_x[i+1] - boundary_x[i] for i in range(len(boundary_x) - 1)])
	default_binsize_y = np.min([boundary_y[i+1] - boundary_y[i] for i in range(len(boundary_y) - 1)])
	multi_covariance = np.zeros( (len(boundary_x), len(boundary_y), 2, 2) )
	for idx_x in range(len(boundary_x)):
		for idx_y in range(len(boundary_y)):
			if idx_x > 0:
				binsize_x = boundary_x[idx_x] - boundary_x[idx_x - 1]
			else:
				binsize_x = default_binsize_x
			if idx_y > 0:
				binsize_y = boundary_x[idx_y] - boundary_y[idx_y - 1]
			else:
				binsize_y = default_binsize_y
			var_x = np.random.uniform(low=0.1*binsize_x, high=2*binsize_x, size=1)[0]
			var_y = np.random.uniform(low=0.1*binsize_y, high=2*binsize_y, size=1)[0]
			max_cov = np.sqrt(var_x * var_y)
			cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
			multi_covariance[idx_x, idx_y, :, :] = np.array([[var_x, cov], [cov, var_y]])
	return multi_covariance


def generate_covariance_multi_two(boundary_x, boundary_y):
	default_binsize_x = np.min([boundary_x[i+1] - boundary_x[i] for i in range(len(boundary_x) - 1)])
	default_binsize_y = np.min([boundary_y[i+1] - boundary_y[i] for i in range(len(boundary_y) - 1)])
	multi_covariance1 = np.zeros( (len(boundary_x), len(boundary_y), 2, 2) )
	multi_covariance2 = np.zeros( (len(boundary_x), len(boundary_y), 2, 2) )
	for idx_x in range(len(boundary_x)):
		for idx_y in range(len(boundary_y)):
			if idx_x > 0:
				binsize_x = boundary_x[idx_x] - boundary_x[idx_x - 1]
			else:
				binsize_x = default_binsize_x
			if idx_y > 0:
				binsize_y = boundary_x[idx_y] - boundary_y[idx_y - 1]
			else:
				binsize_y = default_binsize_y
			# for multi_covariance1
			var_x = np.random.uniform(low=0.1*binsize_x, high=2*binsize_x, size=1)[0]
			var_y = np.random.uniform(low=0.1*binsize_y, high=2*binsize_y, size=1)[0]
			max_cov = np.sqrt(var_x * var_y)
			cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
			multi_covariance1[idx_x, idx_y, :, :] = np.array([[var_x, cov], [cov, var_y]])
			# for multi_covariance2
			var_x = np.random.uniform(low=0.1*binsize_x, high=2*binsize_x, size=1)[0]
			var_y = np.random.uniform(low=0.1*binsize_y, high=2*binsize_y, size=1)[0]
			max_cov = np.sqrt(var_x * var_y)
			cov = np.random.uniform(low=-max_cov, high=max_cov, size=1)[0]
			multi_covariance2[idx_x, idx_y, :, :] = np.array([[var_x, cov], [cov, var_y]])
	# weights
	error_weights = np.random.dirichlet(np.ones(2))
	return multi_covariance1, multi_covariance2, error_weights


def sample_observation(prob, covariance, n_sample=50, n_bootstrap=5):
	'''
		output: real-valued points of dimension n_bootstrap * n_sample * 2
		The observation are real valued.
		First sample which grid each observation is located; then sample the hidden true real-valued (x,y) uniformly from the boundaries in that grid; 
		then sample measurement error from covariance matrix
	'''
	# sample from the grid
	grid_hidden = np.random.multinomial(1, prob.flatten(), size=n_sample)
	# sample the true real-valued
	mean_hidden = np.zeros( (n_sample, 2) )
	for i in range(n_sample):
		idx = np.where(grid_hidden[i,:] > 0)[0]
		assert(len(idx) == 1)
		idx = idx[0]
		idx_x = int(idx / nbin_x)
		idx_y = idx - nbin_x * idx_x
		mean_hidden[i, :] = np.array([mean_x[idx_x], mean_y[idx_y]])
	real_hidden = np.random.uniform(low=-0.5, high=0.5, size=(n_sample, 2)) + mean_hidden
	real_obs = np.zeros( (n_bootstrap, n_sample, 2) )
	# sample the measurement error for n_bootstrap times
	real_error = np.random.multivariate_normal(mean=np.zeros(2), cov=covariance, size=(n_bootstrap, n_sample))
	real_obs = real_hidden + real_error
	return real_obs


def sample_observation_multi(prob, multi_covariance, boundary_x, boundary_y, n_sample=50, n_bootstrap=5):
	'''
		output: real-valued points of dimension n_bootstrap * n_sample * 2
		The observation are real valued.
		First sample which grid each observation is located; then sample the hidden true real-valued (x,y) uniformly from the boundaries in that grid; 
		then sample measurement error from covariance matrix of the corresponding (x_bin, y_bin)
	'''
	assert(multi_covariance.shape == (prob.shape[0], prob.shape[1], 2, 2))
	# sample from the grid
	grid_hidden = np.random.multinomial(1, prob.flatten(), size=n_sample)
	# sample the true real-valued
	sample_indexes = {}
	mean_hidden = np.zeros( (n_sample, 2) )
	for i in range(n_sample):
		idx = np.where(grid_hidden[i,:] > 0)[0]
		assert(len(idx) == 1)
		idx = idx[0]
		idx_x = int(idx / nbin_x)
		idx_y = idx - nbin_x * idx_x
		if (idx_x, idx_y) in sample_indexes:
			sample_indexes[(idx_x, idx_y)].append( i )
		else:
			sample_indexes[(idx_x, idx_y)] = [i]
		mean_hidden[i, :] = np.array([mean_x[idx_x], mean_y[idx_y]])
	real_hidden = np.random.uniform(low=-0.5, high=0.5, size=(n_sample, 2)) + mean_hidden
	real_error = np.zeros( (n_bootstrap, n_sample, 2) )
	# simulate error for each (x_bin, y_bin)
	for idx_x in range(len(boundary_x)):
		for idx_y in range(len(boundary_y)):
			if (idx_x, idx_y) in sample_indexes:
				indexes = np.array(sample_indexes[(idx_x, idx_y)])
				tmp = np.random.multivariate_normal(mean=np.zeros(2), cov=multi_covariance[idx_x,idx_y,:,:], size=(n_bootstrap, len(indexes)))
				real_error[:, indexes] = tmp
	real_obs = real_hidden + real_error
	return real_obs


def sample_observation_multi_two(prob, multi_covariance1, multi_covariance2, error_weights, boundary_x, boundary_y, n_sample=50, n_bootstrap=5):
	'''
		output: real-valued points of dimension n_bootstrap * n_sample * 2
		The observation are real valued.
		First sample which grid each observation is located; then sample the hidden true real-valued (x,y) uniformly from the boundaries in that grid; 
		then sample measurement error from covariance matrix of the corresponding (x_bin, y_bin)
	'''
	assert(multi_covariance1.shape == (prob.shape[0], prob.shape[1], 2, 2))
	assert(multi_covariance2.shape == (prob.shape[0], prob.shape[1], 2, 2))
	# sample from the grid
	grid_hidden = np.random.multinomial(1, prob.flatten(), size=n_sample)
	# sample the true real-valued
	sample_indexes = {}
	mean_hidden = np.zeros( (n_sample, 2) )
	for i in range(n_sample):
		idx = np.where(grid_hidden[i,:] > 0)[0]
		assert(len(idx) == 1)
		idx = idx[0]
		idx_x = int(idx / nbin_x)
		idx_y = idx - nbin_x * idx_x
		if (idx_x, idx_y) in sample_indexes:
			sample_indexes[(idx_x, idx_y)].append( i )
		else:
			sample_indexes[(idx_x, idx_y)] = [i]
		mean_hidden[i, :] = np.array([mean_x[idx_x], mean_y[idx_y]])
	real_hidden = np.random.uniform(low=-0.5, high=0.5, size=(n_sample, 2)) + mean_hidden
	real_error = np.zeros( (n_bootstrap, n_sample, 2) )
	# simulate error for each (x_bin, y_bin)
	for idx_x in range(len(boundary_x)):
		for idx_y in range(len(boundary_y)):
			if (idx_x, idx_y) in sample_indexes:
				indexes = np.array(sample_indexes[(idx_x, idx_y)])
				# for multi_covariance1
				this_n_bootstrap = int(np.round(n_bootstrap * error_weights[0]))
				tmp1 = np.random.multivariate_normal(mean=np.zeros(2), cov=multi_covariance1[idx_x,idx_y,:,:], size=(this_n_bootstrap, len(indexes)))
				# for multi_covariance2
				this_n_bootstrap = int(np.round(n_bootstrap * error_weights[1]))
				tmp2 = np.random.multivariate_normal(mean=np.zeros(2), cov=multi_covariance2[idx_x,idx_y,:,:], size=(this_n_bootstrap, len(indexes)))
				real_error[:, indexes, :] = np.concatenate( (tmp1, tmp2), axis=0 )
	real_obs = real_hidden + real_error
	return real_obs


def Count_discretize(obs, boundary_x, boundary_y):
	'''
		obs of dimension n_sample * 2
	'''
	counts = np.zeros( (nbin_x, nbin_y) )
	for i in range(obs.shape[0]):
		idx_x = np.where(obs[i, 0] < boundary_x)[0]
		idx_y = np.where(obs[i, 1] < boundary_y)[0]
		if len(idx_x) != 0:
			idx_x = idx_x[0]
		else:
			idx_x = len(boundary_x) - 1
		if len(idx_y) != 0:
			idx_y = idx_y[0]
		else:
			idx_y = len(boundary_y) - 1
		counts[idx_x, idx_y] += 1
	return counts


def run_uni():
	sample_sizes = [100, 500, 1000, 10000, 100000]
	bootstrap_sizes = [5,10,20,50,100]
	n_simulation = 100
	# initialize comparison result
	result_t_o = {}
	result_tf_o = {}
	result_t_of = {}
	result_mi_t_o = {}
	result_mi_t_of = {}
	# initialize result vector
	for n_sample in sample_sizes:
		for n_bootstrap in bootstrap_sizes:
			result_t_o[(n_sample, n_bootstrap)] = []
			result_tf_o[(n_sample, n_bootstrap)] = []
			result_t_of[(n_sample, n_bootstrap)] = []
			result_mi_t_o[(n_sample, n_bootstrap)] = []
			result_mi_t_of[(n_sample, n_bootstrap)] = []
	# start simulation
	np.random.seed(0)
	with tqdm.tqdm(total=n_simulation) as pbar:
		for i in range(n_simulation):
			prob = generate_multinomial_prob(nbin_x, nbin_y)
			covariance = generate_covariance(boundary_x, boundary_y)
			# true transition matrix
			# F = TransitionMatrix(covariance, boundary_x, boundary_y, mean_x, mean_y)
			for n_sample in sample_sizes:
				for n_bootstrap in bootstrap_sizes:
					# sample observations
					real_obs = sample_observation(prob, covariance, n_sample, n_bootstrap)
					# estimated transition matrix using the given number of bootstrap
					covariance_est =  BootstrapCovariance(real_obs[:, :, 0], real_obs[:, :, 1])
					F_est = TransitionMatrix(covariance_est, boundary_x, boundary_y, mean_x, mean_y)
					# count of mean bootstrap observations
					counts = Count_discretize(np.mean(real_obs, axis=0), boundary_x, boundary_y)
					# distance between count matrix and probability matrix
					# distance between variations of prob and variations of counts
					diffcount_t_o =  np.sum(np.abs(counts/np.sum(counts) - prob))
					# convoluted prob
					prob_conv = np.matmul(prob.flatten(), F_est).reshape( (nbin_x, nbin_y) )
					diffcount_tf_o = np.sum(np.abs(counts/np.sum(counts) - prob_conv))
					# inverse observation
					counts_corr = np.matmul(counts.flatten(), np.linalg.inv(F_est)).reshape((nbin_x, nbin_y))
					diffcount_t_of =  np.sum(np.abs(counts_corr / np.sum(counts_corr) - prob))
					# distance between calculated mutual information and truth
					# MI truth
					mi_truth = MI_bincount(prob)
					# MI calculated by raw count
					mi_o = MI_bincount(counts)
					# MI calculated by adjusted count
					counts_corr[np.where(counts_corr < 0)] = 0
					mi_of = MI_bincount(counts_corr)
					# append result vector
					result_t_o[(n_sample, n_bootstrap)].append( diffcount_t_o )
					result_tf_o[(n_sample, n_bootstrap)].append( diffcount_tf_o )
					result_t_of[(n_sample, n_bootstrap)].append( diffcount_t_of )
					result_mi_t_o[(n_sample, n_bootstrap)].append( np.abs(mi_truth - mi_o) )
					result_mi_t_of[(n_sample, n_bootstrap)].append( np.abs(mi_truth - mi_of) )
			pbar.update(1)
	# save simulation result
	pickle.dump( result_t_o, open( script_dir + "/results/simulation_half_t_o.pkl", "wb" ) )
	pickle.dump( result_tf_o, open( script_dir + "/results/simulation_half_tf_o.pkl", "wb" ) )
	pickle.dump( result_t_of, open( script_dir + "/results/simulation_half_t_of.pkl", "wb" ) )
	pickle.dump( result_mi_t_o, open( script_dir + "/results/simulation_half_mi_t_o.pkl", "wb" ) )
	pickle.dump( result_mi_t_of, open( script_dir + "/results/simulation_half_mi_t_of.pkl", "wb" ) )
	# draw figures
	# for bootstrap 10
	df2 = None
	for k,v in result_t_o.items():
		if k[1] != 10:
			continue
		tmp = pd.DataFrame( {"sample_sizes" : k[0], "diff_pmf" : v, "label" : "truth - observed"} )
		tmp2 = pd.DataFrame( {"sample_sizes" : k[0], "diff_pmf" : result_t_of[k], "label" : "truth - observed*inv(F)"} )
		tmp3 = pd.DataFrame({"sample_sizes" : k[0], "diff_pmf" : result_tf_o[k], "label" : "truth*F - observed"} )
		if df2 is None:
			df2 = tmp
			df2 = df2.append(tmp2)
			df2 = df2.append(tmp3)
		else:
			df2 = df2.append(tmp, ignore_index = True)
			df2 = df2.append(tmp2, ignore_index = True)
			df2 = df2.append(tmp3, ignore_index = True)
	fig = plt.figure()
	ax = plt.axes()
	ax = seaborn.violinplot(data=df2, x="sample_sizes", y="diff_pmf", hue="label")
	ax.set_xticklabels(ax.get_xticklabels())
	ax.set_title("l2 distance between pmf and count matrix (bootstrap = 10)")
	fig.savefig(script_dir + "/results/simulation_half_diffpmf_10.pdf", transparent = True)
	# for bootstrap 20
	df3 = None
	for k,v in result_t_o.items():
		if k[1] != 20:
			continue
		tmp = pd.DataFrame( {"sample_sizes" : k[0], "diff_pmf" : v, "label" : "truth - observed"} )
		tmp2 = pd.DataFrame( {"sample_sizes" : k[0], "diff_pmf" : result_t_of[k], "label" : "truth - observed*inv(F)"} )
		tmp3 = pd.DataFrame({"sample_sizes" : k[0], "diff_pmf" : result_tf_o[k], "label" : "truth*F - observed"} )
		if df3 is None:
			df3 = tmp
			df3 = df3.append(tmp2)
			df3 = df3.append(tmp3)
		else:
			df3 = df3.append(tmp, ignore_index = True)
			df3 = df3.append(tmp2, ignore_index = True)
			df3 = df3.append(tmp3, ignore_index = True)
	fig = plt.figure()
	ax = plt.axes()
	ax = seaborn.violinplot(data=df3, x="sample_sizes", y="diff_pmf", hue="label")
	ax.set_xticklabels(ax.get_xticklabels())
	ax.set_title("l2 distance between pmf and count matrix (bootstrap = 20)")
	fig.savefig(script_dir + "/results/simulation_half_diffpmf_20.pdf", transparent = True)
	# for bootstrap 50
	df4 = None
	for k,v in result_t_o.items():
		if k[1] != 50:
			continue
		tmp = pd.DataFrame( {"sample_sizes" : k[0], "diff_pmf" : v, "label" : "truth - observed"} )
		tmp2 = pd.DataFrame( {"sample_sizes" : k[0], "diff_pmf" : result_t_of[k], "label" : "truth - observed*inv(F)"} )
		tmp3 = pd.DataFrame({"sample_sizes" : k[0], "diff_pmf" : result_tf_o[k], "label" : "truth*F - observed"} )
		if df4 is None:
			df4 = tmp
			df4 = df4.append(tmp2)
			df4 = df4.append(tmp3)
		else:
			df4 = df4.append(tmp, ignore_index = True)
			df4 = df4.append(tmp2, ignore_index = True)
			df4 = df4.append(tmp3, ignore_index = True)
	fig = plt.figure()
	ax = plt.axes()
	ax = seaborn.violinplot(data=df4, x="sample_sizes", y="diff_pmf", hue="label")
	ax.set_xticklabels(ax.get_xticklabels())
	ax.set_title("l2 distance between pmf and count matrix (bootstrap = 50)")
	fig.savefig(script_dir + "/results/simulation_half_diffpmf_50.pdf", transparent = True)
	# MI: with given bootstrap size (small), deviation from the truth increases as sample size increases (more and more converge to the wrong MI)
	df1 = None
	for k,v in result_mi_t_o.items():
		if k[1] != 10:
			continue
		tmp = pd.DataFrame( {"sample_sizes" : k[0], "diff_MI" : v, "label" : "truth - MI(observed)"} )
		tmp2 = pd.DataFrame( {"sample_sizes" : k[0], "diff_MI" : result_mi_t_of[k], "label" : "truth - MI(observed*inv(F))"} )
		if df1 is None:
			df1 = tmp
			df1 = df1.append(tmp2)
		else:
			df1 = df1.append(tmp, ignore_index = True)
			df1 = df1.append(tmp2, ignore_index = True)
	fig = plt.figure()
	ax = plt.axes()
	ax = seaborn.violinplot(data=df1, x="sample_sizes", y="diff_MI", hue="label")
	ax.set_xticklabels(ax.get_xticklabels())
	ax.set_title("Comparing MI correction on various sample sizes  (bootstrap = 10)")
	fig.savefig(script_dir + "/results/simulation_half_diffMI_10.pdf", transparent = True)
	# MI: with given sample size, deviation from the truth decreases as bootstrap size increases
	df3 = None
	for k,v in result_mi_t_o.items():
		if k[0] != 1000:
			continue
		tmp = pd.DataFrame( {"bootstrap_sizes" : k[1], "diff_MI" : v, "label" : "truth - MI(observed)"} )
		tmp2 = pd.DataFrame( {"bootstrap_sizes" : k[1], "diff_MI" : result_mi_t_of[k], "label" : "truth - MI(observed*inv(F))"} )
		if df3 is None:
			df3 = tmp
			df3 = df3.append(tmp2)
		else:
			df3 = df3.append(tmp, ignore_index = True)
			df3 = df3.append(tmp2, ignore_index = True)
	fig = plt.figure()
	ax = plt.axes()
	ax = seaborn.violinplot(data=df3, x="bootstrap_sizes", y="diff_MI", hue="label")
	ax.set_xticklabels(ax.get_xticklabels())
	ax.set_title("Comparing correction on various bootstrap sizes (sample size = 1000)")
	fig.savefig(script_dir + "/results/simulation_half_diffMI_1000.pdf", transparent = True)


def run_2():
	# multi covariance
	sample_sizes = [100, 500, 1000, 10000, 100000]
	bootstrap_sizes = [5,10,20,50,100]
	n_simulation = 100
	# initialize comparison result
	result_t_o = {}
	result_tf_o = {}
	result_t_of = {}
	result_mi_t_o = {}
	result_mi_t_of = {}
	# initialize result vector
	for n_sample in sample_sizes:
		for n_bootstrap in bootstrap_sizes:
			result_t_o[(n_sample, n_bootstrap)] = []
			result_tf_o[(n_sample, n_bootstrap)] = []
			result_t_of[(n_sample, n_bootstrap)] = []
			result_mi_t_o[(n_sample, n_bootstrap)] = []
			result_mi_t_of[(n_sample, n_bootstrap)] = []

	# start simulation
	np.random.seed(0)
	with tqdm.tqdm(total=n_simulation) as pbar:
		for i in range(n_simulation):
			prob = generate_multinomial_prob(nbin_x, nbin_y)
			multi_covariance = generate_covariance_multi(boundary_x, boundary_y)
			for n_sample in sample_sizes:
				for n_bootstrap in bootstrap_sizes:
					# sample observations
					real_obs = sample_observation_multi(prob, multi_covariance, boundary_x, boundary_y, n_sample, n_bootstrap)
					# estimated transition matrix using the given number of bootstrap
					multi_covariance_est =  BootstrapCovariance_multicov(real_obs[:, :, 0], real_obs[:, :, 1], boundary_x, boundary_y)
					F_est = TransitionMatrix_multicov(multi_covariance_est, boundary_x, boundary_y, mean_x, mean_y)
					# count of mean bootstrap observations
					counts = Count_discretize(np.mean(real_obs, axis=0), boundary_x, boundary_y)
					diffcount_t_o =  np.sum(np.abs(counts/np.sum(counts) - prob))
					# convoluted prob
					prob_conv = np.matmul(prob.flatten(), F_est).reshape( (nbin_x, nbin_y) )
					diffcount_tf_o = np.sum(np.abs(counts/np.sum(counts) - prob_conv))
					# inverse observation
					counts_corr = np.matmul(counts.flatten(), np.linalg.inv(F_est)).reshape((nbin_x, nbin_y))
					diffcount_t_of =  np.sum(np.abs(counts_corr / np.sum(counts_corr) - prob))
					# distance between calculated mutual information and truth
					# MI truth
					mi_truth = MI_bincount(prob)
					# MI calculated by raw count
					mi_o = MI_bincount(counts)
					# MI calculated by adjusted count
					counts_corr[np.where(counts_corr < 0)] = 0
					mi_of = MI_bincount(counts_corr)
					# append result vector
					result_t_o[(n_sample, n_bootstrap)].append( diffcount_t_o )
					result_tf_o[(n_sample, n_bootstrap)].append( diffcount_tf_o )
					result_t_of[(n_sample, n_bootstrap)].append( diffcount_t_of )
					result_mi_t_o[(n_sample, n_bootstrap)].append( np.abs(mi_truth - mi_o) )
					result_mi_t_of[(n_sample, n_bootstrap)].append( np.abs(mi_truth - mi_of) )
			pbar.update(1)

	pickle.dump( result_t_o, open( script_dir + "/results/simulation_half_multi_t_o.pkl", "wb" ) )
	pickle.dump( result_tf_o, open( script_dir + "/results/simulation_half_multi_tf_o.pkl", "wb" ) )
	pickle.dump( result_t_of, open( script_dir + "/results/simulation_half_multi_t_of.pkl", "wb" ) )
	pickle.dump( result_mi_t_o, open( script_dir + "/results/simulation_half_multi_mi_t_o.pkl", "wb" ) )
	pickle.dump( result_mi_t_of, open( script_dir + "/results/simulation_half_multi_mi_t_of.pkl", "wb" ) )

	fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
	df1 = None
	for k,v in result_t_o.items():
		if k[1] != 20:
			continue
		tmp = pd.DataFrame( {"sample size" : k[0], "l1 distance betweem pmf" : v, "label" : "truth - observed"} )
		tmp2 = pd.DataFrame( {"sample size" : k[0], "l1 distance betweem pmf" : result_t_of[k], "label" : "truth - observed*inv(F)"} )
		tmp3 = pd.DataFrame({"sample size" : k[0], "l1 distance betweem pmf" : result_tf_o[k], "label" : "truth*F - observed"} )
		if df1 is None:
			df1 = tmp
			df1 = df1.append(tmp2)
			df1 = df1.append(tmp3)
		else:
			df1 = df1.append(tmp, ignore_index = True)
			df1 = df1.append(tmp2, ignore_index = True)
			df1 = df1.append(tmp3, ignore_index = True)
	seaborn.violinplot(data=df1, x="sample size", y="l1 distance betweem pmf", hue="label", ax=axes[0, 0], palette=seaborn.color_palette("Set3"))
	axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels())
	axes[0, 0].set_title("PMF accuracy with varying sample size (bootstrap = 20)")
	axes[0, 0].text(-1, 0.98, "A", fontsize=14, fontweight='bold')

	df2 = None
	for k,v in result_t_o.items():
		if k[0] != 1000:
			continue
		tmp = pd.DataFrame( {"bootstrap size" : k[1], "l1 distance betweem pmf" : v, "label" : "truth - observed"} )
		tmp2 = pd.DataFrame( {"bootstrap size" : k[1], "l1 distance betweem pmf" : result_t_of[k], "label" : "truth - observed*inv(F)"} )
		tmp3 = pd.DataFrame({"bootstrap size" : k[1], "l1 distance betweem pmf" : result_tf_o[k], "label" : "truth*F - observed"} )
		if df2 is None:
			df2 = tmp
			df2 = df2.append(tmp2)
			df2 = df2.append(tmp3)
		else:
			df2 = df2.append(tmp, ignore_index = True)
			df2 = df2.append(tmp2, ignore_index = True)
			df2 = df2.append(tmp3, ignore_index = True)
	seaborn.violinplot(data=df2, x="bootstrap size", y="l1 distance betweem pmf", hue="label", ax=axes[0, 1], palette=seaborn.color_palette("Set3"))
	axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels())
	axes[0, 1].set_title("PMF accuracy with varying bootstrap size (sample = 1000)")
	axes[0, 1].text(-1, 1.25, "B", fontsize=14, fontweight='bold')

	df3 = None
	for k,v in result_mi_t_o.items():
		if k[1] != 20:
			continue
		tmp = pd.DataFrame( {"sample size" : k[0], "absolute difference between MI" : v, "label" : "truth - MI(observed)"} )
		tmp2 = pd.DataFrame( {"sample size" : k[0], "absolute difference between MI" : result_mi_t_of[k], "label" : "truth - MI(observed*inv(F))"} )
		if df3 is None:
			df3 = tmp
			df3 = df3.append(tmp2)
		else:
			df3 = df3.append(tmp, ignore_index = True)
			df3 = df3.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df3, x="sample size", y="absolute difference between MI", hue="label", ax=axes[1, 0], palette=seaborn.color_palette("pastel"))
	axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels())
	axes[1, 0].set_title("MI accuracy with varying sample sizes  (bootstrap = 20)")
	axes[1, 0].text(-1, 1.25, "C", fontsize=14, fontweight='bold')

	df4 = None
	for k,v in result_mi_t_o.items():
		if k[0] != 1000:
			continue
		tmp = pd.DataFrame( {"bootstrap size" : k[1], "absolute difference between MI" : v, "label" : "truth - MI(observed)"} )
		tmp2 = pd.DataFrame( {"bootstrap size" : k[1], "absolute difference between MI" : result_mi_t_of[k], "label" : "truth - MI(observed*inv(F))"} )
		if df4 is None:
			df4 = tmp
			df4 = df4.append(tmp2)
		else:
			df4 = df4.append(tmp, ignore_index = True)
			df4 = df4.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df4, x="bootstrap size", y="absolute difference between MI", hue="label", ax=axes[1, 1], palette=seaborn.color_palette("pastel"))
	axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels())
	axes[1, 1].set_title("MI accuracy with varying bootstrap sizes  (sample = 1000)")
	axes[1, 1].text(-1, 1.45, "D", fontsize=14, fontweight='bold')

	fig.savefig(script_dir + "/results/simulation_half_all.pdf", transparent = True, bbox_inches='tight')


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

	# multi covariance
	sample_sizes = [100, 500, 1000, 10000, 100000]
	bootstrap_sizes = [10,20,50,100]
	n_simulation = 100
	# initialize comparison result
	result_t_o = {}
	result_tf_o = {}
	result_t_of = {}
	result_mi_t_o = {}
	result_mi_t_of = {}
	# initialize result vector
	for n_sample in sample_sizes:
		for n_bootstrap in bootstrap_sizes:
			result_t_o[(n_sample, n_bootstrap)] = []
			result_tf_o[(n_sample, n_bootstrap)] = []
			result_t_of[(n_sample, n_bootstrap)] = []
			result_mi_t_o[(n_sample, n_bootstrap)] = []
			result_mi_t_of[(n_sample, n_bootstrap)] = []

	# start simulation
	np.random.seed(0)
	with tqdm.tqdm(total=n_simulation) as pbar:
		for i in range(n_simulation):
			prob = generate_multinomial_prob(nbin_x, nbin_y)
			multi_covariance1, multi_covariance2, error_weights = generate_covariance_multi_two(boundary_x, boundary_y)
			for n_sample in sample_sizes:
				for n_bootstrap in bootstrap_sizes:
					# sample observations
					real_obs = sample_observation_multi_two(prob, multi_covariance1, multi_covariance2, error_weights, boundary_x, boundary_y, n_sample, n_bootstrap)
					# estimated transition matrix using the given number of bootstrap
					multi_covariance_est =  BootstrapCovariance_multicov(real_obs[:, :, 0], real_obs[:, :, 1], boundary_x, boundary_y)
					F_est = TransitionMatrix_multicov(multi_covariance_est, boundary_x, boundary_y, mean_x, mean_y)
					# count of mean bootstrap observations
					counts = Count_discretize(np.mean(real_obs, axis=0), boundary_x, boundary_y)
					diffcount_t_o =  np.sum(np.abs(counts/np.sum(counts) - prob))
					# convoluted prob
					prob_conv = np.matmul(prob.flatten(), F_est).reshape( (nbin_x, nbin_y) )
					diffcount_tf_o = np.sum(np.abs(counts/np.sum(counts) - prob_conv))
					# inverse observation
					counts_corr = np.matmul(counts.flatten(), np.linalg.inv(F_est)).reshape((nbin_x, nbin_y))
					diffcount_t_of =  np.sum(np.abs(counts_corr / np.sum(counts_corr) - prob))
					# distance between calculated mutual information and truth
					# MI truth
					mi_truth = MI_bincount(prob)
					# MI calculated by raw count
					mi_o = MI_bincount(counts)
					# MI calculated by adjusted count
					counts_corr[np.where(counts_corr < 0)] = 0
					mi_of = MI_bincount(counts_corr)
					# append result vector
					result_t_o[(n_sample, n_bootstrap)].append( diffcount_t_o )
					result_tf_o[(n_sample, n_bootstrap)].append( diffcount_tf_o )
					result_t_of[(n_sample, n_bootstrap)].append( diffcount_t_of )
					result_mi_t_o[(n_sample, n_bootstrap)].append( np.abs(mi_truth - mi_o) )
					result_mi_t_of[(n_sample, n_bootstrap)].append( np.abs(mi_truth - mi_of) )
			pbar.update(1)

	pickle.dump( result_t_o, open( script_dir + "/results/simulation_half_multidouble_t_o.pkl", "wb" ) )
	pickle.dump( result_tf_o, open( script_dir + "/results/simulation_half_multidouble_tf_o.pkl", "wb" ) )
	pickle.dump( result_t_of, open( script_dir + "/results/simulation_half_multidouble_t_of.pkl", "wb" ) )
	pickle.dump( result_mi_t_o, open( script_dir + "/results/simulation_half_multidouble_mi_t_o.pkl", "wb" ) )
	pickle.dump( result_mi_t_of, open( script_dir + "/results/simulation_half_multidouble_mi_t_of.pkl", "wb" ) )

	# plt.rcParams.update({'font.size': 12})
	fig, axes = plt.subplots(1, 2, figsize=(8, 3.25))
	df1 = None
	for k,v in result_t_o.items():
		if k[1] != 20:
			continue
		tmp = pd.DataFrame( {"sample size" : k[0], "l1 distance betweem PMF" : v, "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"sample size" : k[0], "l1 distance betweem PMF" : result_t_of[k], "label" : "corrected"} )
		if df1 is None:
			df1 = tmp
			df1 = df1.append(tmp2)
		else:
			df1 = df1.append(tmp, ignore_index = True)
			df1 = df1.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df1, x="sample size", y="l1 distance betweem PMF", hue="label", ax=axes[0], palette=seaborn.color_palette("Set3"))
	axes[0].set_xticklabels(axes[0].get_xticklabels())
	axes[0].set_title("PMF accuracy (bootstrap = 20)")
	handles, labels = axes[0].get_legend_handles_labels()
	axes[0].legend(handles=handles, labels=labels)
	axes[0].text(-1, 0.78, "A", fontsize=14, fontweight='bold')

	df2 = None
	for k,v in result_t_o.items():
		if k[0] != 1000:
			continue
		tmp = pd.DataFrame( {"bootstrap size" : k[1], "l1 distance betweem PMF" : v, "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"bootstrap size" : k[1], "l1 distance betweem PMF" : result_t_of[k], "label" : "corrected"} )
		if df2 is None:
			df2 = tmp
			df2 = df2.append(tmp2)
		else:
			df2 = df2.append(tmp, ignore_index = True)
			df2 = df2.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df2, x="bootstrap size", y="l1 distance betweem PMF", hue="label", ax=axes[1], palette=seaborn.color_palette("Set3"))
	axes[1].set_xticklabels(axes[1].get_xticklabels())
	axes[1].set_title("PMF accuracy (sample = 1000)")
	handles, labels = axes[1].get_legend_handles_labels()
	axes[1].legend(handles=handles, labels=labels)
	axes[1].text(-1, 0.55, "B", fontsize=14, fontweight='bold')

	fig.subplots_adjust(bottom=0.15, wspace = 0.35)
	fig.savefig(script_dir + "/results/simulation_half_double_pmf.pdf", transparent = True, bbox_inches='tight')


	fig, axes = plt.subplots(1, 2, figsize=(8, 3.25))
	df3 = None
	for k,v in result_mi_t_o.items():
		if k[1] != 20:
			continue
		tmp = pd.DataFrame( {"sample size" : k[0], "absolute difference between MI" : v, "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"sample size" : k[0], "absolute difference between MI" : result_mi_t_of[k], "label" : "corrected"} )
		if df3 is None:
			df3 = tmp
			df3 = df3.append(tmp2)
		else:
			df3 = df3.append(tmp, ignore_index = True)
			df3 = df3.append(tmp2, ignore_index = True)


	seaborn.violinplot(data=df3, x="sample size", y="absolute difference between MI", hue="label", ax=axes[0], palette=seaborn.color_palette("pastel"))
	axes[0].set_xticklabels(axes[0].get_xticklabels())
	axes[0].set_title("MI accuracy (bootstrap = 20)")
	handles, labels = axes[0].get_legend_handles_labels()
	axes[0].legend(handles=handles, labels=labels)
	axes[0].text(-1, 0.91, "A", fontsize=14, fontweight='bold')

	df4 = None
	for k,v in result_mi_t_o.items():
		if k[0] != 1000:
			continue
		tmp = pd.DataFrame( {"bootstrap size" : k[1], "absolute difference between MI" : v, "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"bootstrap size" : k[1], "absolute difference between MI" : result_mi_t_of[k], "label" : "corrected"} )
		if df4 is None:
			df4 = tmp
			df4 = df4.append(tmp2)
		else:
			df4 = df4.append(tmp, ignore_index = True)
			df4 = df4.append(tmp2, ignore_index = True)


	seaborn.violinplot(data=df4, x="bootstrap size", y="absolute difference between MI", hue="label", ax=axes[1], palette=seaborn.color_palette("pastel"))
	axes[1].set_xticklabels(axes[1].get_xticklabels())
	axes[1].set_title("MI accuracy (sample = 1000)")
	handles, labels = axes[1].get_legend_handles_labels()
	axes[1].legend(handles=handles, labels=labels)
	axes[1].text(-1, 0.93, "B", fontsize=14, fontweight='bold')

	fig.subplots_adjust(bottom=0.15, wspace = 0.35)
	fig.savefig(script_dir + "/results/simulation_half_double_mi.pdf", transparent = True, bbox_inches='tight')
