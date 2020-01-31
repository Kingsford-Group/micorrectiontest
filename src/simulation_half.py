#!/bin/python

import sys
import numpy as np
from utils import *
from scipy.stats import mvn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import tqdm
import copy
from cvxopt import matrix, solvers
import pickle
plt.rcParams.update({'font.size': 13})


nbin_x = 5
nbin_y = 5
boundary_x = list(range(1, nbin_x + 1))
boundary_y = list(range(1, nbin_y + 1))
mean_x = [x - 0.5 for x in boundary_x]
mean_y = [x - 0.5 for x in boundary_y]


def generate_multinomial_prob(nbin_x, nbin_y):
	prob = np.random.dirichlet(np.ones(nbin_x * nbin_y)).reshape( (nbin_x,nbin_y) )
	return prob


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


def adjust_transition(counts, F_est):
	counts_corr = np.matmul(counts.flatten(), np.linalg.inv(F_est))
	index_avoid = []
	multiplier = np.ones(F_est.shape[0]) * 0.99
	while np.sum(counts_corr[np.where(counts_corr < 0)]) < -1:
		index_avoid = []
		for i in np.where(counts_corr < 0)[0]:
			F_est[i,:] *= multiplier
			F_est[:,i] *= multiplier
		# index_avoid and index_retain is used for different ways of normalization
		index_avoid += list(np.where(counts_corr < 0)[0])
		index_avoid = set(index_avoid)
		index_retain = np.array([i for i in range(F_est.shape[0]) if not (i in index_avoid)])
		index_avoid = np.array(list(index_avoid))
		for i in index_avoid:
			F_est[i,i] += 1 - np.sum(F_est[i,:])
		for i in index_retain:
			F_est[i,index_retain] += (1 - np.sum(F_est[i,:])) * F_est[i, index_retain] / np.sum(F_est[i, index_retain])
			assert( np.abs(np.sum(F_est[i,:]) - 1) < 1e-4 )
		counts_corr = np.matmul(counts.flatten(), np.linalg.inv(F_est))
	return counts_corr, F_est


def plot_combined(codedir):
	result_t_o = pickle.load( open( codedir + "../results/ismb_simulation_half_multidouble_t_o.pkl", "rb" ) )
	result_tf_o = pickle.load( open( codedir + "../results/ismb_simulation_half_multidouble_tf_o.pkl", "rb" ) )
	result_t_of = pickle.load( open( codedir + "../results/ismb_simulation_half_multidouble_t_of.pkl", "rb" ) )
	result_mi_t = pickle.load( open( codedir + "../results/ismb_simulation_half_multidouble_mi_truth.pkl", "rb" ) )
	result_mi_o = pickle.load( open( codedir + "../results/ismb_simulation_half_multidouble_mi_obs.pkl", "rb" ) )
	result_mi_corr = pickle.load( open(codedir + "../results/ismb_simulation_half_multidouble_mi_corr_new.pkl", "rb") )
	# plot
	fig, axes = plt.subplots(2, 2, figsize=(9.69, 7))
	# subplot 1
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
	seaborn.violinplot(data=df1, x="sample size", y="l1 distance betweem PMF", hue="label", cut=0, ax=axes[0, 0], palette=seaborn.color_palette("Set3"))
	axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels())
	axes[0, 0].set_title("PMF accuracy (measurement = 20)")
	handles, labels = axes[0, 0].get_legend_handles_labels()
	axes[0, 0].legend(handles=handles, labels=labels)
	axes[0, 0].text(-1.3, 0.75, "A", fontsize=14, fontweight='bold')
	# subplot 2
	df2 = None
	for k,v in result_t_o.items():
		if k[0] != 1000:
			continue
		tmp = pd.DataFrame( {"measurement size" : k[1], "l1 distance betweem PMF" : v, "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"measurement size" : k[1], "l1 distance betweem PMF" : result_t_of[k], "label" : "corrected"} )
		if df2 is None:
			df2 = tmp
			df2 = df2.append(tmp2)
		else:
			df2 = df2.append(tmp, ignore_index = True)
			df2 = df2.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df2, x="measurement size", y="l1 distance betweem PMF", hue="label", cut=0, ax=axes[0, 1], palette=seaborn.color_palette("Set3"))
	axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels())
	axes[0, 1].set_title("PMF accuracy (sample = 1000)")
	handles, labels = axes[0, 1].get_legend_handles_labels()
	axes[0, 1].legend(handles=handles, labels=labels)
	axes[0, 1].text(-1.3, 0.73, "B", fontsize=14, fontweight='bold')
	# subplot 3
	df3 = None
	for k,v in result_mi_t.items():
		if k[1] != 20:
			continue
		tmp = pd.DataFrame( {"sample size" : k[0], "abs diff between MI" : np.abs(v - result_mi_o[k]), "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"sample size" : k[0], "abs diff between MI" : np.abs(v - result_mi_corr[k]), "label" : "corrected"} )
		if df3 is None:
			df3 = tmp
			df3 = df3.append(tmp2)
		else:
			df3 = df3.append(tmp, ignore_index = True)
			df3 = df3.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df3, x="sample size", y="abs diff between MI", hue="label", cut=0, ax=axes[1, 0], palette=seaborn.color_palette("pastel"))
	axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels())
	axes[1, 0].set_title("MI accuracy (measurement = 20)")
	handles, labels = axes[1, 0].get_legend_handles_labels()
	axes[1, 0].legend(handles=handles, labels=labels)
	axes[1, 0].text(-1.3, 0.65, "C", fontsize=14, fontweight='bold')
	# subplot 4
	df4 = None
	for k,v in result_mi_t.items():
		if k[0] != 1000:
			continue
		tmp = pd.DataFrame( {"measurement size" : k[1], "abs diff between MI" : np.abs(v - result_mi_o[k]), "label" : "baseline"} )
		tmp2 = pd.DataFrame( {"measurement size" : k[1], "abs diff between MI" : np.abs(v - result_mi_corr[k]), "label" : "corrected"} )
		if df4 is None:
			df4 = tmp
			df4 = df4.append(tmp2)
		else:
			df4 = df4.append(tmp, ignore_index = True)
			df4 = df4.append(tmp2, ignore_index = True)
	seaborn.violinplot(data=df4, x="measurement size", y="abs diff between MI", hue="label", cut=0, ax=axes[1, 1], palette=seaborn.color_palette("pastel"))
	axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels())
	axes[1, 1].set_title("MI accuracy (sample = 1000)")
	handles, labels = axes[1, 1].get_legend_handles_labels()
	axes[1, 1].legend(handles=handles, labels=labels)
	axes[1, 1].text(-1.3, 0.81, "D", fontsize=14, fontweight='bold')
	fig.subplots_adjust(hspace=0.4, wspace = 0.4)
	fig.savefig("../results/ismb_simulation_half_double_all.pdf", transparent = True, bbox_inches='tight')


def run_simulate(codedir):
	# multi covariance
	sample_sizes = [100, 500, 1000, 10000]
	bootstrap_sizes = [5,10,20, 50]
	n_simulation = 100
	# initialize comparison result
	result_t_o = {}
	result_tf_o = {}
	result_t_of = {}
	result_mi_t = {}
	result_mi_o = {}
	result_mi_corr_new = {}
	# initialize result vector
	for n_sample in sample_sizes:
		for n_bootstrap in bootstrap_sizes:
			result_t_o[(n_sample, n_bootstrap)] = []
			result_tf_o[(n_sample, n_bootstrap)] = []
			result_t_of[(n_sample, n_bootstrap)] = []
			result_mi_t[(n_sample, n_bootstrap)] = []
			result_mi_o[(n_sample, n_bootstrap)] = []
			result_mi_corr_new[(n_sample, n_bootstrap)] = []
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
					counts_corr, tmp_F_est = adjust_transition(counts, copy.copy(F_est))
					counts_corr = counts_corr.reshape((nbin_x, nbin_y))
					diffcount_t_of =  np.sum(np.abs(counts_corr / np.sum(counts_corr) - prob))
					# distance between calculated mutual information and truth
					# MI truth
					mi_truth = MI_bincount(prob)
					# MI calculated by raw count
					mi_o = MI_bincount(counts)
					# MI calculated by adjusted count
					counts_corr[np.where(counts_corr < 0)] = -counts_corr[np.where(counts_corr < 0)]
					mi_corr_new = MI_bincount(counts_corr)
					# append result vector
					result_t_o[(n_sample, n_bootstrap)].append( diffcount_t_o )
					result_tf_o[(n_sample, n_bootstrap)].append( diffcount_tf_o )
					result_t_of[(n_sample, n_bootstrap)].append( diffcount_t_of )
					result_mi_t[(n_sample, n_bootstrap)].append( mi_truth )
					result_mi_o[(n_sample, n_bootstrap)].append( mi_o )
					result_mi_corr_new[(n_sample, n_bootstrap)].append( mi_corr_new )
			pbar.update(1)
	# convert ty numpy array
	for n_sample in sample_sizes:
		for n_bootstrap in bootstrap_sizes:
			result_t_o[(n_sample, n_bootstrap)] = np.array(result_t_o[(n_sample, n_bootstrap)])
			result_tf_o[(n_sample, n_bootstrap)] = np.array(result_tf_o[(n_sample, n_bootstrap)])
			result_t_of[(n_sample, n_bootstrap)] = np.array(result_t_of[(n_sample, n_bootstrap)])
			result_mi_t[(n_sample, n_bootstrap)] = np.array(result_mi_t[(n_sample, n_bootstrap)])
			result_mi_o[(n_sample, n_bootstrap)] = np.array(result_mi_o[(n_sample, n_bootstrap)])
			result_mi_corr_new[(n_sample, n_bootstrap)] = np.array(result_mi_corr_new[(n_sample, n_bootstrap)])
	# pickle result to file
	pickle.dump( result_t_o, open( codedir + "../results/ismb_simulation_half_multidouble_t_o.pkl", "wb" ) )
	pickle.dump( result_tf_o, open( codedir + "../results/ismb_simulation_half_multidouble_tf_o.pkl", "wb" ) )
	pickle.dump( result_t_of, open( codedir + "../results/ismb_simulation_half_multidouble_t_of.pkl", "wb" ) )
	pickle.dump( result_mi_t, open( codedir + "../results/ismb_simulation_half_multidouble_mi_truth.pkl", "wb" ) )
	pickle.dump( result_mi_o, open( codedir + "../results/ismb_simulation_half_multidouble_mi_obs.pkl", "wb" ) )
	pickle.dump( result_mi_corr_new, open( codedir + "../results/ismb_simulation_half_multidouble_mi_corr_new.pkl", "wb" ) )

if __name__ == "__main__":
	codedir = "/".join(sys.argv[0].split("/")[:-1])
	if codedir == "":
		codedir = "./"
	else:
		codedir += "/"
		
	run_simulate()
	plot_combined()
