#!/bin/python

import numpy as np
from scipy.stats import mvn
import sklearn.cluster


def MI_bincount(counts):
	'''
		calculate discrete mutual information.
		Input: count matrix (joint PMF with or without normalization), rows and columns are the categories of the two signals.
	'''
	px = 1.0 * np.sum(counts, axis=0) / np.sum(counts)
	py = 1.0 * np.sum(counts, axis=1) / np.sum(counts)
	pxy = 1.0 * counts / np.sum(counts)
	MI = 0
	for i in range(len(px)):
		for j in range(len(py)):
			if px[i] == 0 or py[j] == 0 or pxy[i,j] == 0:
				continue
			MI += pxy[i,j] * np.log(pxy[i,j] / px[i] / py[j])
	return MI


def BootstrapCovariance(bootstrap_x, bootstrap_y):
	'''
		Assumption: same error covariance for all samples
		bootstrap_x and bootstrap_y has the dimension of n_bootstrap * n_sample
	'''
	# for each sample, subtract the mean bootstrap
	meanshift_x = bootstrap_x - bootstrap_x.mean(axis = 0, keepdims = True)
	meanshift_y = bootstrap_y - bootstrap_y.mean(axis = 0, keepdims = True)
	# flatten and calculate covariance matrix
	flatten_ms_x = meanshift_x.flatten()
	flatten_ms_y = meanshift_y.flatten()
	# the following covariance is the covariance of one single observation
	cov = np.cov( np.vstack((flatten_ms_x, flatten_ms_y)) )
	# the following covariance is the covariance for mean observations
	cov /= bootstrap_x.shape[0]
	return cov


def BootstrapCovariance_multicov(bootstrap_x, bootstrap_y, boundary_x, boundary_y):
	'''
		Assumption: discrete case + each joint category has its own error covariance matrix
		Estimating error covariance for each (x_bin, y_bin).
		x bins: (-infinity, boundary_x[0]), (boundary_x[0], boundary_x[1]), ... 
		y bins: (-infinity, boundary_y[0]), (boundary_y[0], boundary_y[1]), ... 
		bootstrap_x and bootstrap_y has the dimension of n_bootstrap * n_sample
	'''
	mean_bootstrap_x = np.mean(bootstrap_x, axis=0)
	mean_bootstrap_y = np.mean(bootstrap_y, axis=0)
	# for each sample, get which bin it falls into based on the mean of bootstrap
	sample_indexes = {}
	for s in range(bootstrap_x.shape[1]):
		if mean_bootstrap_x[s] > boundary_x[-1]:
			idx_x = len(boundary_x) - 1
		else:
			idx_x = np.where(mean_bootstrap_x[s] < boundary_x)[0][0]
		if mean_bootstrap_y[s] > boundary_y[-1]:
			idx_y = len(boundary_y) - 1
		else:
			idx_y = np.where(mean_bootstrap_y[s] < boundary_y)[0][0]
		if (idx_x, idx_y) in sample_indexes:
			sample_indexes[(idx_x, idx_y)].append(s)
		else:
			sample_indexes[(idx_x, idx_y)] = [s]
	# estimate a covariance using all samples and use that as default when the corresponding bin does not have any sample in it
	meanshift_x = bootstrap_x - bootstrap_x.mean(axis = 0, keepdims = True)
	meanshift_y = bootstrap_y - bootstrap_y.mean(axis = 0, keepdims = True)
	flatten_ms_x = meanshift_x.flatten()
	flatten_ms_y = meanshift_y.flatten()
	cov = np.cov( np.vstack((flatten_ms_x, flatten_ms_y)) ) / bootstrap_x.shape[0]
	# initialize result matrix
	multi_covariance = np.zeros( (len(boundary_x), len(boundary_x), 2, 2) )
	for idx_x in range(len(boundary_x)):
		for idx_y in range(len(boundary_x)):
			if not ((idx_x, idx_y) in sample_indexes):
				multi_covariance[idx_x, idx_y, :, :] = cov
			else:
				# estimate from the corresponding samples
				idx_samples = np.array(sample_indexes[(idx_x, idx_y)])
				meanshift_x = bootstrap_x[:, idx_samples] - bootstrap_x[:, idx_samples].mean(axis = 0, keepdims = True)
				meanshift_y = bootstrap_y[:, idx_samples] - bootstrap_y[:, idx_samples].mean(axis = 0, keepdims = True)
				flatten_ms_x = meanshift_x.flatten()
				flatten_ms_y = meanshift_y.flatten()
				multi_covariance[idx_x, idx_y, :, :] = np.cov( np.vstack((flatten_ms_x, flatten_ms_y)) ) / bootstrap_x.shape[0]
	return multi_covariance


def BootstrapCovariance_cluster(bootstrap_x, bootstrap_y, k):
	'''
		Assumption: continuous case + each k-means cluster has its own error covariance matrix
		bootstrap_x and bootstrap_y has the dimension of n_bootstrap * n_sample
		Estimate the covariance for each sample, cluster smilar covariance matrices into k cluster.
		Finally estimate a covariance for each of k cluster using all points.
	'''
	n_sample = bootstrap_x.shape[1]
	cov_individial = np.zeros( (n_sample, 4) )
	# for each sample, subtract the mean bootstrap
	meanshift_x = bootstrap_x - bootstrap_x.mean(axis = 0, keepdims = True)
	meanshift_y = bootstrap_y - bootstrap_y.mean(axis = 0, keepdims = True)
	for i in range(n_sample):
		cov_individial[i,:] = np.cov( np.vstack((meanshift_x[:,i], meanshift_y[:,i])) ).flatten()
	# clustering
	# centroid, label = scipy.cluster.vq.kmeans2(cov_individial, k, minit='points')
	kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(cov_individial)
	label = kmeans.labels_
	# estimate the final covariance matrix for each cluster
	multi_covariance = []
	new_label = np.zeros(n_sample, dtype=np.int)
	count = 0
	for i in range(k):
		indexes = np.where(label == i)[0]
		if len(indexes) > 0:
			flatten_ms_x = meanshift_x[:,indexes].flatten()
			flatten_ms_y = meanshift_y[:,indexes].flatten()
			# the following covariance is the covariance of one single observation
			cov = np.cov( np.vstack((flatten_ms_x, flatten_ms_y)) )
			# the following covariance is the covariance for mean observations
			cov /= bootstrap_x.shape[0]
			multi_covariance.append( cov )
			new_label[indexes] = count
			count += 1
	multi_covariance = np.array(multi_covariance)
	return multi_covariance, label


def GaussianUniformIntegral(x_range, y_range, x_range_shift, y_range_shift, cov, num_x_grid = 3, num_y_grid = 3):
	'''
		Exact equation: 1 / (size of x_range) * 1 / (size of y_range) * integral_{box of x_range y range} dxdy integral_{box of x_range_shift y_range_shift} dxshift dyshift N(xshift - x, yshift - y | cov)
		Approximation: the uniform distribution using grids in x_range and y_range
						1 / (number x grids) * 1 / (number y grids) * sum_{x grids} sum_{y grids} integral dxshift dyshift N(xgrid - xshift, ygrid - yshift | cov)
	'''
	value = 0
	size_x_grid = (x_range[1] - x_range[0]) / (num_x_grid + 1)
	size_y_grid = (y_range[1] - y_range[0]) / (num_y_grid + 1)
	for xgrid in np.arange(x_range[0] + size_x_grid, x_range[1], size_x_grid):
		for ygrid in np.arange(y_range[0] + size_y_grid, y_range[1], size_y_grid):
			prob,_ = mvn.mvnun(np.array([x_range_shift[0] - xgrid, y_range_shift[0] - ygrid]), np.array([x_range_shift[1] - xgrid, y_range_shift[1] - ygrid]), np.zeros(2), cov)
			value += prob
	value /= (num_x_grid * num_y_grid)
	return value


def TransitionMatrix(cov, boundary_x, boundary_y, mean_x, mean_y):
	'''
		Assumption: the same measurement error distribution for all samples
		Calculate transition matrix F for discrete random variable.
		Input:
			cov: covariance matrix of measurement error 2 * 2.
			boundary_x: partition of real-valued x-axis into the categories.
			boundary_y: partition of real-valued y-axis into the categories.
			mean_x: center of each partition of x-axis.
			mean_y: center of each partition of y-axis.
	'''
	assert(cov.shape == (2,2))
	F = np.zeros( (len(boundary_x)*len(boundary_y), len(boundary_x)*len(boundary_y)) )
	for i in range(len(boundary_x)):
		for j in range(len(boundary_y)):
			# check the multivariate Gaussian integral for each possible grid that are not (i,j) itself
			for ti in range(len(boundary_x)):
				for tj in range(len(boundary_y)):
					if (ti, tj) == (i,j):
						continue
					# the range of x and y in the box of (i, j)
					if i > 0:
						x_range = (boundary_x[i - 1], boundary_x[i])
					else:
						x_range = (boundary_x[0] - 2*(boundary_x[0] - mean_x[0]), boundary_x[0])
					if j > 0:
						y_range = (boundary_y[j - 1], boundary_y[j])
					else:
						y_range = (boundary_y[0] - 2*(boundary_y[0] - mean_y[0]), boundary_y[0])
					# the range of x and y in the box of (ti, tj)
					if ti > 0:
						x_range_shift = (boundary_x[ti-1], boundary_x[ti])
					else:
						x_range_shift = (boundary_x[0] - 2*(boundary_x[0]-mean_x[0]), boundary_x[ti])
					if tj > 0:
						y_range_shift = (boundary_y[tj-1], boundary_y[tj])
					else:
						y_range_shift = (boundary_y[0] - 2*(boundary_y[0]-mean_y[0]), boundary_y[tj])
					# the integration of multivariate Gaussian within the box of x_shift * y_shift
					prob = GaussianUniformIntegral(x_range, y_range, x_range_shift, y_range_shift, cov)
					# remove too small values
					if prob > 1e-4:
						F[i*len(boundary_x) + j, ti*len(boundary_x) + tj] = prob
	# add the diagonal elements (self transition)
	for i in range(F.shape[0]):
		rest = 1 - np.sum(F[i,:])
		F[i,i] = rest
		assert(F[i,i] > 0)
	return F


def TransitionMatrix_multicov(multi_covariance, boundary_x, boundary_y, mean_x, mean_y):
	'''
		Assumption: each joint category has its own error covariance.
		Calculate transition matrix F for discrete random variable.
		Input:
			multi_covariance: covariance matrix of measurement error (n joint categories * 2 * 2.)
			boundary_x: partition of real-valued x-axis into the categories.
			boundary_y: partition of real-valued y-axis into the categories.
			mean_x: center of each partition of x-axis.
			mean_y: center of each partition of y-axis.
	'''
	assert(multi_covariance.shape == (len(boundary_x), len(boundary_y), 2,2))
	F = np.zeros( (len(boundary_x)*len(boundary_y), len(boundary_x)*len(boundary_y)) )
	for i in range(len(boundary_x)):
		for j in range(len(boundary_y)):
			# check the multivariate Gaussian integral for each possible grid that are not (i,j) itself
			for ti in range(len(boundary_x)):
				for tj in range(len(boundary_y)):
					if (ti, tj) == (i,j):
						continue
					# the range of x and y in the box of (i, j)
					if i > 0:
						x_range = (boundary_x[i - 1], boundary_x[i])
					else:
						x_range = (boundary_x[0] - 2*(boundary_x[0] - mean_x[0]), boundary_x[0])
					if j > 0:
						y_range = (boundary_y[j - 1], boundary_y[j])
					else:
						y_range = (boundary_y[0] - 2*(boundary_y[0] - mean_y[0]), boundary_y[0])
					# the range of x and y in the box of (ti, tj)
					if ti > 0:
						x_range_shift = (boundary_x[ti-1], boundary_x[ti])
					else:
						x_range_shift = (boundary_x[0] - 2*(boundary_x[0]-mean_x[0]), boundary_x[ti])
					if tj > 0:
						y_range_shift = (boundary_y[tj-1], boundary_y[tj])
					else:
						y_range_shift = (boundary_y[0] - 2*(boundary_y[0]-mean_y[0]), boundary_y[tj])
					# the integration of multivariate Gaussian within the box of x_shift * y_shift
					prob = GaussianUniformIntegral(x_range, y_range, x_range_shift, y_range_shift, multi_covariance[i,j,:,:])
					# remove too small values
					if prob > 1e-4:
						F[i*len(boundary_x) + j, ti*len(boundary_x) + tj] = prob
	# add the diagonal elements (self transition)
	for i in range(F.shape[0]):
		rest = 1 - np.sum(F[i,:])
		F[i,i] = rest
		assert(F[i,i] > 0)
	return F


def MI_KDE(x, y, h_x, h_y):
	'''
		x, y are vectors of the same length.
		h is the bandwidth.
		Assuming x and y have the same bandwidth
	'''
	assert(len(x) == len(y))
	# estimate the marginal probability at points x
	diff_x = x.reshape((len(x),1)).dot(np.ones((1,len(x)))) - np.ones((len(x),1)).dot(x.reshape((1,len(x))))
	kernel_x = np.exp(- diff_x**2 / 2 / h_x**2)
	px = np.sum(1.0 / len(x) / h_x / np.sqrt(2 * np.pi) * kernel_x, axis=1)
	# estimate the marginal probability at points y
	diff_y = y.reshape((len(y),1)).dot(np.ones((1,len(y)))) - np.ones((len(y),1)).dot(y.reshape((1,len(y))))
	kernel_y = np.exp(- diff_y**2 / 2 / h_y**2)
	py = np.sum(1.0 / len(y) / h_y / np.sqrt(2 * np.pi) * kernel_y, axis=1)
	# estimate the joint distribution via KDE
	sum_diff_xy = diff_x**2 / h_x**2 + diff_y**2 / h_y**2
	kernel_xy = 1.0 / len(x) / h_x/h_y / 2 / np.pi * np.exp(- sum_diff_xy / 2)
	pxy = np.sum(kernel_xy, axis=1)
	# mutual information
	mi = 1.0 / len(x) * np.sum(np.log( pxy / px / py))
	return px, py, pxy, mi


def MI_KDE_correction(x, y, h_x, h_y, covariance_est):
	'''
		Correct estimation for px and py in close form.
		For pxy, constructing X^TAX such that expectation of exp(- X^TAX / 2h^2) is equal to exp(- mu^Tmu / 2h^2)
		When correcting px and py: correct on kernel level for each individual (xi - xj)
	'''
	# for x
	rx = covariance_est[0,0] / (h_x**2 - 2 * covariance_est[0,0])
	diff_x = x.reshape((len(x),1)).dot(np.ones((1,len(x)))) - np.ones((len(x),1)).dot(x.reshape((1,len(x))))
	# theoretically kernel_mu_x = np.exp(- diff_mu_x**2 / 2 / h**2), however diff_mu_x is not observed, using the following estimator
	# kernel_mu_x = np.exp(- diff_x**2 * rx) * np.sqrt(1 + 2 * rx)
	kernel_mu_x = np.exp(- diff_x**2 / 2 / (h_x**2 - 2 * covariance_est[0,0])) * np.sqrt(1 + 2 * rx)
	p_mu_x = np.sum(1.0 / len(x) / h_x / np.sqrt(2 * np.pi) * kernel_mu_x, axis=1)
	# for y
	ry = covariance_est[1,1] / (h_y**2 - 2 * covariance_est[1,1])
	diff_y = y.reshape((len(y),1)).dot(np.ones((1,len(y)))) - np.ones((len(y),1)).dot(y.reshape((1,len(y))))
	# theoretically kernel_mu_y = np.exp(- diff_mu_y**2 / 2 / h**2), however diff_mu_y is not observed, using the following estimator
	# kernel_mu_y = np.exp(- diff_y**2 * ry) * np.sqrt(1 + 2 * ry)
	kernel_mu_y = np.exp(- diff_y**2 / 2 / (h_y**2 - 2 * covariance_est[1,1])) * np.sqrt(1 + 2 * ry)
	p_mu_y = np.sum(1.0 / len(x) / h_y / np.sqrt(2 * np.pi) * kernel_mu_y, axis=1)
	# for xy
	# construct A by eigen value decomposition of covariance_est
	tmp_xsi, tmp_P = np.linalg.eig(covariance_est * 2)
	# the formula of numpy eig is P*Sigma*P.T = A. However, in my derivation, I assume P.T * Sigma * P = A. Thus need to transpose P
	tmp_P = tmp_P.transpose()
	Sigma_1_2 = tmp_P.transpose().dot(np.diag(np.sqrt(tmp_xsi))).dot(tmp_P)
	Sigma_neg_1_2 = tmp_P.transpose().dot(np.diag(1 / np.sqrt(tmp_xsi))).dot(tmp_P)
	xsi, P = np.linalg.eig( Sigma_1_2.dot(np.diag(1 / 2 / np.array([h_x**2, h_y**2]))).dot(Sigma_1_2) )
	# similarly, need to transpose P
	P = P.transpose()
	# set t = -1
	lambd = xsi / (1 - 2 * xsi)
	A = Sigma_neg_1_2.dot(P.transpose()).dot(np.diag(lambd)).dot(P).dot(Sigma_neg_1_2)
	# calculating kernel for X^TAX
	diff_xax = diff_x**2 * A[0,0] + diff_y**2 * A[1,1] + diff_x*diff_y * A[0,1] + diff_x*diff_y * A[1,0]
	kernel_xax = np.exp( - diff_xax)
	# calculate kernel of mu by adjusting kernel_xax
	kernel_mu_x_mu_y = np.sqrt(np.prod(2 * lambd + 1)) * kernel_xax
	p_mu_x_mu_y = 1.0 / h_x / h_y / 2 / np.pi * np.mean(kernel_mu_x_mu_y, axis=1)
	# mutual information
	mi = 1.0 / len(x) * np.sum(np.log( p_mu_x_mu_y / p_mu_x / p_mu_y))
	return mi


def MI_KDE_correction_cluster(x, y, h_x, h_y, multi_covariance_est, label):
	indexes = np.argsort(label)
	x = x[indexes]
	y = y[indexes]
	label = label[indexes]
	# diff among original x
	diff_x = x.reshape((len(x),1)).dot(np.ones((1,len(x)))) - np.ones((len(x),1)).dot(x.reshape((1,len(x))))
	# diff among original y
	diff_y = y.reshape((len(y),1)).dot(np.ones((1,len(y)))) - np.ones((len(y),1)).dot(y.reshape((1,len(y))))
	# create kernel matrix
	kernel_mu_x = np.zeros(diff_x.shape)
	kernel_mu_y = np.zeros(diff_y.shape)
	kernel_mu_x_mu_y = np.zeros(diff_x.shape)
	# for each cluster, fill in the adjusted diff matrix
	for i in range(label[-1] + 1):
		# starting and ending index of the corresponding cluster
		s_i = np.min(np.where(label == i)[0])
		t_i = np.max(np.where(label == i)[0]) + 1
		for j in range(i, label[-1] + 1):
			this_covariance_est = (multi_covariance_est[i,:,:] + multi_covariance_est[j,:,:]) / 2
			s_j = np.min(np.where(label == j)[0])
			t_j = np.max(np.where(label == j)[0]) + 1
			# for x
			rx = this_covariance_est[0,0] / (h_x**2 - 2 * this_covariance_est[0,0])
			kernel_mu_x[s_i:t_i, s_j:t_j] = np.exp(- diff_x[s_i:t_i, s_j:t_j]**2 / 2 / (h_x**2 - 2 * this_covariance_est[0,0])) * np.sqrt(1 + 2 * rx)
			if i != j:
				kernel_mu_x[s_j:t_j, s_i:t_i] = np.exp(- diff_x[s_j:t_j, s_i:t_i]**2 / 2 / (h_x**2 - 2 * this_covariance_est[0,0])) * np.sqrt(1 + 2 * rx)
			# for y
			ry = this_covariance_est[1,1] / (h_y**2 - 2 * this_covariance_est[1,1])
			kernel_mu_y[s_i:t_i, s_j:t_j] = np.exp(- diff_y[s_i:t_i, s_j:t_j]**2 / 2 / (h_y**2 - 2 * this_covariance_est[1,1])) * np.sqrt(1 + 2 * ry)
			if i != j:
				kernel_mu_y[s_j:t_j, s_i:t_i] = np.exp(- diff_y[s_j:t_j, s_i:t_i]**2 / 2 / (h_y**2 - 2 * this_covariance_est[1,1])) * np.sqrt(1 + 2 * ry)
			# for xy
				# construct A by eigen value decomposition of this_covariance_est
			tmp_xsi, tmp_P = np.linalg.eig(this_covariance_est * 2)
			# the formula of numpy eig is P*Sigma*P.T = A. However, in my derivation, I assume P.T * Sigma * P = A. Thus need to transpose P
			tmp_P = tmp_P.transpose()
			Sigma_1_2 = tmp_P.transpose().dot(np.diag(np.sqrt(tmp_xsi))).dot(tmp_P)
			Sigma_neg_1_2 = tmp_P.transpose().dot(np.diag(1 / np.sqrt(tmp_xsi))).dot(tmp_P)
			xsi, P = np.linalg.eig( Sigma_1_2.dot(np.diag(1 / 2 / np.array([h_x**2, h_y**2]))).dot(Sigma_1_2) )
			# similarly, need to transpose P
			P = P.transpose()
			# set t = -1
			lambd = xsi / (1 - 2 * xsi)
			A = Sigma_neg_1_2.dot(P.transpose()).dot(np.diag(lambd)).dot(P).dot(Sigma_neg_1_2)
			# calculating kernel for X^TAX
			tmp = diff_x[s_i:t_i, s_j:t_j]**2 * A[0,0] + diff_y[s_i:t_i, s_j:t_j]**2 * A[1,1] + diff_x[s_i:t_i, s_j:t_j]*diff_y[s_i:t_i, s_j:t_j] * A[0,1] + diff_x[s_i:t_i, s_j:t_j]*diff_y[s_i:t_i, s_j:t_j] * A[1,0]
			kernel_mu_x_mu_y[s_i:t_i, s_j:t_j] = np.sqrt(np.prod(2 * lambd + 1)) * np.exp( - tmp)
			if i != j:
				kernel_mu_x_mu_y[s_j:t_j, s_i:t_i] = kernel_mu_x_mu_y[s_i:t_i, s_j:t_j].transpose()
	# KDE from the kernel
	p_mu_x = 1.0 / len(x) / h_x / np.sqrt(2 * np.pi) * np.sum(kernel_mu_x, axis=1)
	p_mu_y = 1.0 / len(x) / h_y / np.sqrt(2 * np.pi) * np.sum(kernel_mu_y, axis=1)
	p_mu_x_mu_y = 1.0 / h_x / h_y / 2 / np.pi * np.mean(kernel_mu_x_mu_y, axis=1)
	# mutual information
	mi = 1.0 / len(x) * np.sum(np.log( p_mu_x_mu_y / p_mu_x / p_mu_y))
	return p_mu_x, p_mu_y, p_mu_x_mu_y, mi


def MI_singleGaussian(covariance):
	rho = covariance[1,0] / np.sqrt(covariance[0,0] * covariance[1,1])
	mi = -0.5 * np.log2(1 - rho*rho)
	return mi