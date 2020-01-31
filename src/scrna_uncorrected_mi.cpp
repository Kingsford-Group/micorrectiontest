#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <omp.h>
#include <mutex>
#include <numeric>
#include <algorithm>
#include "boost/algorithm/string.hpp"
#include "Eigen/Dense"
#include "ProgressBar.hpp"
#include "cnpy.h"

using namespace std;


Eigen::MatrixXd load_normalize_scrna(string npyfile)
{
	cnpy::NpyArray npy_counts = cnpy::npy_load(npyfile);
	double* loaded_counts = npy_counts.data<double>();
	Eigen::MatrixXd counts = Eigen::MatrixXd::Zero(npy_counts.shape[0], npy_counts.shape[1]);
	for (int32_t i = 0; i < npy_counts.shape[0]; i++)
		for (int32_t j = 0; j < npy_counts.shape[1]; j++)
			counts(i,j) = loaded_counts[i * npy_counts.shape[1] + j];
	// normalize column sum to 1e4
	Eigen::VectorXd x = counts.colwise().sum();
	for (int32_t i = 0; i < counts.cols(); i++)
		counts.col(i) /= (x(i) / 1e4);
	// taking log
	Eigen::MatrixXd log_counts = (counts.array() > 0).select(counts.array().log().matrix(), Eigen::MatrixXd::Ones(counts.rows(), counts.cols()) * -25);
	log_counts.transposeInPlace();
	cout << "Finish read and log-transform the expression matrix of size " << log_counts.rows() <<"*"<< log_counts.cols() << endl;
	return log_counts;
};


Eigen::VectorXi discretize_updated(Eigen::MatrixXd x, int32_t n_bins, vector<double>& boundary)
{
	assert(x.cols() == 1);
	vector<double> vec(x.data(), x.data() + x.rows() * x.cols());
	sort(vec.begin(), vec.end());
	// resize vec to remove the first a few log(1e-8)
	vector<double>::const_iterator it_zero = std::lower_bound(vec.cbegin(), vec.cend(), -24.5);
	if (it_zero == vec.end()) {
		// if all expression values are zero, return a zero vector
		Eigen::VectorXi y = Eigen::VectorXi::Zero(x.rows());
		return y;
	}
	vector<double> tmp_vec;
	for (vector<double>::const_iterator it = it_zero; it != vec.cend(); it++)
		tmp_vec.push_back( *it );
	vec = tmp_vec;
	double quantile_20 = vec[(int32_t)(vec.size() * 0.2)];
	double quantile_90 = vec[(int32_t)(vec.size() * 0.9)];
	boundary.assign(n_bins, 0);
	// first boundary
	boundary[0] = -25 + 0.1;
	// boundaries in the middle
	for (int32_t i = 1; i < n_bins - 1; i++)
		boundary[i] = quantile_20 + (i-1) * (quantile_90 - quantile_20) / (n_bins - 3);
	// last boundary
	boundary[n_bins-1] = vec[(int32_t)vec.size()-1] + 100;
	// categorical values
	Eigen::VectorXi y = Eigen::VectorXi::Zero(x.rows());
	for (int32_t i = 0; i < x.rows(); i++) {
		vector<double>::iterator lb = std::lower_bound (boundary.begin(), boundary.end(), x(i, 0));
		y(i) = distance(boundary.begin(), lb);
	}
	assert(boundary.size() == n_bins);
	return y;
};


void process_exp_discretize(Eigen::MatrixXd& logexp, int32_t n_bins, Eigen::MatrixXi & bin_logexp, vector< vector<double> >& all_boundary)
{
	// initialize
	bin_logexp = Eigen::MatrixXi::Zero( logexp.rows(), logexp.cols() );
	all_boundary.clear();
	for (int32_t i = 0; i < logexp.cols(); i++) {
		vector<double> boundary;
		// Eigen::VectorXi	y = discretize(logexp.col(i), n_bins, boundary);
		Eigen::VectorXi	y = discretize_updated(logexp.col(i), n_bins, boundary);
		bin_logexp.col(i) = y;
		all_boundary.push_back( boundary );
	}
};


double mutual_information(Eigen::MatrixXi bin_logexp1, Eigen::MatrixXi bin_logexp2, int32_t n_bins)
{
	assert(bin_logexp1.cols() == 1 && bin_logexp2.cols() == 1);
	assert(bin_logexp1.rows() == bin_logexp2.rows());
	// count matrix
	Eigen::MatrixXd count_mat = Eigen::MatrixXd::Zero(n_bins, n_bins);
	for (int32_t k = 0; k < bin_logexp1.rows(); k++) {
		count_mat(bin_logexp1(k, 0), bin_logexp2(k, 0)) ++;
	}
	// mutual information
	count_mat /= count_mat.sum();
	// rowwise and colwise sum to calculate the marginal probability
	Eigen::VectorXd px = count_mat.rowwise().sum();
	Eigen::VectorXd py = count_mat.colwise().sum();
	double MI = 0;
	for (int32_t i = 0; i < px.size(); i++) 
		for (int32_t j = 0; j < py.size(); j++) {
			if (px(i) == 0 || py(j) ==  0 || count_mat(i,j) == 0)
				continue;
			MI += count_mat(i,j) * log(count_mat(i,j) / px(i) / py(j));
		}
	return MI;
};


Eigen::MatrixXd	process_all_mutual_information(Eigen::MatrixXi& bin_logexp, int32_t n_bins)
{
	time_t CurrentTime;
	string CurrentTimeStr;
	time(&CurrentTime);
	CurrentTimeStr=ctime(&CurrentTime);
	cout<<"["<<CurrentTimeStr.substr(0, CurrentTimeStr.size()-1)<<"] "<<"Start calculating pairwise mutual information."<<endl;

	int32_t n_genes = bin_logexp.cols();
	// prepare pair of gene indices for parallelization
	vector< pair<int32_t,int32_t> > idx_pair_genes;
	for (int32_t i = 0; i < n_genes; i++)
		for (int32_t j = i+1; j < n_genes; j++)
			idx_pair_genes.push_back( make_pair(i, j) );
	// result matrix
	Eigen::MatrixXd mi_matrix = Eigen::MatrixXd::Zero(n_genes, n_genes);
	ProgressBar progressBar(n_genes * (n_genes-1) / 2, 70);
	// ProgressBar progressBar(5000, 70);

	mutex mi_mutex;
	omp_set_num_threads(12);
	#pragma omp parallel for
	for(int32_t p = 0; p < idx_pair_genes.size(); p++) {
	// for(int32_t p = 0; p < 10000; p++) {
		int32_t i = idx_pair_genes[p].first;
		int32_t j = idx_pair_genes[p].second;
		mi_matrix(i,j) = mutual_information(bin_logexp.col(i), bin_logexp.col(j), n_bins);

		lock_guard<std::mutex> guard(mi_mutex);
		++progressBar;
		progressBar.display();
	}

	progressBar.done();
	time(&CurrentTime);
	CurrentTimeStr=ctime(&CurrentTime);
	cout << "[" << CurrentTimeStr.substr(0, CurrentTimeStr.size()-1) << "] " << "Finish MI calculation.\n";

	return mi_matrix;
};


void save_eigen_matrix(string outfile, const Eigen::MatrixXd & exp)
{
	ofstream ss(outfile, ios::out | ios::binary);
	// number rows and columns of matrix
	int32_t n_rows = exp.rows();
	int32_t n_cols = exp.cols();
	ss.write((char*)(&n_rows), sizeof(int32_t));
	ss.write((char*)(&n_cols), sizeof(int32_t));
	// write the specific rows and columns
	ss.write((char*)(exp.data()), n_rows * n_cols * sizeof(double));
	ss.close();
};


int32_t main(int32_t argc, char* argv[]) 
{
	Eigen::MatrixXd logexp = load_normalize_scrna(argv[1]);
	int32_t n_bins = atoi(argv[2]);

	// discretize
	Eigen::MatrixXi bin_logexp;
	vector< vector<double> > all_boundary;
	process_exp_discretize(logexp, n_bins, bin_logexp, all_boundary);
	cout << "Finish binning.\n";

	// calculate mutual information
	Eigen::MatrixXd mi_matrix = process_all_mutual_information(bin_logexp, n_bins);
	save_eigen_matrix(argv[3], mi_matrix);
}