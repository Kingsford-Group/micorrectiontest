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


void mean_from_boundary(const vector< vector<double> >& all_boundary, vector< vector<double> >& all_mean)
{
	// clear result
	all_mean.clear();
	// start calculate mean value of each bin
	for (int32_t i = 0; i < all_boundary.size(); i++) {
		int32_t n_bins = all_boundary[i].size();
		assert( n_bins > 3 );
		vector<double> this_mean(n_bins, 0);
		for (int32_t j = 2; j < this_mean.size() - 1; j++)
			this_mean[j] = (all_boundary[i][j-1] + all_boundary[i][j]) / 2;
		this_mean[1] = 2 * this_mean[2] - this_mean[3];
		this_mean[0] = all_boundary[i][0];
		this_mean[n_bins - 1] = 2 * this_mean[n_bins - 2] - this_mean[n_bins - 3];
		all_mean.push_back(this_mean);
	}
};


vector<Eigen::MatrixXd> read_vector_of_eigen_matrix(string filename)
{
	ifstream fpin(filename, ios::binary);
	int32_t n_mat;
	int32_t n_rows;
	int32_t n_cols;
	fpin.read((char*)&n_mat, sizeof(int32_t));
	fpin.read((char*)&n_rows, sizeof(int32_t));
	fpin.read((char*)&n_cols, sizeof(int32_t));

	// initialize
	vector<Eigen::MatrixXd> all_bootstrap;
	for (int32_t i = 0; i < n_mat; i++) {
		Eigen::MatrixXd tpm = Eigen::MatrixXd::Zero(n_rows, n_cols);
		for (int32_t j = 0; j < n_cols; j++) {
			vector<double> tmp(n_rows, 0);
			fpin.read((char*)(tmp.data()), n_rows*sizeof(double));
			Eigen::VectorXd tmpexp_eigen = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());
			tpm.col(j) = tmpexp_eigen;
		}
		all_bootstrap.push_back( tpm );
	}
	fpin.close();
	return all_bootstrap;
};


Eigen::MatrixXd pair_up_dropout_transitions( const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2)
{
	int32_t n_bins = t1.rows();
	assert(t1.cols() == n_bins && t2.rows() == n_bins && t2.cols() == n_bins);
	Eigen::MatrixXd tpair = Eigen::MatrixXd::Zero( n_bins*n_bins, n_bins*n_bins );

	for (int32_t i = 0; i < n_bins; i++) {
		for (int32_t j = 0; j < n_bins; j++) {
			if (i == 0 && j == 0)
				tpair(0, i + j * n_bins) = 1;
			else if (i == 0 && j > 0) {
				tpair(0, i + j * n_bins) = t2(j, 0);
				tpair(i + j * n_bins, i + j * n_bins) = t2(j, j);
			}
			else if (i > 0 && j == 0) {
				tpair(0, i + j * n_bins) = t1(i, 0);
				tpair(i + j * n_bins, i + j * n_bins) = t1(i, i);
			}
			else {
				tpair(0 + j * n_bins, i + j * n_bins) = t1(i, 0) * t2(j, j);
				tpair(i + 0, i + j * n_bins) = t1(i, i) * t2(j, 0);
				tpair(0, i + j * n_bins) = t1(i, 0) * t2(j, 0);
				tpair(i + j * n_bins, i + j * n_bins) = t1(i, i) * t2(j, j);
			}
		}
		assert( fabs(tpair.col(i).sum() - 1) < 1e-4 );
	}
	return tpair;
};


double mutual_information_corrected(const Eigen::MatrixXi& bin_logexp1, const Eigen::MatrixXi& bin_logexp2, const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2, int32_t n_bins)
{
	assert(bin_logexp1.cols() == 1 && bin_logexp2.cols() == 1);
	assert(bin_logexp1.rows() == bin_logexp2.rows());
	// count matrix
	Eigen::MatrixXd count_mat = Eigen::MatrixXd::Zero(n_bins, n_bins);
	for (int32_t k = 0; k < bin_logexp1.rows(); k++) {
		count_mat(bin_logexp1(k, 0), bin_logexp2(k, 0)) ++;
	}
	count_mat.resize(n_bins * n_bins, 1);
	
	Eigen::MatrixXd tpair = pair_up_dropout_transitions(t1, t2);
	Eigen::MatrixXd corrected_mat;
	
	vector<int32_t> index_avoid;
	vector<double> multiplier(n_bins * n_bins, 0.99);
	int32_t round = 0;
	double negsum = 0;
	while (true) {
		// round ++;
		// if (round < 10) {
			// If after so many rounds, the sum of negative values is still very large in absolute values, the following case may happen:
			// in the previous round, the transition probability w.r.t a bin is shrinked, but in the next round it is enlarged when shrinking other bins and normalizing to 1.
			index_avoid.clear();
		// }
		negsum = 0;
		corrected_mat = tpair.partialPivLu().solve(count_mat);
		for (int32_t i = 0; i < corrected_mat.rows(); i++) {
			if (corrected_mat(i, 0) < 0) {
				negsum += corrected_mat(i, 0);
				tpair.row(i) *= multiplier[i];
				tpair.col(i) *= multiplier[i];
				index_avoid.push_back(i);
			}
		}
		corrected_mat.resize(n_bins, n_bins);
		if (negsum > -1)
			break;
		sort(index_avoid.begin(), index_avoid.end());
		index_avoid.resize( distance(index_avoid.begin(), unique(index_avoid.begin(), index_avoid.end())) );
		vector<int32_t> index_retain;
		for (int32_t i = 0; i < n_bins * n_bins; i++) {
			if (!binary_search(index_avoid.begin(), index_avoid.end(), i))
				index_retain.push_back(i);
		}
		for (int32_t i : index_avoid) {
			tpair(i, i) += 1 - tpair.col(i).sum();
			// multiplier[i] -= 0.1;
		}
		for (int32_t i : index_retain) {
			// re-distribute the probability in proportion to the transition probability of index_retain
			tpair(index_retain, i) += (1 - tpair.col(i).sum()) * tpair(index_retain, i) / tpair(index_retain, i).array().sum();
			assert( fabs(tpair.col(i).sum() - 1) < 1e-4 );
			// multiplier[i] = 0.9;
		}
	}
	// change the negative entries to positive
	corrected_mat = (corrected_mat.array() >= 0).select(corrected_mat, -corrected_mat);
	// mutual information
	corrected_mat.resize(n_bins, n_bins);
	corrected_mat /= corrected_mat.sum();
	// rowwise and colwise sum to calculate the marginal probability
	Eigen::VectorXd px = corrected_mat.rowwise().sum();
	Eigen::VectorXd py = corrected_mat.colwise().sum();
	double MI = 0;
	for (int32_t i = 0; i < px.size(); i++) 
		for (int32_t j = 0; j < py.size(); j++) {
			if (px(i) == 0 || py(j) ==  0 || corrected_mat(i,j) == 0)
				continue;
			MI += corrected_mat(i,j) * log(corrected_mat(i,j) / px(i) / py(j));
		}
	return MI;
};


Eigen::MatrixXd	process_all_corrected_mutual_information(const Eigen::MatrixXi& bin_logexp, const vector< Eigen::MatrixXd >& transitions, int32_t n_bins, bool is_part_1)
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
	int32_t part_start, part_end;
	if (is_part_1) {
		part_start = 0;
		part_end = idx_pair_genes.size() / 2;
	}
	else {
		part_start = idx_pair_genes.size() / 2;
		part_end = idx_pair_genes.size();
	}

	time(&CurrentTime);
	CurrentTimeStr=ctime(&CurrentTime);
	cout<<"["<<CurrentTimeStr.substr(0, CurrentTimeStr.size()-1)<<"] "<<part_start << "\t"<< part_end << endl;
	
	ProgressBar progressBar(part_end - part_start, 70);
	// ProgressBar progressBar(15000, 70);

	mutex mi_mutex;
	omp_set_num_threads(32);
	#pragma omp parallel for
	for(int32_t p = part_start; p < part_end; p++) {
	// for(int32_t p = 0; p < 15000; p++) {
		int32_t i = idx_pair_genes[p].first;
		int32_t j = idx_pair_genes[p].second;
		mi_matrix(i,j) = mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins);

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
	string count_npyfile(argv[1]);
	int32_t n_bins = atoi(argv[2]);
	string transition_folder(argv[3]);
	bool is_part_1 = atoi(argv[4]);
	string out_prefix(argv[5]);

	Eigen::MatrixXd logexp = load_normalize_scrna( count_npyfile );
	cout << "out mi file: " << (out_prefix + "_" + to_string(is_part_1) + ".dat") << endl;

	// discretize
	Eigen::MatrixXi bin_logexp;
	vector< vector<double> > all_boundary;
	process_exp_discretize(logexp, n_bins, bin_logexp, all_boundary);
	vector< vector<double> > all_mean;
	mean_from_boundary(all_boundary, all_mean);
	cout << "Finish binning.\n";

	// transition matrix corresponding to the dropout
	vector< Eigen::MatrixXd > transitions = read_vector_of_eigen_matrix(transition_folder + "/c_sc_transition_" + to_string(n_bins) + ".dat");

	int32_t i = 22, j = 5977; // supposed to be small
	cout << mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins) << endl;

	i = 40; j = 9652; // supposed to be small
	cout << mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins) << endl;

	i = 138; j = 4149; // supposed to be small
	cout << mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins) << endl;

	i = 1483, j = 1484;
	cout << mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins) << endl;

	// this is supposed to be small?
	i = 47; j = 13504;
	cout << mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins) << endl;

	i = 1913; j = 1914;
	cout << mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins) << endl;

	// this is supposed to be large?
	i = 12414; j = 12417;
	cout << mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins) << endl;

	i = 3; j = 8;
	cout << mutual_information_corrected(bin_logexp.col(i), bin_logexp.col(j), transitions[i], transitions[j], n_bins) << endl;

	// calculate mutual information
	Eigen::MatrixXd mi_matrix = process_all_corrected_mutual_information(bin_logexp, transitions, n_bins, is_part_1);
	save_eigen_matrix(out_prefix + "_" + to_string(is_part_1) + ".dat", mi_matrix);
}
