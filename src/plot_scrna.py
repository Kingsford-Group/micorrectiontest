#!/bin/python

import sys
import numpy as np
import copy
import tqdm
import time
import struct
from matplotlib import pyplot as plt
import pandas as pd
import seaborn
import SeabornFig2Grid as sfg
import matplotlib.gridspec as gridspec
plt.rcParams.update({'font.size': 14})


def read_eigen_matrix(filename):
    # read binary file
    fp = open(filename, 'rb')
    n_rows = struct.unpack('i', fp.read(4))[0]
    n_cols = struct.unpack('i', fp.read(4))[0]
    # read matrix entries
    matrix = np.zeros( (n_rows, n_cols) )
    for i in range(n_cols):
        for j in range(n_rows):
            matrix[j, i] = struct.unpack('d', fp.read(8))[0]
    fp.close()
    return matrix


def read_similar_genes(filename, name_index, threshold=0.8):
    pairs = []
    fp = open(filename, 'r')
    for line in fp:
        if line[0] == '#':
            continue
        strs = line.strip().split("\t")
        if float(strs[5]) > threshold and (strs[0] in name_index) and (strs[1] in name_index):
            genes = [name_index[strs[i]] for i in range(2)]
            genes.sort()
            pairs.append( tuple(genes) )
    fp.close()
    return pairs


def read_similar_genes_jaccardonly(filename, name_index, threshold=0.8):
    pairs = []
    fp = open(filename, 'r')
    for line in fp:
        if line[0] == '#':
            continue
        strs = line.strip().split("\t")
        if float(strs[2]) > threshold and (strs[0] in name_index) and (strs[1] in name_index):
            genes = [name_index[strs[i]] for i in range(2)]
            genes.sort()
            pairs.append( tuple(genes) )
    fp.close()
    return pairs


def read_similar_genes_ensembl(filename, name_index):
    # map from the gene id without version number to the gene id with version number
    version_map = {k.split(".")[0]:k for k in name_index.keys()}
    pairs = []
    fp = open(filename, 'r')
    for line in fp:
        if line[0] == '#':
            continue
        strs = line.strip().split("\t")
        # check that the gene id without version number is within the protein coding genes
        if (strs[0] in version_map) and (strs[2] in version_map):
            gene_names = [version_map[strs[0]], version_map[strs[2]]]
            genes = [name_index[x] for x in gene_names]
            genes.sort()
            pairs.append( tuple(genes) )
    fp.close()
    pairs = list(set(pairs))
    pairs.sort()
    return pairs


def GetFeature(line, key):
    s = line.index(key)
    t = line.index(";", s+1)
    return line[(s+len(key)+2):(t-1)]


def Gene_Symbol_Map(gtffile):
    Gene_Symbol = {}
    Symbol_Gene = {}
    fp = open(gtffile, 'r')
    for line in fp:
        if line[0] == '#':
            continue
        strs = line.strip().split("\t")
        if strs[2] == "gene":
            gene_id = GetFeature(line, "gene_id")
            gene_symbol = GetFeature(line, "gene_name")
            Gene_Symbol[gene_id] = gene_symbol
            Symbol_Gene[gene_symbol] = gene_id
    fp.close()
    print("Read {} gene symbol maps.".format(len(Gene_Symbol)))
    return Gene_Symbol, Symbol_Gene


def ReadTFlinks(filename, Symbol_Gene, name_index):
    TFlinks = {}
    fp = open(filename, 'r')
    for line in fp:
        strs = line.strip().split("\t")
        if (strs[0] in Symbol_Gene) and (strs[1] in Symbol_Gene):
            TFlinks[(Symbol_Gene[strs[0]], Symbol_Gene[strs[1]])] = strs[2]
    fp.close()
    print("Read {} TF-target gene links.".format(len(TFlinks)))
    # further convert to index
    pairs = []
    for k,v in TFlinks.items():
        if (k[0] in name_index) and (k[1] in name_index):
            genes = [name_index[k[0]], name_index[k[1]]]
            genes.sort()
            pairs.append( tuple(genes) )
    return pairs


def ReadKEGGlinks(filename, gene_names, name_index):
    # map the gene name without version number to the gene name with version number
    name_map = {x.split(".")[0]:x for x in gene_names}
    # result
    KEGGlinks = []
    # read file
    fp = open(filename, 'r')
    linecount = 0
    for line in fp:
        linecount += 1
        if linecount == 1:
            continue
        strs = line.strip().split("\t")
        if (strs[7] in name_map) and (strs[8] in name_map):
            KEGGlinks.append( (name_map[strs[7]], name_map[strs[8]]) )
    fp.close()
    # further convert to index
    pairs = []
    for k in KEGGlinks:
        if (k[0] in name_index) and (k[1] in name_index):
            genes = [name_index[k[0]], name_index[k[1]]]
            genes.sort()
            pairs.append( tuple(genes) )
    return pairs


def plot_example_setdiff(mi, corrmi, iu, log_counts, quantile = 99.99999):
    # plot set diff with a given percentage threshold
    idx1, idx2 = np.where(mi > np.percentile(mi[iu], quantile))
    v2_idx1, v2_idx2 = np.where(corrmi > np.percentile(corrmi[iu], quantile))
    mi_unique = set([(idx1[i], idx2[i]) for i in range(len(idx1)) ]) - set([(v2_idx1[i], v2_idx2[i]) for i in range(len(v2_idx1)) ])
    corrmi_unique = set([(v2_idx1[i], v2_idx2[i]) for i in range(len(v2_idx1)) ]) - set([(idx1[i], idx2[i]) for i in range(len(idx1)) ])
    # restrict to 12
    if len(corrmi_unique) > 12:
        corrmi_unique = list(corrmi_unique)[:12]
    if len(mi_unique) > 12:
        mi_unique = list(mi_unique)[:12]
    # two figures
    # mi
    fig_mi, axes_mi = plt.subplots(4, 3, figsize=(10, 12), sharex = True, sharey = True)
    for i in range(4):
        for j in range(3):
            k1 = mi_unique[i * 3 + j][0]
            k2 = mi_unique[i * 3 + j][1]
            axes_mi[i,j].scatter(log_counts[k1,:], log_counts[k2,:], alpha=0.2, s = 4)
            axes_mi[i,j].set_xlabel(gene_names[k1])
            axes_mi[i,j].set_ylabel(gene_names[k2])
    fig_mi.subplots_adjust(hspace = 0.4, wspace = 0.3)
    # corrmi
    fig_corr, axes_corr = plt.subplots(4, 3, figsize=(10, 12), sharex = True, sharey = True)
    for i in range(4):
        for j in range(3):
            k1 = corrmi_unique[i * 3 + j][0]
            k2 = corrmi_unique[i * 3 + j][1]
            axes_corr[i,j].scatter(log_counts[k1,:], log_counts[k2,:], alpha=0.2, s = 4)
            axes_corr[i,j].set_xlabel(gene_names[k1])
            axes_corr[i,j].set_ylabel(gene_names[k2])
    fig_corr.subplots_adjust(hspace = 0.4, wspace = 0.3)
    return fig_mi, fig_corr


def plot_scatter_hist(mi, percentages, iu, outputfile):
    # average percentage between gene pairs
    avg_percentage = np.diag(percentages).dot(np.ones( mi.shape )) + np.ones( mi.shape ).dot( np.diag(percentages) )
    avg_percentage /= 2
    # plot
    df = pd.DataFrame( {"gene pair average dropout rate" : avg_percentage[iu], "mutual information" : mi[iu]} )
    g0 = seaborn.jointplot(data = df, x = "gene pair average dropout rate", y = "mutual information", alpha = 0.1)
    df = pd.DataFrame( {"gene pair average dropout rate" : avg_percentage[iu], "corrected mutual information" : corrmi[iu]} )
    g1 = seaborn.jointplot(data = df, x = "gene pair average dropout rate", y = "corrected mutual information", alpha = 0.1)
    fig = plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(1, 2)
    mg0 = sfg.SeabornFig2Grid(g0, fig, gs[0])
    mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])
    gs.tight_layout(fig)
    fig.savefig(outputfile, transparent = True, bbox_inches='tight')


if __name__ == "__main__":
    mi = read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xmouse_brain/alevin/raw_mi_10.dat")
    corrmi = read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xmouse_brain/alevin/corr_mi_10_0.dat")
    corrmi += read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xmouse_brain/alevin/corr_mi_10_1.dat")
    iu = np.triu_indices(mi.shape[0])

    # gene names
    fp = open("/home/cong/Documents/MI_uncertainty/10xmouse_brain/alevin/seurat_selection_genenames.txt", 'r')
    gene_names = [line.strip() for line in fp.readlines()]
    fp.close()
    name_index = {gene_names[i]:i for i in range(len(gene_names))}

    ori_pairs = read_similar_genes_ensembl("/home/cong/Documents/MI_uncertainty/ensembl_mouse_paralog", name_index)
    utr_sim = read_similar_genes_jaccardonly("/home/cong/Documents/MI_uncertainty/nucmer.gencode.vM23.transcript.utr_geneutr_similarity.txt", name_index)
    pairs = set(ori_pairs) & set(utr_sim)

    for quantile in [99.9, 99.99, 99.999]:
        idx1, idx2 = np.where(mi > np.percentile(mi[iu], quantile))
        overlap_raw =  len( pairs & set([(idx1[i], idx2[i]) for i in range(len(idx1))]))
        v2_idx1, v2_idx2 = np.where(corrmi > np.percentile(corrmi[iu], quantile))
        overlap_corr = len( pairs & set([(v2_idx1[i], v2_idx2[i]) for i in range(len(v2_idx2))])) 
        agreement = len(set([(idx1[i], idx2[i]) for i in range(len(idx1))]) & set([(v2_idx1[i], v2_idx2[i]) for i in range(len(v2_idx2))]) ) / len(idx1)
        print("quantile = {}\tagreement = {}\traw = {}\tcorr = {}".format( quantile, agreement, overlap_raw, overlap_corr ) )

    # Gene_Symbol, Symbol_Gene = Gene_Symbol_Map("/home/cong/Documents/Gencode/gencode.vM23.primary_assembly.annotation.gtf")
    # interactions = ReadTFlinks("/home/cong/Documents/MI_uncertainty/trrust_rawdata.mouse.tsv", Symbol_Gene, name_index)
    # interactions += ReadKEGGlinks("/home/cong/Documents/MI_uncertainty/kegg_pathway_interaction_mouse.txt", gene_names, name_index)
 
    log_counts = np.load("/home/cong/Documents/MI_uncertainty/10xmouse_brain/alevin/seurat_selection_count.npy")
    log_counts /= (log_counts.sum(axis=0, keepdims=True) / 1e3)
    log_counts[np.where(log_counts == 0)] = -25
    log_counts[np.where(log_counts > 0)] = np.log(log_counts[np.where(log_counts > 0)])
    percentages = np.sum(log_counts < -24.5, axis=1) / log_counts.shape[1]

    plot_scatter_hist(mi, percentages, iu, "/home/cong/Documents/MI_uncertainty/10xmouse_brain/alevin/mi_distribution.png")

    # mouse heart
    mi = read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xmouse_heart/alevin/raw_mi_10.dat")
    corrmi = read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xmouse_heart/alevin/corr_mi_10_0.dat")
    corrmi += read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xmouse_heart/alevin/corr_mi_10_1.dat")
    iu = np.triu_indices(mi.shape[0])

    fp = open("/home/cong/Documents/MI_uncertainty/10xmouse_heart/alevin/seurat_selection_genenames.txt", 'r')
    gene_names = [line.strip() for line in fp.readlines()]
    fp.close()
    name_index = {gene_names[i]:i for i in range(len(gene_names))}

    ori_pairs = read_similar_genes_ensembl("/home/cong/Documents/MI_uncertainty/ensembl_mouse_paralog", name_index)
    utr_sim = read_similar_genes_jaccardonly("/home/cong/Documents/MI_uncertainty/nucmer.gencode.vM23.transcript.utr_geneutr_similarity.txt", name_index)
    pairs = set(ori_pairs) & set(utr_sim)
    # pairs = read_similar_genes("/home/cong/Documents/alevin/alevin/gencode_M23_gene_similarity_selected.txt", name_index)
    # pairs = set(pairs)

    for quantile in [99.9, 99.99, 99.999]:
        idx1, idx2 = np.where(mi > np.percentile(mi[iu], quantile))
        overlap_raw =  len( pairs & set([(idx1[i], idx2[i]) for i in range(len(idx1))]))
        v2_idx1, v2_idx2 = np.where(corrmi > np.percentile(corrmi[iu], quantile))
        overlap_corr = len( pairs & set([(v2_idx1[i], v2_idx2[i]) for i in range(len(v2_idx2))])) 
        agreement = len(set([(idx1[i], idx2[i]) for i in range(len(idx1))]) & set([(v2_idx1[i], v2_idx2[i]) for i in range(len(v2_idx2))]) ) / len(idx1)
        print("quantile = {}\tagreement = {}\traw = {}\tcorr = {}".format( quantile, agreement, overlap_raw, overlap_corr ) )


    log_counts = np.load("/home/cong/Documents/MI_uncertainty/10xmouse_heart/alevin/seurat_selection_count.npy")
    log_counts /= (log_counts.sum(axis=0, keepdims=True) / 1e3)
    log_counts[np.where(log_counts == 0)] = -25
    log_counts[np.where(log_counts > 0)] = np.log(log_counts[np.where(log_counts > 0)])
    percentages = np.sum(log_counts < -24.5, axis=1) / log_counts.shape[1]

    plot_scatter_hist(mi, percentages, iu, "/home/cong/Documents/MI_uncertainty/10xmouse_heart/alevin/mi_distribution.png")

    # human
    # load all data
    mi = read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xhuman_pbmc/raw_mi_10.dat")
    corrmi = read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xhuman_pbmc/corr_mi_10_0.dat")
    corrmi += read_eigen_matrix("/home/cong/Documents/MI_uncertainty/10xhuman_pbmc/corr_mi_10_1.dat")
    iu = np.triu_indices(mi.shape[0])

    log_counts = np.load("/home/cong/Documents/MI_uncertainty/10xhuman_pbmc/alevin/seurat_selection_count.npy")
    log_counts /= (log_counts.sum(axis=0, keepdims=True) / 1e3)
    log_counts[np.where(log_counts == 0)] = -25
    log_counts[np.where(log_counts > 0)] = np.log(log_counts[np.where(log_counts > 0)])
    percentages = np.sum(log_counts < -24.5, axis=1) / log_counts.shape[1]
    means = np.zeros(log_counts.shape[0])
    medians = np.zeros(log_counts.shape[0])
    for i in range(log_counts.shape[0]):
        means[i] = np.mean(log_counts[i, np.where(log_counts[i,:] > -14.5)[0]] )
        medians[i] = np.median( log_counts[i, np.where(log_counts[i,:] > -14.5)[0]] )

    fp = open("/home/cong/Documents/MI_uncertainty/10xhuman_pbmc/alevin/seurat_selection_genenames.txt", 'r')
    gene_names = [line.strip() for line in fp.readlines()]
    fp.close()
    name_index = {gene_names[i]:i for i in range(len(gene_names))}

    ori_pairs = read_similar_genes_ensembl("/home/cong/Documents/MI_uncertainty/ensembl_human_paralog", name_index)
    utr_sim = read_similar_genes_jaccardonly("/home/cong/Documents/MI_uncertainty/nucmer.gencode.v26.involved.transcript.utr_geneutr_similarity.txt", name_index)
    pairs = set(ori_pairs) & set(utr_sim)
    # pairs = read_similar_genes("/home/cong/Documents/MI_uncertainty/nucmer_gencode_v26_gene_similarity.txt", name_index)
    # pairs = set(pairs)

    for quantile in [99.9, 99.99, 99.999]:
        idx1, idx2 = np.where(mi > np.percentile(mi[iu], quantile))
        overlap_raw =  len( pairs & set([(idx1[i], idx2[i]) for i in range(len(idx1))]))
        v2_idx1, v2_idx2 = np.where(corrmi > np.percentile(corrmi[iu], quantile))
        overlap_corr = len( pairs & set([(v2_idx1[i], v2_idx2[i]) for i in range(len(v2_idx2))])) 
        agreement = len(set([(idx1[i], idx2[i]) for i in range(len(idx1))]) & set([(v2_idx1[i], v2_idx2[i]) for i in range(len(v2_idx2))]) ) / len(idx1)
        print("quantile = {}\tagreement = {}\traw = {}\tcorr = {}".format( quantile, agreement, overlap_raw, overlap_corr ) )

    # plot example of top raw/corrected mutual information
    fig_mi, fig_corr = plot_example_setdiff(mi, corrmi, iu, log_counts)
    fig_mi.savefig("/home/cong/Documents/MI_uncertainty/10xhuman_pbmc/alevin/example_top_rawmi.png", transparent=True, bbox_inches='tight')
    fig_corr.savefig("/home/cong/Documents/MI_uncertainty/10xhuman_pbmc/alevin/example_top_corrmi.png", transparent=True, bbox_inches='tight')

    # plot scatter plot + histogram
    plot_scatter_hist(mi, percentages, iu, "/home/cong/Documents/MI_uncertainty/10xhuman_pbmc/alevin/mi_distribution.png")