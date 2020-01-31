#!/bin/python

import sys
import numpy as np
from pathlib import Path
import subprocess


def GetFeature(line, key):
    s = line.index(key)
    t = line.index(";", s+1)
    return line[(s+len(key)+2):(t-1)]


def Map_Gene_Trans(gtffile):
    GeneTransMap = {}
    TransGeneMap = {}
    fp = open(gtffile, 'r')
    for line in fp:
        if line[0] == '#':
            continue
        strs = line.strip().split("\t")
        if strs[2] != "transcript":
            continue
        trans_id = GetFeature(line, "transcript_id")
        gene_id = GetFeature(line, "gene_id")
        # add the mapping between gene and transcript
        TransGeneMap[trans_id] = gene_id
        if gene_id in GeneTransMap:
            GeneTransMap[gene_id].append( trans_id )
        else:
            GeneTransMap[gene_id] = [trans_id]
    fp.close()
    return GeneTransMap, TransGeneMap


def Read3UTR(gtffile):
    transcript_utrs = {}
    transcript_exons = {}
    transcript_chr = {}
    transcript_strand = {}
    fp = open(gtffile, 'r')
    for line in fp:
        if line[0] == '#':
            continue
        strs = line.strip().split("\t")
        if strs[2] == "transcript":
            trans_id = GetFeature(line, "transcript_id")
            transcript_strand[trans_id] = (strs[6] == "+")
            transcript_chr[trans_id] = strs[0]
        elif strs[2] == "UTR":
            trans_id = GetFeature(line, "transcript_id")
            # adding all UTRs regardless whether it is 3' or 5'
            if trans_id in transcript_utrs:
                transcript_utrs[trans_id].append( (int(strs[3])-1, int(strs[4])) )
            else:
                transcript_utrs[trans_id] = [ (int(strs[3])-1, int(strs[4])) ]
        elif strs[2] == "exon":
            trans_id = GetFeature(line, "transcript_id")
            if trans_id in transcript_exons:
                transcript_exons[trans_id].append( (int(strs[3])-1, int(strs[4])) )
            else:
                transcript_exons[trans_id] = [(int(strs[3])-1, int(strs[4]))]
    fp.close()
    # filtering out 5' utrs
    for t,v in transcript_utrs.items():
        v.sort()
        # assert that each utr interval is not overlap
        for i in range(1, len(v)):
            if v[i-1][1] > v[i][0]:
                print( (t, v) )
            assert(v[i-1][1] <= v[i][0])
        exons = transcript_exons[t]
        exons.sort()
        strand = transcript_strand[t]
        utr_3 = []
        if strand:
            # reading exons and utrs from backward, and stop reading when either (1) utr within exon and exon start is before utr start (2) utr outside exon
            for i in range(min(len(exons), len(v))):
                this_e = exons[len(exons) - 1 - i]
                this_u = v[len(v) - 1 - i]
                if max(this_e[0], this_u[0]) < min(this_e[1], this_u[1]):
                    utr_3.append( this_u )
                    if this_e[0] < this_u[0]:
                        break
                else:
                    break
        else:
            # reading exons and utrs from forward, and stop reading when either (1) utr within exon and right side of exon is behind right side of utr (2) utr outside exon
            for i in range(min(len(exons), len(v))):
                this_e = exons[i]
                this_u = v[i]
                if max(this_e[0], this_u[0]) < min(this_e[1], this_u[1]):
                    utr_3.append( this_u )
                    if this_e[1] > this_u[1]:
                        break
                else:
                    break
        transcript_utrs[t] = utr_3
    return transcript_utrs, transcript_chr, transcript_strand


Nucleotide={'A':'T', 'C':'G', 'G':'C', 'T':'A', 'R':'Y', 'Y':'R', 'S':'W', 'W':'S', 'K':'M', 'M':'K', 'B':'V', 'V':'B', 'D':'H', 'H':'D', 'N':'N', '.':'.', '-':'-'}

def ReverseComplement(seq):
    rcseq = "".join([Nucleotide[x] for x in seq])
    return rcseq[::-1]


def ReadGenome(fafile):
    genome={}
    fp=open(fafile,'r')
    tmpseq=''
    tmpname=''
    for line in fp:
        if line[0]=='>':
            if len(tmpseq)!=0:
                print(tmpname)
                genome[tmpname]=tmpseq
            tmpseq=''
            strs=line.split()
            tmpname=strs[0][1:]
        else:
            tmpseq+=line.strip()
    genome[tmpname]=tmpseq
    fp.close()
    return genome


def WriteFlattenGene(outputfile, gene_exons, gene_chr, gene_strand, genome, threshold = 10):
    fp = open(outputfile, 'w')
    for g,v in gene_exons.items():
        if np.sum([e[1] - e[0] for e in v]) < threshold:
            continue
        this_chr = gene_chr[g]
        this_strand = gene_strand[g]
        seq = ""
        for e in v:
            seq += genome[this_chr][e[0]:e[1]]
        if not this_strand:
            seq = ReverseComplement(seq)
        # write to file
        fp.write(">" + g + "\n")
        count = 0
        while count < len(seq):
            fp.write( seq[count:min(len(seq),count+70)] + "\n")
            count += 70
    fp.close()


def WriteUTRLength(outputfile, transcript_utrs):
    fp = open(outputfile, 'w')
    fp.write("# transcript\tutr_length\n")
    for t,v in transcript_utrs.items():
        length = np.sum([e[1] - e[0] for e in v])
        fp.write("{}\t{}\n".format(t, length))
    fp.close()


def ReadNucmerLength(filename):
    aligned_blocks = {}
    fp = open(filename, 'r')
    linecount = 0
    for line in fp:
        linecount += 1
        if linecount <= 5:
            continue
        strs = line.strip().split("|")
        genes = [x.replace(" ","") for x in strs[4].split("\t") if x != ""]
        assert( len(genes) == 2 )
        # don't keep the same-gene alignment
        if genes[0] == genes[1]:
            continue
        block1 = [int(x) for x in strs[0].split(" ") if x != ""]
        block1.sort()
        block2 = [int(x) for x in strs[1].split(" ") if x != ""]
        block2.sort()
        # add to result
        if tuple(genes) in aligned_blocks:
            aligned_blocks[ tuple(genes) ].append( block1 )
        else:
            aligned_blocks[ tuple(genes) ] = [ block1 ]
        genes = genes[::-1]
        if tuple(genes) in aligned_blocks:
            aligned_blocks[ tuple(genes) ].append( block2 )
        else:
            aligned_blocks[ tuple(genes) ] = [ block2 ]
    fp.close()
    # sort and make unique of the aligned blocks
    for k,v in aligned_blocks.items():
        v.sort(key = lambda x:(x[0],x[1]))
        merged_v = [v[0]]
        for e in v[1:]:
            if e[0] <= merged_v[-1][1]:
                merged_v[-1][1] = max(merged_v[-1][1], e[1])
            else:
                merged_v.append( e )
        aligned_blocks[k] = merged_v
    # sum to a single overlapping length
    aligned_length = {}
    for k,v in aligned_blocks.items():
        sorted_k = tuple(sorted([k[0], k[1]]))
        this_len = np.sum([x[1] - x[0] for x in v])
        if (not sorted_k in aligned_length) or (aligned_length[sorted_k] > this_len):
            aligned_length[sorted_k] = this_len
    return aligned_length


def WriteGeneSimilarity(aligned_length, gene_length, outputfile):
    fp = open(outputfile, 'w')
    fp.write("# gene1\tgene2\tlength1\tlength2\toverlap_length\tjaccard\n")
    for k,v in aligned_length.items():
        g1, g2 = k
        jaccard = v / (gene_length[g1] + gene_length[g2] - v)
        if jaccard - 1 > 1e-3:
            print( (k, jaccard) )
            break
        fp.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(g1, g2, gene_length[g1], gene_length[g2], v, jaccard) )
    fp.close()


def WriteGeneSimilarity_jaccardonly(jaccard, outputfile):
    fp = open(outputfile, 'w')
    fp.write("# name1\tname2\tjaccard\n")
    for p,v in jaccard.items():
        fp.write("{}\t{}\t{}\n".format(p[0], p[1], v))
    fp.close()


def ReadSimilarityJaccard(filename):
    jaccard = {}
    fp = open(filename, 'r')
    for line in fp:
        if line[0] == '#':
            continue
        strs = line.strip().split("\t")
        jaccard[(strs[0], strs[1])] = float(strs[5])
    fp.close()
    return jaccard


def ConvertGene_max(trans_jaccard, TransGeneMap):
    gene_jaccard = {}
    for p,v in trans_jaccard.items():
        assert( (p[0] in TransGeneMap) and (p[1] in TransGeneMap) )
        gene_pairs = (TransGeneMap[p[0]], TransGeneMap[p[1]])
        if gene_pairs[0] == gene_pairs[1]:
            continue
        if (not gene_pairs in gene_jaccard) or (gene_jaccard[gene_pairs] < v):
            gene_jaccard[gene_pairs] = v
    return gene_jaccard


def ReadParalogGene_ensembl(filename, GeneTransMap):
    # map from the gene id without version number to the gene id with version number
    version_map = {k.split(".")[0]:k for k in GeneTransMap.keys()}
    gene_list = []
    # read file
    fp = open(filename, 'r')
    for line in fp:
        if line[0] == '#':
            continue
        strs = line.strip().split("\t")
        if (strs[0] in version_map) and (strs[2] in version_map):
            gene_list += [version_map[strs[0]], version_map[strs[2]]]
    fp.close()
    gene_list = list(set(gene_list))
    return gene_list


if __name__=="__main__":
    if len(sys.argv) == 1:
        print("python gene_similarity.py <gtffile> <genomefasta> <out_utrseq> <out_utrlength> <out_nucmerprefix>")
    else:
        gtffile = sys.argv[1]
        genomefasta = sys.argv[2]
        out_utrseq= sys.argv[3]
        out_utrlength = sys.argv[4]
        out_nucmerprefix = sys.argv[5]

        transcript_utrs, transcript_chr, transcript_strand = Read3UTR(gtffile)
        # a special filtering for human paralog
        GeneTransMap, TransGeneMap = Map_Gene_Trans(gtffile)
        ensembl_gene_list = ReadParalogGene_ensembl("/home/congm1/savanna/savannacong37/Code/mi_test/ensembl_human_paralog", GeneTransMap)
        involved_trans = set(sum([GeneTransMap[g] for g in ensembl_gene_list], []))
        transcript_utrs = {t:v for t,v in transcript_utrs.items() if t in involved_trans}
        transcript_chr = {t:v for t,v in transcript_chr.items() if t in involved_trans}
        transcript_strand = {t:v for t,v in transcript_strand.items() if t in involved_trans}
        # end special filtering
        utr_length = {t:np.sum([x[1]-x[0] for x in v]) for t,v in transcript_utrs.items()}
        if not Path(out_utrseq).exists():
            genome = ReadGenome(genomefasta)
            WriteFlattenGene(out_utrseq, transcript_utrs, transcript_chr, transcript_strand, genome)
            WriteUTRLength(out_utrlength, transcript_utrs)

        # run nucmer
        if not Path(out_nucmerprefix + "_selfalign.delta").exists():
            p = subprocess.Popen("nucmer -t 8 -c 10 --maxmatch --nosimplify -p {}_selfalign {} {}".format(out_nucmerprefix, out_utrseq, out_utrseq), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
        if not Path(out_nucmerprefix + "_selfalign.txt").exists():
            p = subprocess.Popen("show-coords {}_selfalign.delta > {}_selfalign.txt".format(out_nucmerprefix, out_nucmerprefix), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()

        # read nucmer file and calculate similarity
        if not Path(out_nucmerprefix + "_transutr_similarity.txt").exists():
            aligned_length = ReadNucmerLength(out_nucmerprefix + "_selfalign.txt")
            WriteGeneSimilarity(aligned_length, utr_length, out_nucmerprefix + "_transutr_similarity.txt")

        # convert to gene
        if not Path(out_nucmerprefix + "_geneutr_similarity.txt").exists():
            GeneTransMap, TransGeneMap = Map_Gene_Trans(gtffile)
            trans_jaccard = ReadSimilarityJaccard(out_nucmerprefix + "_transutr_similarity.txt")
            gene_jaccard = ConvertGene_max(trans_jaccard, TransGeneMap)
            WriteGeneSimilarity_jaccardonly(gene_jaccard, out_nucmerprefix + "_geneutr_similarity.txt")
