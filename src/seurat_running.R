library(tximport)
library(Seurat)

args <- commandArgs(trailingOnly = TRUE)
alevin_quant_file <- args[1]
gene_type_file <- args[2]
# retrieve the quantification directory
alevin_quant_dir <- unlist(strsplit(alevin_quant_file, "/"))
alevin_quant_dir <- alevin_quant_dir[1:(length(alevin_quant_dir)-1)]
alevin_quant_dir <- paste(alevin_quant_dir, collapse = "/")

txi <- tximport(alevin_quant_file, type="alevin")
gene_types <- read.table(gene_type_file, header=F, sep="\t")
colnames(gene_types) <- c("gene_name", "gene_type")

# select only pc genes
pc_gene_names <- gene_types[gene_types$gene_type == "protein_coding", "gene_name"]
txi$counts <- txi$counts[rownames(txi$counts) %in% pc_gene_names, ]

# seurat
mouse <- CreateSeuratObject(txi$counts , min.cells = 3, min.features = 200, project = "10X_MOUSE")
mouse <- NormalizeData(mouse, normalization.method = "LogNormalize", scale.factor = 10000)
mouse <- FindVariableFeatures(mouse, selection.method = "vst", nfeatures = 2000)
all.genes <- rownames(mouse)
mouse <- ScaleData(mouse, features = all.genes)

# save count matrix of the selected cells and selected genes to file
write.table(mouse@assays$RNA@counts, paste0(alevin_quant_dir, "/seurat_selection_count.txt", sep=""), append=F, quote=F, sep="\t", row.names=F, col.names=F)
write.table(rownames(mouse@assays$RNA@counts), paste0(alevin_quant_dir, "/seurat_selection_genenames.txt", sep=""), append=F, quote=F, sep="\t", row.names=F, col.names=F)
