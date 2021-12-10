## =========================================
# evaluation scripts to:
# take in best performing model in Polarbear co-assay and Polarbear models
# calculate and plot:
# - gene-wise correlation
# - differentially expressed genes
# - peak-wise AUROC and AUPRnorm 
# - FOSCTTM
# we evaluate on SNARE-seq study's cell type specific genes and peaks since they are most important and is more memory efficient 
## =========================================

library(data.table)
library(readxl)
library(ROCR)
library(Matrix)
library(scran)
library(ggplot2)
library(ggallin)

cur_dir <- './'
source(paste0(cur_dir, "/bin/evaluation_functions.R"))
data_dir <- 'https://noble.gs.washington.edu/~ranz0/Polarbear/'

## TODO: select best hyperparameters based on gene-wise correlation on the validation set, as an example here we just use the following default hyperparameters
ndim <- "25_25"
klweight <- "1"


## ============================================================
## generate normalized gene expression based on Lun et al. 2016, so that we could compare the predicted value with these "true" deconvolution-normalized profiles
## ============================================================
snareseq_rna_mat <- readMM(paste0(data_dir, "/data/adultbrainfull50_rna_outer_snareseq.mtx"))
snareseq_rna_barcodes <- fread(paste0(data_dir, "/data/adultbrainfull50_rna_outer_snareseq_barcodes.tsv"), header=T)$index
snareseq_rna_mat_tmp <- t(as.matrix(snareseq_rna_mat))
snareseq_rna_mat_libsize <- apply(snareseq_rna_mat, 1, sum)
snareseq_rna_mat_deconv_sizefactor <- computeSumFactors(snareseq_rna_mat_tmp)
snareseq_rna_mat_deconv_norm <- t(snareseq_rna_mat_tmp)/snareseq_rna_mat_deconv_sizefactor
rm(snareseq_rna_mat_tmp)
rm(snareseq_rna_mat)

snareadult_barcodes <- readRDS(paste0(data_dir, "/data/AdBrainCortex_SNAREseq_metadata.rds"))
snareadult_barcodes$Barcode <- row.names(snareadult_barcodes)
snareadult_barcodes <- as.data.table(snareadult_barcodes)

## ============================================================
## import cell type markers and previously calculated differentially expressed genes
## determine overlapping genes and peaks
## ============================================================
## cell-type specific gene markers used in the SNARE paper (Chen et al. 2019, based on Supplementary Table 1 and Supplementary figure 14)
celltype_mapping <- data.table(rbind(
  c("Ast","Ast", "Apoe"),
  c("Ast","Ast", "Slc1a3"),
  c("Claustrum","Clau", "Nr4a2"),
  c("Endo","Endo", "Kdr"),
  c("Ex-L2/3-Rasgrf2","E2Rasgrf2","Rasgrf2"),
  c("Ex-L3/4-Rmst","E3Rmst", "Rmst"),
  c("Ex-L3/4-Rorb","E3Rorb", "Rorb"),
  c("Ex-L4/5-Il1rapl2","E4Il1rapl2", "Il1rapl2"),
  c("Ex-L4/5-Thsd7a","E4Thsd7a", "Thsd7a"),
  c("Ex-L5-Galnt14","E5Galnt14", "Galnt14"),
  c("Ex-L5-Parm1","E5Parm1", "Parm1"),
  c("Ex-L5/6-Sulf1","E5Sulf1", "Sulf1"),
  c("Ex-L5/6-Tshz2","E5Tshz2", "Tshz2"),
  c("Ex-L6-Tle4","E6Tle4", "Tle4"),
  c("In-Npy","InN", "Npy"),
  c("In-Pvalb","InP", "Pvalb"),
  c("In-Sst","InS", "Sst"),
  c("In-Vip","InV", "Vip"),
  c("Oli-Itpr2","OliI", "Itpr2"),
  c("Oli-Mal","OliM", "Mal"),
  c("OPC","OPC", "Vcan"),
  c("OPC","OPC", "Sox6"),
  c("OPC","OPC", "Pdgfra"),
  c("Peri","Peri", "Rgs5"),
  c("Peri","Peri", "Vtn"),
  c("Mic","Mic", "Apbb1ip")
))
# Mis:cells of miscellaneous clusters.
names(celltype_mapping) <- c("Cluster","Ident", "marker")

snareadult_diffgenes <- read_excel(paste0(data_dir, "/data/NIHMS1539957-supplement-sup_tab1.xlsx"), sheet = "Adult_cerebral_cortex", skip=3)
#table(snareadult_diffgenes$Cluster)
snareadult_data <- merge(snareadult_diffgenes, celltype_mapping, by="Cluster")
snareadult_data <- merge(snareadult_data, snareadult_barcodes, by="Ident")

snareadult_marker_data <- merge(snareadult_barcodes, celltype_mapping, by="Ident")
snareadult_marker_data <- snareadult_marker_data[!is.na(snareadult_marker_data$marker),]

## import scATAC diff peaks
for (celltype in celltype_mapping$Cluster){
  print(celltype)
  if (celltype=="Ast"){
    snareadult_diffpeaks <- read_excel(paste0(data_dir, "/data/NIHMS1539957-supplement-sup_tab3.xlsx"), sheet = gsub("/","",celltype), skip=3)
    snareadult_diffpeaks <- as.data.table(snareadult_diffpeaks)
    snareadult_diffpeaks$celltype <- celltype
  }else{
    snareadult_diffpeaks_i <- read_excel(paste0(data_dir, "/data/NIHMS1539957-supplement-sup_tab3.xlsx"), sheet = gsub("/","",celltype))
    snareadult_diffpeaks_i <- as.data.table(snareadult_diffpeaks_i)
    snareadult_diffpeaks_i$celltype <- celltype
    snareadult_diffpeaks <- rbind(snareadult_diffpeaks, snareadult_diffpeaks_i)
  }
}
snareadult_diffpeaks$peaks <- paste0(snareadult_diffpeaks$chrom,":",snareadult_diffpeaks$start, "-", snareadult_diffpeaks$end)

## to make a fair comparison with BABEL, only include genes that are predicted in both methods
polarbear_genes <- fread(paste0(data_dir, "/data/adultbrainfull50_rna_outer_genes.txt"), header=F)$V1
babel_genes <- fread(paste0(data_dir, "/data/babel_genes.txt"), header=F)$V1
overlapping_genes <- intersect(polarbear_genes, babel_genes)


## ============================================================
## gene-wise pearson correlation
## ============================================================
## plot pairwise comparison on differentially expressed genes

gene_signature_ver <- "diffexp"
cor_method <- 'pearson'
for (split_ver in c("babel", "random")){
  if (gene_signature_ver=="marker"){
    gene_signature <- celltype_mapping$marker[!is.na(celltype_mapping$marker)] # marker genes for each cell type
  }else if (gene_signature_ver=="diffexp"){
    if (split_ver=="random"){
      gene_signature <- unique(snareadult_diffgenes$Gene) # all genes differentially expressed across cell types
    }else{
      gene_signature <- unique(snareadult_diffgenes[snareadult_diffgenes$Cluster=="Ex-L2/3-Rasgrf2",]$Gene) # genes differentially expressed in Ex-L2 cell type, which is the major cell type of BABEL"s test set
    }
  }else{
    gene_signature <- c() # use all the genes
  }
  
  npos_cutoff = 0
  pred_prefix_coassay <- paste0(cur_dir, "/output_coassay_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")
  pred_prefix_semi <- paste0(cur_dir, "/output_semi_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")
  
  # compare correlation (calculated against size-normalized original expression)
  plot_cor_compare(cur_dir, data_dir, split_ver, gene_signature_ver, gene_signature, npos_cutoff, pred_prefix_coassay, pred_prefix_semi)
  # compare correlation (calculated against Lun et al. deconvolution-normalized original expression)
  plot_cor_compare_deconv(cur_dir, split_ver, snareseq_rna_mat_deconv_norm, snareseq_rna_barcodes, gene_signature_ver, gene_signature, npos_cutoff, pred_prefix_coassay, pred_prefix_semi, cor_method, snareseq_rna_mat_libsize, polarbear_genes, babel_genes, pred_prefix_babel="")
}



## ============================================================
## peak-wise AUROC and AUPRnorm
## ============================================================
peaks <- fread(paste0(data_dir, "/data/adultbrainfull50_atac_outer_peaks_noXY_diffexp.txt"), header=T) ## differentially expressed autochromosome peaks (whose auroc and auprnorm is )

for (split_ver in c('random','babel')){
  pred_prefix_coassay <- paste0(cur_dir, "/output_coassay_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")
  pred_prefix_semi <- paste0(cur_dir, "/output_semi_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")
  barcodes <- fread(paste0(pred_prefix_coassay, "_test_barcodes.txt"), header=F)$V1
  for (metric in c("auc", "auprc")){
    plor_auc_aupr_compare(split_ver, metric, length(barcodes), peaks, snareadult_diffpeaks, pred_prefix_coassay, pred_prefix_semi)
  }
}




## ============================================================
## plot FOSCTTM
## ============================================================
for (split_ver in c("random","babel")){
  pred_prefix_coassay <- paste0(cur_dir, "/output_coassay_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")
  pred_prefix_semi <- paste0(cur_dir, "/output_semi_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")
  
  foscttm_coassay <- load_foscttm_matrix(pred_prefix_coassay)
  foscttm_coassay$model <- "Polarbear co-assay"
  foscttm_semi <- load_foscttm_matrix(pred_prefix_semi)
  foscttm_semi$model <- "Polarbear"
  foscttm <- rbind(foscttm_coassay, foscttm_semi)
  foscttm$model <- factor(foscttm$model, levels =c("Polarbear co-assay","Polarbear"))
  foscttm$query <- as.factor(foscttm$query)
  
  for (query_i in c("RNA","ATAC")){
    if (split_ver == "random"){
      foscttm_title = paste0("sc", query_i, "-seq, random 20%")
    }else{
      foscttm_title = paste0("sc", query_i, "-seq, unseen cell type")
    }
    p1_foscttm <- ggplot(data=foscttm[query==query_i], aes(x=Cell, y=FOSCTTM, color=model)) + 
      ggtitle(foscttm_title)+
      geom_line(stat="identity", size=1)+ 
      scale_colour_manual(values = c("#bdbdbd","#1b9e77","#d95f02"))+
      theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
      theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=12, colour="black"))+
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"))+ theme(legend.text=element_text(size=11), legend.title=element_text(size=11))+
      theme(legend.position="none")+
      theme(plot.margin = margin(10, 10, 10, 10))
    png(paste0(cur_dir, "/result/eval_", split_ver,"_foscttm", query_i, ".png"), width = 350, height = 350, res=110)
    print(p1_foscttm)
    dev.off()
  }
}




## ============================================================
## cell type classification through marker gene predictions
## ============================================================
split_ver = "random"
marker_ver <- "marker"
if (marker_ver == "marker"){
  marker_list <- celltype_mapping$marker[!is.na(celltype_mapping$marker)]
}else if (marker_ver == "diffexp"){
  marker_list <- unique(snareadult_diffgenes$Gene)
}

pred_prefix_coassay <- paste0(cur_dir, "/output_coassay_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")
pred_prefix_semi <- paste0(cur_dir, "/output_semi_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")

auc_mat_combined <- c()
for (model in c("Polarbear","Polarbear co-assay")){
  if (model=="Polarbear"){
    pred_prefix = pred_prefix_semi
  }else if (model=="Polarbear co-assay"){
    pred_prefix = pred_prefix_coassay
  }
  pred_mat <- fread(paste0(pred_prefix, "_test_rnanorm_pred.txt"), header=F)
  matched_genes <- match(overlapping_genes, polarbear_genes)
  matched_genes <- matched_genes[!is.na(matched_genes)]
  pred_mat <- pred_mat[, matched_genes, with=FALSE]
  barcodes <- fread(paste0(pred_prefix, "_test_barcodes.txt"), header=F)$V1
  true_mat <- snareseq_rna_mat_deconv_norm[match(barcodes, snareseq_rna_barcodes),]
  true_mat <- as.data.table(true_mat[, matched_genes])
  
  for (markeri in marker_list){
    if (markeri %in% overlapping_genes){
      if (marker_ver=="marker"){
        Cluster <- unique(celltype_mapping[marker==markeri]$Cluster)
        matched_barcode <- match(snareadult_marker_data[marker==markeri]$Barcode, barcodes)
        matched_barcode <- matched_barcode[!is.na(matched_barcode)]
      }else{
        Cluster <- unique(celltype_mapping[marker==markeri]$Cluster)
        matched_barcode <- match(snareadult_data[snareadult_data$Gene==markeri,]$Barcode, barcodes)
        matched_barcode <- matched_barcode[!is.na(matched_barcode)]
      }
      if (length(matched_barcode)>0){
        auc_mat_combined <- rbind(auc_mat_combined, c(plot_marker_exp(markeri, pred_mat[, match(markeri, overlapping_genes), with=FALSE], true_mat[, match(markeri, overlapping_genes), with=FALSE], matched_barcode), model))
      }
    }
  }
}

auc_mat_combined <- as.data.table(auc_mat_combined)
names(auc_mat_combined) <- c("marker","AUC_true","AUC_pred","npos", "model")
auc_mat_combined$AUC_pred <- as.numeric(auc_mat_combined$AUC_pred)
auc_mat_combined$AUC_true <- as.numeric(auc_mat_combined$AUC_true)
auc_mat_combined$npos <- as.numeric(auc_mat_combined$npos)
auc_mat_combined <- auc_mat_combined[npos>8]
auc_mat_combined_pairwise <- as.data.table(cbind(auc_mat_combined[model == "Polarbear co-assay"]$AUC_pred, 
                                                 auc_mat_combined[model == "Polarbear"]$AUC_pred, 
                                                 auc_mat_combined[model == "Polarbear co-assay"]$npos))
names(auc_mat_combined_pairwise) <- c("Polarbear co-assay", "Polarbear","npos")
apply(auc_mat_combined_pairwise, 2, mean)
apply(auc_mat_combined_pairwise, 2, median)
count_diagonal = table(auc_mat_combined_pairwise$`Polarbear`>auc_mat_combined_pairwise$`Polarbear co-assay`)
pval = wilcox.test(auc_mat_combined_pairwise$`Polarbear co-assay`, auc_mat_combined_pairwise$Polarbear, alternative = "less")$p.value
p2_auc_compare <- ggplot(auc_mat_combined_pairwise, aes(x = `Polarbear co-assay`, y = Polarbear)) + 
  geom_point(size = 2, alpha=.8, color="#2c7bb6")+ ggtitle("AUROC of marker genes")+
  theme_classic()+
  theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
  theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))+
  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
  annotate("text", y = min(auc_mat_combined_pairwise$`Polarbear`)+0.04, x = max(auc_mat_combined_pairwise$`Polarbear co-assay`) -0.06 , label = count_diagonal[1])+
  annotate("text", y = max(auc_mat_combined_pairwise$`Polarbear`)-0.04, x = min(auc_mat_combined_pairwise$`Polarbear co-assay`) +0.04 , label = count_diagonal[2])+
  annotate("text", x = max(auc_mat_combined_pairwise$`Polarbear co-assay`)-0.07, y = min(auc_mat_combined_pairwise$`Polarbear`) +0.02 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))

png(paste0(cur_dir, "/result/eval_random_compare_normexp_",marker_ver,"_auc_scatterplot.png"), width = 330, height = 350, res=110)
print(p2_auc_compare)
dev.off()

print(apply(auc_mat_combined_pairwise, 2, median))



## ============================================================
## differentailly expressed genes in unseen cell type
## ============================================================
## check AUPRC of differentially expressed genes between unseen cell type (test set) and training set
split_ver = "babel"
pred_prefix_coassay <- paste0(cur_dir, "/output_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")
pred_prefix_semi <- paste0(cur_dir, "/output_semi_gpu/polarbear_", split_ver, "_genebatch_2l_lr0.0001_0.001_0.001_0.001_dropout0.1_ndim", ndim, "_batch16_linear_improvement45_nwarmup_400_80_klstart0_0_klweight", klweight, "_hiddenfrac2")

#eval_mat_combined <- compare_model_performance(split_ver)
barcodes <- fread(paste0(cur_dir, "/data/babel_test_barcodes.txt"), header=F)$V1

## based on original expression
matched_genes <- match(overlapping_genes, polarbear_genes)
matched_genes <- matched_genes[!is.na(matched_genes)]
diffexp_ori <- calculate_diffexp(data.matrix(snareseq_rna_mat_deconv_norm[snareseq_rna_barcodes %in% barcodes, matched_genes]), 
                                 data.matrix(snareseq_rna_mat_deconv_norm[!snareseq_rna_barcodes %in% barcodes, matched_genes]), method="pval")
diffexp_ori_fdr = p.adjust(diffexp_ori, method = "BH")

diffexp_ori_logFC <- calculate_diffexp(data.matrix(snareseq_rna_mat_deconv_norm[snareseq_rna_barcodes %in% barcodes, matched_genes]), 
                                       data.matrix(snareseq_rna_mat_deconv_norm[!snareseq_rna_barcodes %in% barcodes, matched_genes]), method="logFC")

## based on best Polarbear co-assay prediction
pred_mat <- fread(paste0(pred_prefix_coassay, "_test_rnanorm_pred.txt"), header=F)
pred_train_mat <- fread(paste0(pred_prefix_coassay, "_train_rnanorm_pred.txt"), header=F)
diffexp_polarbear_coassay <- calculate_diffexp(data.matrix(pred_mat[, matched_genes, with=FALSE]), 
                                               data.matrix(pred_train_mat[, matched_genes, with=FALSE]), method="pval")
diffexp_polarbear_coassay_fdr <- p.adjust(diffexp_polarbear_coassay, method = "BH")

## based on best Polarbear prediction
pred_mat <- fread(paste0(pred_prefix_semi, "_test_rnanorm_pred.txt"), header=F)
pred_train_mat <- fread(paste0(pred_prefix_semi, "_train_rnanorm_pred.txt"), header=F)
diffexp_polarbear <- calculate_diffexp(data.matrix(pred_mat[, matched_genes, with=FALSE]), 
                                       data.matrix(pred_train_mat[, matched_genes, with=FALSE]), method="pval")
diffexp_polarbear_fdr <- p.adjust(diffexp_polarbear, method = "BH")

diffexp_fdr_combined_mat <- cbind(diffexp_ori_logFC, diffexp_ori_fdr, diffexp_polarbear_coassay, diffexp_polarbear)
diffexp_fdr_combined_mat <- as.data.table(diffexp_fdr_combined_mat)
diffexp_fdr_combined_mat$gene <- overlapping_genes
diffexp_fdr_combined_mat <- diffexp_fdr_combined_mat[!is.na(diffexp_ori_logFC)]

fdr_cutoff <- 0.01
diffexp_fdr_combined_mat$label <- 0
diffexp_fdr_combined_mat[diffexp_ori_fdr<=fdr_cutoff]$label <- 1

diffexp_aupr_combined_mat <- c()
perf <- calc_overlap_aupr(diffexp_fdr_combined_mat$label, -diffexp_fdr_combined_mat$diffexp_polarbear, fdr_cutoff)
diffexp_aupr_combined_mat <- rbind(diffexp_aupr_combined_mat, cbind(perf@x.values[[1]], perf@y.values[[1]], rep("Polarbear", length(perf@x.values[[1]]))))
perf <- calc_overlap_aupr(diffexp_fdr_combined_mat$label, -diffexp_fdr_combined_mat$diffexp_polarbear_coassay, fdr_cutoff)
diffexp_aupr_combined_mat <- rbind(diffexp_aupr_combined_mat, cbind(perf@x.values[[1]], perf@y.values[[1]], rep("Polarbear co-assay", length(perf@x.values[[1]]))))
diffexp_aupr_combined_mat <- as.data.table(diffexp_aupr_combined_mat)
names(diffexp_aupr_combined_mat) <- c("Recall", "Precision", "model")
diffexp_aupr_combined_mat$Precision <- as.numeric(diffexp_aupr_combined_mat$Precision)
diffexp_aupr_combined_mat$Recall <- as.numeric(diffexp_aupr_combined_mat$Recall)
diffexp_aupr_combined_mat$model <- factor(diffexp_aupr_combined_mat$model, levels =c("BABEL","Polarbear co-assay","Polarbear"))
p1_diffexp_aupr <- ggplot(data=diffexp_aupr_combined_mat, aes(x=Recall, y=Precision, color=model)) + 
  ggtitle("Precision-recall curve")+
  geom_line(stat="identity", size=1)+ 
  scale_colour_manual(values = c("#bdbdbd","#1b9e77","#d95f02"))+
  theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
  theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))+ theme(legend.text=element_text(size=11), legend.title=element_text(size=11)) #+

png(paste0(cur_dir, "/result/eval_babel_normexp_diffexp_prc.png"), width = 530, height = 350, res=110)
print(p1_diffexp_aupr)
dev.off()

