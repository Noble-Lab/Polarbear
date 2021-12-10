library(data.table)
library(readxl)
library(ROCR)
library(Matrix)
library(scran)
library(ggplot2)
library(ggallin)


calculate_cor <- function(true_profile_mat, pred_profile_mat, snareseq_rna_mat_libsize, cor_method="pearson", norm_pred=FALSE){
  # calculate correlation: overall, by gene, and by cell
  # cor_method: "pearson" or "spearman"
  if (norm_pred){
    pred_profile_mat <- pred_profile_mat*mean(snareseq_rna_mat_libsize)
  }
  true_profile_mat <- log(true_profile_mat+1)
  pred_profile_mat <- log(pred_profile_mat+1)
  cor_bygene <- c()
  cor_bycell <- c()
  gene_npos <- apply(true_profile_mat, 2, function(x) sum(x>0))
  for (i in 1:ncol(true_profile_mat)){
    cor_bygene <- c(cor_bygene, cor(true_profile_mat[,i], pred_profile_mat[,i], method=cor_method))
  }
  for (i in 1:nrow(true_profile_mat)){
    cor_bycell <- c(cor_bycell, cor(true_profile_mat[i,], pred_profile_mat[i,], method=cor_method))
  }
  return(list("cor_overall"= cor(c(true_profile_mat), c(pred_profile_mat), method=cor_method), "cor_gene" =cor_bygene, "cor_cell" = cor_bycell, "npos"=gene_npos))
}



generate_cor_deconv <- function(cur_dir, snareseq_rna_mat_deconv_norm, snareseq_rna_barcodes, pred_prefix, selected_genes, cor_method, snareseq_rna_mat_libsize, polarbear_genes, babel_genes, plot=FALSE){
  ## generate correlation for flattened, cell-wise and gene-wise scRNA prediction vs true value. Also calculate baseline performance.
  if (length(selected_genes)>0){
    selected_genes <- intersect(intersect(selected_genes, polarbear_genes), babel_genes)
  }else{
    selected_genes <- intersect(polarbear_genes, babel_genes)
  }
  
  pred_mat <- fread(paste0(pred_prefix, "_test_rnanorm_pred.txt"), header=F)
  barcodes <- fread(paste0(pred_prefix, "_test_barcodes.txt"), header=F)$V1
  ## baseline: mean of training set
  snareseq_rna_mat_deconv_norm_baseline_training <- apply(snareseq_rna_mat_deconv_norm[!snareseq_rna_barcodes %in% barcodes,], 2, mean)
  snareseq_rna_mat_deconv_norm_baseline_training_mat <- matrix(snareseq_rna_mat_deconv_norm_baseline_training, nrow=length(barcodes), ncol=length(snareseq_rna_mat_deconv_norm_baseline_training), byrow=TRUE)
  ## compare MSE of our prediction v.s. baseline
  matched_barcode <- match(barcodes, snareseq_rna_barcodes)
  snareseq_rna_mat_deconv_norm_test <- snareseq_rna_mat_deconv_norm[matched_barcode,]
  
  if (length(selected_genes)>0){
    matched_genes <- match(selected_genes, polarbear_genes)
    matched_genes <- matched_genes[!is.na(matched_genes)]
    print(length(matched_genes))
    snareseq_rna_mat_deconv_norm_baseline_training_mat <- snareseq_rna_mat_deconv_norm_baseline_training_mat[,matched_genes]
    snareseq_rna_mat_deconv_norm_test <- snareseq_rna_mat_deconv_norm_test[,matched_genes]
    pred_mat <- data.matrix(pred_mat)[,matched_genes]
  }
  cor_pred <- calculate_cor(snareseq_rna_mat_deconv_norm_test, data.matrix(pred_mat), snareseq_rna_mat_libsize, cor_method=cor_method, norm_pred = TRUE)
  cor_baseline <- calculate_cor(snareseq_rna_mat_deconv_norm_test, snareseq_rna_mat_deconv_norm_baseline_training_mat, snareseq_rna_mat_libsize, cor_method=cor_method)
  return(list("cor_pred"= cor_pred, "cor_baseline" =cor_baseline))
}



plot_cor_compare_deconv <- function(cur_dir, split_ver, snareseq_rna_mat_deconv_norm, snareseq_rna_barcodes, gene_signature_ver, selected_genes, npos_cutoff, pred_prefix_coassay, pred_prefix_semi, cor_method, snareseq_rna_mat_libsize, polarbear_genes, babel_genes, pred_prefix_babel=""){
  # compare correlation (calculated against Lun et al. deconvolution-normalized original expression)
  cor_coassay <- generate_cor_deconv(cur_dir, snareseq_rna_mat_deconv_norm, snareseq_rna_barcodes, pred_prefix_coassay, selected_genes, cor_method, snareseq_rna_mat_libsize, polarbear_genes, babel_genes)
  cor_semi <- generate_cor_deconv(cur_dir, snareseq_rna_mat_deconv_norm, snareseq_rna_barcodes, pred_prefix_semi, selected_genes, cor_method, snareseq_rna_mat_libsize, polarbear_genes, babel_genes)
  if (pred_prefix_babel!=""){
    mse_babel <- generate_cor_deconv(cur_dir, snareseq_rna_mat_deconv_norm, snareseq_rna_barcodes, pred_prefix_babel, selected_genes, cor_method, snareseq_rna_mat_libsize, polarbear_genes, babel_genes)
  }
  ## scatterplot of Polarbear (semi) vs Polarbear co-assay
  if (pred_prefix_babel!=""){
    mse_gene_mat_pairwise <- as.data.table(cbind(cor_coassay$cor_pred$cor_gene, cor_semi$cor_pred$cor_gene, mse_babel$cor_pred$cor_gene, cor_semi$cor_pred$npos))
    names(mse_gene_mat_pairwise) <- c("Polarbear co-assay","Polarbear","BABEL","npos")
    mse_gene_mat_pairwise$`expression` = "low"
    mse_gene_mat_pairwise = na.omit(mse_gene_mat_pairwise)
    mse_gene_mat_pairwise <- mse_gene_mat_pairwise[npos>npos_cutoff]
    
    ## print out stats
    print(apply(mse_gene_mat_pairwise[,c("Polarbear co-assay","Polarbear","BABEL")], 2, median))
    print(apply(mse_gene_mat_pairwise[,c("Polarbear co-assay","Polarbear","BABEL")], 2, IQR))
    print(apply(mse_gene_mat_pairwise[,c("Polarbear co-assay","Polarbear","BABEL")], 2, mean))
    print(apply(mse_gene_mat_pairwise[,c("Polarbear co-assay","Polarbear","BABEL")], 2, sd))
  }else{
    mse_gene_mat_pairwise <- as.data.table(cbind(cor_coassay$cor_pred$cor_gene, cor_semi$cor_pred$cor_gene, cor_semi$cor_pred$npos))
    names(mse_gene_mat_pairwise) <- c("Polarbear co-assay","Polarbear","npos")
    mse_gene_mat_pairwise$`expression` = "low"
    mse_gene_mat_pairwise = na.omit(mse_gene_mat_pairwise)
    mse_gene_mat_pairwise <- mse_gene_mat_pairwise[npos>npos_cutoff]
  }
  
  count_diagonal = table(mse_gene_mat_pairwise$`Polarbear`>mse_gene_mat_pairwise$`Polarbear co-assay`)
  pval = wilcox.test(mse_gene_mat_pairwise$`Polarbear co-assay`, mse_gene_mat_pairwise$Polarbear, alternative = "less")$p.value
  
  p2_cor_compare <- ggplot(mse_gene_mat_pairwise, aes(x = `Polarbear co-assay`, y = Polarbear)) + 
    geom_point(size = 1, alpha=.5, color="#2c7bb6")+ ggtitle("gene-wise correlation")+
    theme_classic()+
    theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
    theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))
  
  if (split_ver==""){
    p2_cor_compare <- p2_cor_compare +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
      annotate("text", y = min(mse_gene_mat_pairwise$`Polarbear`)+0.12, x = max(mse_gene_mat_pairwise$`Polarbear co-assay`) -0.15 , label = count_diagonal[1])+
      annotate("text", y = max(mse_gene_mat_pairwise$`Polarbear`)-0.1, x = min(mse_gene_mat_pairwise$`Polarbear co-assay`) +0.1 , label = count_diagonal[2])+
      annotate("text", x = max(mse_gene_mat_pairwise$`Polarbear co-assay`)-0.12, y = min(mse_gene_mat_pairwise$`Polarbear`) +0.05 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
  }else{
    p2_cor_compare <- p2_cor_compare +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
      annotate("text", y = min(mse_gene_mat_pairwise$`Polarbear`)+0.07, x = max(mse_gene_mat_pairwise$`Polarbear co-assay`) -0.1 , label = count_diagonal[1])+
      annotate("text", y = max(mse_gene_mat_pairwise$`Polarbear`)-0.07, x = min(mse_gene_mat_pairwise$`Polarbear co-assay`) +0.07 , label = count_diagonal[2])+
      annotate("text", x = max(mse_gene_mat_pairwise$`Polarbear co-assay`)-0.07, y = min(mse_gene_mat_pairwise$`Polarbear`) +0.03 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
  }
  png(paste0(cur_dir, "/result/eval_", split_ver, "_",gene_signature_ver, npos_cutoff,"_", cor_method, "_correlation_genewise_scattercomparecoassay.png"), width = 330, height = 350, res=110)
  print(p2_cor_compare)
  dev.off()
  
  if (pred_prefix_babel!=""){
    # compare Polarbear with BABEL
    count_diagonal = table(mse_gene_mat_pairwise$`Polarbear`>mse_gene_mat_pairwise$`BABEL`)
    pval = wilcox.test(mse_gene_mat_pairwise$`BABEL`, mse_gene_mat_pairwise$Polarbear, alternative = "less")$p.value
    
    p2_cor_compare <- ggplot(mse_gene_mat_pairwise, aes(x = BABEL, y = Polarbear)) + 
      geom_point(size = 1, alpha=.5, color="#2c7bb6")+ ggtitle("gene-wise correlation")+
      theme_classic()+
      theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
      theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))
    
    if (split_ver==""){
      p2_cor_compare <- p2_cor_compare +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
        annotate("text", y = min(mse_gene_mat_pairwise$`Polarbear`)+0.12, x = max(mse_gene_mat_pairwise$BABEL) -0.12 , label = count_diagonal[1])+
        annotate("text", y = max(mse_gene_mat_pairwise$`Polarbear`)-0.1, x = min(mse_gene_mat_pairwise$BABEL) +0.1 , label = count_diagonal[2])+
        annotate("text", x = max(mse_gene_mat_pairwise$`BABEL`)-0.12, y = min(mse_gene_mat_pairwise$`Polarbear`) +0.05 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
    }else{
      p2_cor_compare <- p2_cor_compare +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
        annotate("text", y = min(mse_gene_mat_pairwise$`Polarbear`)+0.07, x = max(mse_gene_mat_pairwise$BABEL) -0.07 , label = count_diagonal[1])+
        annotate("text", y = max(mse_gene_mat_pairwise$`Polarbear`)-0.07, x = min(mse_gene_mat_pairwise$BABEL) +0.07 , label = count_diagonal[2])+
        annotate("text", x = max(mse_gene_mat_pairwise$`BABEL`)-0.07, y = min(mse_gene_mat_pairwise$`Polarbear`) +0.03 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
    }
    png(paste0(cur_dir, "/result/eval_", split_ver, "_",gene_signature_ver, npos_cutoff,"_correlation_genewise_scattercomparebabel.png"), width = 330, height = 350, res=110)
    print(p2_cor_compare)
    dev.off()
  }
}



plot_cor_compare <- function(cur_dir, data_dir, split_ver, gene_signature_ver, selected_genes, npos_cutoff, pred_prefix_coassay, pred_prefix_semi, pred_prefix_babel=""){
  # compare correlation (calculated against size-normalized original expression)
  polarbear_genes <- fread(paste0(data_dir, "/data/adultbrainfull50_rna_outer_genes.txt"), header=F)$V1
  babel_genes <- fread(paste0(data_dir, "/data/babel_genes.txt"), header=F)$V1
  if (length(selected_genes)>0){
    selected_genes <- intersect(intersect(selected_genes, polarbear_genes), babel_genes)
  }else{
    selected_genes <- intersect(polarbear_genes, babel_genes)
  }
  
  cor_coassay <- fread(paste0(pred_prefix_coassay, "_test_rna_cor.txt"), header=F)$V1
  cor_semi <- fread(paste0(pred_prefix_semi, "_test_rna_cor.txt"), header=F)$V1
  mse_gene_mat_pairwise <- cbind(cor_coassay[match(selected_genes, polarbear_genes)],
                                 cor_semi[match(selected_genes, polarbear_genes)])
  
  mse_gene_mat_pairwise <- as.data.table(mse_gene_mat_pairwise)
  names(mse_gene_mat_pairwise) <- c("Polarbear co-assay","Polarbear")
  count_diagonal = table(mse_gene_mat_pairwise$`Polarbear`>mse_gene_mat_pairwise$`Polarbear co-assay`)
  pval = wilcox.test(mse_gene_mat_pairwise$`Polarbear co-assay`, mse_gene_mat_pairwise$Polarbear, alternative = "less")$p.value
  
  p2_cor_compare <- ggplot(mse_gene_mat_pairwise, aes(x = `Polarbear co-assay`, y = Polarbear)) + 
    geom_point(size = 1, alpha=.5, color="#2c7bb6")+ ggtitle("gene-wise correlation")+
    theme_classic()+
    theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
    theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))
  
  p2_cor_compare <- p2_cor_compare +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
    annotate("text", y = min(mse_gene_mat_pairwise$`Polarbear`)+0.07, x = max(mse_gene_mat_pairwise$`Polarbear co-assay`) -0.1 , label = count_diagonal[1])+
    annotate("text", y = max(mse_gene_mat_pairwise$`Polarbear`)-0.07, x = min(mse_gene_mat_pairwise$`Polarbear co-assay`) +0.07 , label = count_diagonal[2])+
    annotate("text", x = max(mse_gene_mat_pairwise$`Polarbear co-assay`)-0.07, y = min(mse_gene_mat_pairwise$`Polarbear`) +0.03 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
  
  png(paste0(cur_dir, "/result/eval_", split_ver, "_",gene_signature_ver, npos_cutoff,"_correlation_ori_genewise_scattercomparecoassay.png"), width = 330, height = 350, res=110)
  print(p2_cor_compare)
  dev.off()
}



plot_marker_exp <- function(markeri, pred_mat, true_mat, matched_barcode){
  # calculate AUROC for marker gene "markeri", both on pred_mat and true_mat, using the matched barcode as labels
  names(pred_mat) <- "expression"
  pred_mat$type <- "prediction"
  pred_mat$celltype <- "unmatched"
  pred_mat$celltype[matched_barcode] <- "matched"
  
  names(true_mat) <- "expression"
  true_mat$type <- "original"
  true_mat$celltype <- pred_mat$celltype
  pred_mat <- rbind(pred_mat, true_mat)
  
  pred_mat$label <- 0
  pred_mat[celltype=="matched"]$label <- 1
  pred <- prediction(pred_mat[type=="prediction"]$expression, pred_mat[type=="prediction"]$label)
  perf <- performance(pred, "auc")
  auc_pred <- as.numeric(perf@y.values)
  
  pred <- prediction(pred_mat[type=="original"]$expression, pred_mat[type=="original"]$label)
  perf <- performance(pred, "auc")
  auc_true <- as.numeric(perf@y.values)
  return(c(markeri, auc_true, auc_pred, length(matched_barcode)))
}



calculate_diffexp <- function(pred_profile_matched_mat, pred_profile_unmatched_mat, method="pval"){
  # calculate differentially expressed p-value for each gene, between pred_profile_matched_mat and pred_profile_unmatched_mat
  pval_bygene <- c()
  for (i in 1:ncol(pred_profile_matched_mat)){
    if (method=="pval"){
      pval = wilcox.test(pred_profile_matched_mat[,i], pred_profile_unmatched_mat[,i], alternative = "greater")$p.value
    }else if (method=="logFC"){
      pval = log(mean(pred_profile_matched_mat[,i])/mean(pred_profile_unmatched_mat[,i]))
    }else if (method=="auc"){
      pred_mat = as.data.table(cbind(c(pred_profile_matched_mat[,i], pred_profile_unmatched_mat[,i]), c(rep(1, nrow(pred_profile_matched_mat)), rep(0, nrow(pred_profile_unmatched_mat)))))
      names(pred_mat) <- c("prediction","label")
      pred <- prediction(pred_mat$prediction, pred_mat$label)
      perf <- performance(pred, "auc")
      pval <- as.numeric(perf@y.values)
    }
    pval_bygene = c(pval_bygene, pval)
  }
  return(pval_bygene)
}


plor_auc_aupr_compare <- function(split_ver, metric, ncells, peaks, snareadult_diffpeaks, pred_prefix_coassay, pred_prefix_semi, pred_prefix_babel=""){
  peak_auc_mat <- c()
  peak_auc_npos <- fread(paste0(pred_prefix_coassay, "_test_atac_npos.txt"), header=F)$V1
  for (model in c("Polarbear co-assay","Polarbear")){
    if (model == "BABEL"){
      if (split_ver=="random"){
        file_ver = "output_snareseq_0.01_50_128_0.2"
      }else{
        file_ver = "output_snareseq_0.01_25_128_5"
      }
      peak_auc <- fread(paste0(pred_prefix_babel, "_test_atac_", metric, ".txt"), header=F)
    }else if (model == "Polarbear co-assay"){
      peak_auc <- fread(paste0(pred_prefix_coassay, "_test_atac_",metric,".txt"), header=F)
    }else{
      peak_auc <- fread(paste0(pred_prefix_semi, "_test_atac_",metric,".txt"), header=F)
    }
    peak_auc$model <- model
    if (split_ver=="random"){
      peak_auc_mat <- rbind(peak_auc_mat, peak_auc[peak_auc_npos>= ncells/100,])
    }else{
      peak_auc_mat <- rbind(peak_auc_mat, peak_auc[snareadult_diffpeaks$celltype=="Ex-L2/3-Rasgrf2" & peak_auc_npos>= ncells/100,])
    }
  }
  
  names(peak_auc_mat)[1] <- "AUC"
  peak_auc_mat <- peak_auc_mat[!is.na(AUC)]
  peak_auc_mat$AUC <- as.numeric(peak_auc_mat$AUC)
  
  ## compare Polarbear with Polarbear co-assay
  peak_auc_mat_pairwise = peak_auc_mat[model=="Polarbear co-assay"]
  names(peak_auc_mat_pairwise)[1] <- "Polarbear co-assay"
  peak_auc_mat_pairwise$Polarbear = peak_auc_mat[model=="Polarbear"]$AUC
  print("== median ==")
  print(apply(peak_auc_mat_pairwise[,c("Polarbear","Polarbear co-assay")], 2, median))
  print(apply(peak_auc_mat_pairwise[,c("Polarbear","Polarbear co-assay")], 2, IQR))
  print("== mean ==")
  print(apply(peak_auc_mat_pairwise[,c("Polarbear","Polarbear co-assay")], 2, mean))
  print(apply(peak_auc_mat_pairwise[,c("Polarbear","Polarbear co-assay")], 2, sd))
  count_diagonal = table(peak_auc_mat_pairwise$`Polarbear`>peak_auc_mat_pairwise$`Polarbear co-assay`)
  pval = wilcox.test(peak_auc_mat_pairwise$`Polarbear co-assay`, peak_auc_mat_pairwise$Polarbear, alternative = "less")$p.value
  if (metric=="auc"){
    p1_atac_auc_diffgene_scatter <- ggplot(peak_auc_mat_pairwise, aes(x = `Polarbear co-assay`, y = Polarbear)) + 
      geom_point(size = 0.8, alpha=.1, color="#2c7bb6")+ ggtitle("peak-wise AUROC")+
      theme_classic()+
      theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
      theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))
    
    p1_atac_auc_diffgene_scatter <- p1_atac_auc_diffgene_scatter +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
      annotate("text", x = min(peak_auc_mat_pairwise$`Polarbear co-assay`)+0.1, y = max(peak_auc_mat_pairwise$`Polarbear`) -0.1 , label = count_diagonal[2])+
      annotate("text", x = max(peak_auc_mat_pairwise$`Polarbear co-assay`)-0.1, y = min(peak_auc_mat_pairwise$`Polarbear`) +0.1 , label = count_diagonal[1])+
      annotate("text", x = max(peak_auc_mat_pairwise$`Polarbear co-assay`)-0.1, y = min(peak_auc_mat_pairwise$`Polarbear`) +0.03 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
  }else{
    p1_atac_auc_diffgene_scatter <- ggplot(peak_auc_mat_pairwise, aes(x = `Polarbear co-assay`, y = Polarbear)) + 
      geom_point(size = 0.8, alpha=.1, color="#2c7bb6")+ ggtitle("peak-wise AUPRnorm")+
      theme_classic()+
      theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
      theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))+
      scale_x_continuous(trans=ssqrt_trans) +
      scale_y_continuous(trans=ssqrt_trans)
    
    p1_atac_auc_diffgene_scatter <- p1_atac_auc_diffgene_scatter +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
      annotate("text", x = min(peak_auc_mat_pairwise$`Polarbear co-assay`)+0.008, y = max(peak_auc_mat_pairwise$`Polarbear`) -0.05 , label = count_diagonal[2])+
      annotate("text", x = max(peak_auc_mat_pairwise$`Polarbear co-assay`)-0.07, y = min(peak_auc_mat_pairwise$`Polarbear`) +0.008 , label = count_diagonal[1])+
      annotate("text", x = max(peak_auc_mat_pairwise$`Polarbear co-assay`)-0.07, y = min(peak_auc_mat_pairwise$`Polarbear`) +0.003 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
  }
  
  png(paste0(cur_dir, "/result/eval_", split_ver, "_diffexp_", metric,"_scatterplot_comparecoassay.png"), width = 330, height = 350, res=110)
  print(p1_atac_auc_diffgene_scatter)
  dev.off()
  
  ## compare Polarbear with BABEL
  if (FALSE){
    peak_auc_mat_pairwise = peak_auc_mat[model=="BABEL"]
    names(peak_auc_mat_pairwise)[1] <- "BABEL"
    peak_auc_mat_pairwise$Polarbear = peak_auc_mat[model=="Polarbear"]$AUC
    print("== median ==")
    print(apply(peak_auc_mat_pairwise[,c("Polarbear","BABEL")], 2, median))
    print(apply(peak_auc_mat_pairwise[,c("Polarbear","BABEL")], 2, IQR))
    print("== mean ==")
    print(apply(peak_auc_mat_pairwise[,c("Polarbear","BABEL")], 2, mean))
    print(apply(peak_auc_mat_pairwise[,c("Polarbear","BABEL")], 2, sd))
    count_diagonal = table(peak_auc_mat_pairwise$`Polarbear`>peak_auc_mat_pairwise$`BABEL`)
    pval = wilcox.test(peak_auc_mat_pairwise$`BABEL`, peak_auc_mat_pairwise$Polarbear, alternative = "less")$p.value
    
    if (metric=="auc"){
      p1_atac_auc_diffgene_scatter <- ggplot(peak_auc_mat_pairwise, aes(x = BABEL, y = Polarbear)) + 
        geom_point(size = 0.8, alpha=.1, color="#2c7bb6")+ ggtitle("peak-wise AUROC")+
        theme_classic()+
        theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
        theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))
      
      p1_atac_auc_diffgene_scatter <- p1_atac_auc_diffgene_scatter +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
        annotate("text", x = min(peak_auc_mat_pairwise$`BABEL`)+0.1, y = max(peak_auc_mat_pairwise$`Polarbear`) -0.1 , label = count_diagonal[2])+
        annotate("text", x = max(peak_auc_mat_pairwise$`BABEL`)-0.1, y = min(peak_auc_mat_pairwise$`Polarbear`) +0.1 , label = count_diagonal[1])+
        annotate("text", x = max(peak_auc_mat_pairwise$`BABEL`)-0.1, y = min(peak_auc_mat_pairwise$`Polarbear`) +0.03 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
    }else{
      p1_atac_auc_diffgene_scatter <- ggplot(peak_auc_mat_pairwise, aes(x = BABEL, y = Polarbear)) + 
        geom_point(size = 0.8, alpha=.1, color="#2c7bb6")+ ggtitle("peak-wise AUROC")+
        theme_classic()+
        theme(panel.background = element_rect(fill = "white", colour = "white"), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) +
        theme(axis.text=element_text(size=12,colour="black"), axis.title=element_text(size=15, colour="black"))+
        scale_x_continuous(trans=ssqrt_trans) +
        scale_y_continuous(trans=ssqrt_trans)
      
      p1_atac_auc_diffgene_scatter <- p1_atac_auc_diffgene_scatter +  geom_abline(intercept =0 , slope = 1, linetype = "dashed")+
        annotate("text", x = min(peak_auc_mat_pairwise$`BABEL`)+0.008, y = max(peak_auc_mat_pairwise$`Polarbear`) -0.05 , label = count_diagonal[2])+
        annotate("text", x = max(peak_auc_mat_pairwise$`BABEL`)-0.07, y = min(peak_auc_mat_pairwise$`Polarbear`) +0.008 , label = count_diagonal[1])+
        annotate("text", x = max(peak_auc_mat_pairwise$`BABEL`)-0.07, y = min(peak_auc_mat_pairwise$`Polarbear`) +0.003 , label = paste0("P = ",formatC(pval, format = "e", digits = 2)))
    }
    
    png(paste0(cur_dir, "/result/eval_", split_ver, "_diffexp_", metric,"_scatterplot_comparebabel.png"), width = 330, height = 350, res=110)
    print(p1_atac_auc_diffgene_scatter)
    dev.off()
  }
}



calc_overlap_aupr <- function(label, pred_fdr,fdr_cutoff){
  # calculate precision recall curve ROCR perf object
  pred_mat <- as.data.table(cbind(label,pred_fdr))
  pred <- prediction(pred_mat$pred_fdr, pred_mat$label)
  perf <- performance(pred, measure = "prec", x.measure = "rec")
  return(perf)
}



load_foscttm_matrix <- function(pred_prefix){
  # load foscttm score calculation, both querying each RNA cell and each ATAC cell
  foscttm_rna = fread(paste0(pred_prefix, "_test_sorted_fraction_1to2_atacVAE.txt"), header=F)$V1
  foscttm_atac = fread(paste0(pred_prefix, "_test_sorted_fraction_2to1_atacVAE.txt"), header=F)$V1
  foscttm_mat <- as.data.table(rbind(cbind(1:length(foscttm_rna), sort(foscttm_rna, decreasing=FALSE), rep("RNA",length(foscttm_rna))), 
                                     cbind(1:length(foscttm_atac), sort(foscttm_atac, decreasing=FALSE), rep("ATAC",length(foscttm_atac))))
  )
  names(foscttm_mat) <- c("Cell","FOSCTTM","query")
  foscttm_mat$FOSCTTM <- as.numeric(foscttm_mat$FOSCTTM)
  foscttm_mat$Cell <- as.integer(foscttm_mat$Cell)
  return(foscttm_mat)
}

