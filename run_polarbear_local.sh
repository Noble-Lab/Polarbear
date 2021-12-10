#!/bin/bash
#
# train and evaluate Polarbear model
#conda activate tf1.13

cur_dir=.
data_dir=https://noble.gs.washington.edu/~ranz0/Polarbear/data/

train_test_split=$1
semi_version=$2
if [ "$semi_version" == "semi" ]
then
	path_x_single=${data_dir}/data/adultbrainfull50_rna_outer_single.mtx
	path_y_single=${data_dir}/data/adultbrainfull50_atac_outer_single.mtx
else
	path_x_single=nornasingle
	path_y_single=noatacsingle
fi

## train the model
python ${cur_dir}/bin/run_polarbear.py --path_x ${data_dir}/data/adultbrainfull50_rna_outer_snareseq.mtx --path_y ${data_dir}/data/adultbrainfull50_atac_outer_snareseq.mtx --outdir ${cur_dir}/output_${semi_version}_gpu/ --patience 45 --path_x_single $path_x_single --path_y_single $path_y_single --train_test_split ${train_test_split} --train train

## evaluate
#python ${cur_dir}/bin/run_polarbear.py --path_x ${data_dir}/data/adultbrainfull50_rna_outer_snareseq.mtx --path_y ${data_dir}/data/adultbrainfull50_atac_outer_snareseq.mtx --outdir ${cur_dir}/output_${semi_version}_gpu/ --patience 45 --path_x_single $path_x_single --path_y_single $path_y_single --train_test_split ${train_test_split} --train predict  --evaluate evaluate

## output predictions on test set
#python ${cur_dir}/bin/run_polarbear.py --path_x ${data_dir}/data/adultbrainfull50_rna_outer_snareseq.mtx --path_y ${data_dir}/data/adultbrainfull50_atac_outer_snareseq.mtx --outdir ${cur_dir}/output_${semi_version}_gpu/ --patience 45 --path_x_single $path_x_single --path_y_single $path_y_single --train_test_split ${train_test_split} --train predict --predict predict

## plot pairwise comparison
#mkdir -p ${cur_dir}/result/
#Rscript evaluate_polarbear.R

