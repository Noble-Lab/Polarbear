#!/usr/bin/env python

import os, sys, argparse, random;
import tensorflow as tf;
from tensorflow.python.keras import backend as K
import numpy as np;
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeCV
import scipy
from scipy.io import mmread
import math

"""
train_model.py builds separate autoencoders for each domain, and then connects two bottle-neck layers with translator layer. 
Loss = reconstr_loss of each domain + prediction_error of co-assay data from one domain to the other.
"""

class TranslateAE:
    def __init__(self, input_dim_x, input_dim_y, batch_dim_x, batch_dim_y, embed_dim_x, embed_dim_y, dispersion, chr_list, nlayer, dropout_rate, output_model, learning_rate_x, learning_rate_y, learning_rate_xy, learning_rate_yx, trans_ver, hidden_frac=2, kl_weight=1):
        """
        Network architecture and optimization

        Inputs
        ----------
        input_x: scRNA expression, ncell x input_dim_x, float
        input_y: scATAC expression, ncell x input_dim_y, float
        batch_x: scRNA batch factor, ncell x batch_dim_x, int
        batch_y: scATAC batch factor, ncell x batch_dim_y, int
        chr_list: dictionary using chr as keys and corresponding peak index as vals

        Parameters
        ----------
        kl_weight_x: non-negative value, float
        kl_weight_y: non-negative value, float
        input_dim_x: #genes, int
        input_dim_y: #peaks, int
        batch_dim_x: dimension of batch matrix in s domain, int
        batch_dim_y: dimension of batch matrix in t domain, int
        embed_dim_x: embedding dimension in s VAE, int
        embed_dim_y: embedding dimension in t VAE, int
        learning_rate_x: scRNA VAE learning rate, float
        learning_rate_y: scATAC VAE learning rate, float
        learning_rate_xy: scRNA embedding to scATAC embedding translation learning rate, float
        learning_rate_yx: scATAC embedding to scRNA embedding translation learning rate, float
        nlayer: number of hidden layers in encoder/decoder, int, >=1
        dropout_rate: dropout rate in VAE, float
        trans_ver: translation layer in between embeddings, "linear" or "1l" or "2l"
        dispersion: estimate dispersion per gene&batch: "genebatch" or per gene&cell: "genecell"
        hidden_frac: used to divide intermediate layer dimension, int
        kl_weight: weight of KL divergence loss in VAE, float

        """
        self.input_dim_x = input_dim_x;
        self.input_dim_y = input_dim_y;
        self.batch_dim_x = batch_dim_x;
        self.batch_dim_y = batch_dim_y;
        self.embed_dim_x = embed_dim_x;
        self.embed_dim_y = embed_dim_y;
        self.learning_rate_x = learning_rate_x;
        self.learning_rate_y = learning_rate_y;
        self.learning_rate_xy = learning_rate_xy;
        self.learning_rate_yx = learning_rate_yx;
        self.nlayer = nlayer;
        self.dropout_rate = dropout_rate;
        self.trans_ver = trans_ver
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_dim_x]);
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.input_dim_y]);
        self.batch_x = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x]);
        self.batch_y = tf.placeholder(tf.float32, shape=[None, self.batch_dim_y]);
        self.kl_weight_x = tf.placeholder(tf.float32, None);
        self.kl_weight_y = tf.placeholder(tf.float32, None);
        self.dispersion = dispersion
        self.hidden_frac = hidden_frac
        self.kl_weight = kl_weight
        self.chr_list = chr_list

        def encoder_rna(input_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA encoder
            Parameters
            ----------
            hidden_frac: used to divide intermediate dimension to shrink the total paramater size to fit into memory
            input_data: generated from tf.concat([self.input_x, self.batch_x], 1), ncells x (input_dim_x + batch_dim_x)
            """
            with tf.variable_scope('encoder_x', reuse=tf.AUTO_REUSE):
                self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x)/hidden_frac)
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                encoder_output_mean = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_mean')(l1)
                encoder_output_var = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_var')(l1)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min = -2000000, clip_value_max=15)
                encoder_output_var = tf.math.exp(encoder_output_var) + 0.0001
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim_x), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.sqrt(encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;
            

        def decoder_rna(encoded_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA decoder
            Parameters
            ----------
            hidden_frac: intermadiate layer dim, used hidden_frac to shrink the size to fit into memory
            layer_norm_type: how we normalize layer, don't worry about it now
            encoded_data: generated from concatenation of the encoder output self.encoded_x and batch_x: tf.concat([self.encoded_x, self.batch_x], 1), ncells x (embed_dim_x + batch_dim_x)
            """

            self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x)/hidden_frac); ## TODO: similarly, here we need to add another dimension for each new input batch factor (where previous batch dimension will be 0 and new dimension will be 1). Add an option of only fine tune weights coming out of this added batch dimension when we input new dataset.
            with tf.variable_scope('decoder_x', reuse=tf.AUTO_REUSE):
                #self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x));
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                px = tf.layers.Dense(self.intermediate_dim, activation=tf.nn.relu, name='decoder_x_px')(l1);
                px_scale = tf.layers.Dense(self.input_dim_x, activation=tf.nn.softmax, name='decoder_x_px_scale')(px);
                px_dropout = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_dropout')(px) 
                px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_r')(px)#, use_bias=False
                    
                return px_scale, px_dropout, px_r
                    

        def encoder_atac(input_data, nlayer, hidden_frac, chr_list, reuse=tf.AUTO_REUSE):
            """
            scATAC encoder, only allow within chromosome connections for the first several layers
            """
            with tf.variable_scope('encoder_y', reuse=tf.AUTO_REUSE):
                dic_intermediate_dim = {}
                dic_l1 = {}
                dic_l2 = {}
                dic_l2_list = []
                for chri in chr_list.keys():
                    dic_intermediate_dim[chri] = int(math.sqrt((len(chr_list[chri]) + self.batch_dim_y) * self.embed_dim_y)/hidden_frac)
                    dic_l1[chri] = tf.layers.Dense(dic_intermediate_dim[chri], activation=None, name='encoder_y_0'+str(chri))(tf.gather(input_data, chr_list[chri]+list(range(self.input_dim_y-self.batch_dim_y, self.input_dim_y)), axis=1));
                    dic_l1[chri] = tf.contrib.layers.layer_norm(inputs=dic_l1[chri], center=True, scale=True);
                    dic_l1[chri] = tf.nn.leaky_relu(dic_l1[chri])
                    dic_l1[chri] = tf.nn.dropout(dic_l1[chri], rate=self.dropout_rate);
                    for layer_i in range(1, nlayer):
                        dic_l1[chri] = tf.layers.Dense(dic_intermediate_dim[chri], activation=None, name='encoder_y_'+str(layer_i)+'_'+str(chri))(dic_l1[chri]);
                        dic_l1[chri] = tf.contrib.layers.layer_norm(inputs=dic_l1[chri], center=True, scale=True);
                        dic_l1[chri] = tf.nn.leaky_relu(dic_l1[chri])
                        dic_l1[chri] = tf.nn.dropout(dic_l1[chri], rate=self.dropout_rate);
                    dic_l2[chri] = tf.layers.Dense(self.embed_dim_y, activation=None, name='encoder_y_output_'+str(chri))(dic_l1[chri]);
                    dic_l2[chri] = tf.contrib.layers.layer_norm(inputs=dic_l2[chri], center=True, scale=True);
                    dic_l2[chri] = tf.nn.leaky_relu(dic_l2[chri])
                    dic_l2[chri] = tf.nn.dropout(dic_l2[chri], rate=self.dropout_rate);
                    dic_l2_list.append(dic_l2[chri])

                l2_concatenate = tf.concat(dic_l2_list, 1)
                encoder_output_mean = tf.layers.Dense(self.embed_dim_y, activation=None, name='encoder_y_mean')(l2_concatenate)
                encoder_output_var = tf.layers.Dense(self.embed_dim_y, activation=None, name='encoder_y_var')(l2_concatenate)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min = -2000000, clip_value_max=15)
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim_y), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.exp(0.5 * encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;

        def decoder_atac(encoded_data, nlayer, hidden_frac, chr_list, reuse=tf.AUTO_REUSE):
            """
            scATAC decoder, only allow within chromosome connections for the last several layers
            """
            with tf.variable_scope('decoder_y', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(self.embed_dim_y * 22, activation=None, name='decoder_y_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                dic_intermediate_dim_decode  = {}
                dic_l1_decode = {}
                dic_l2_decode = {}
                py = []
                for chri in chr_list.keys():
                    dic_intermediate_dim_decode[chri] = int(math.sqrt((len(chr_list[chri]) + self.batch_dim_y) * self.embed_dim_y)/hidden_frac)
                    dic_l2_decode[chri] = tf.layers.Dense(dic_intermediate_dim_decode[chri], activation=None, name='decoder_y_1'+str(chri))(l1);
                    dic_l2_decode[chri] = tf.contrib.layers.layer_norm(inputs=dic_l2_decode[chri], center=True, scale=True);
                    dic_l2_decode[chri] = tf.nn.leaky_relu(dic_l2_decode[chri])
                    dic_l2_decode[chri] = tf.nn.dropout(dic_l2_decode[chri], rate=self.dropout_rate);
                    for layer_i in range(1, nlayer):
                        dic_l2_decode[chri] = tf.layers.Dense(dic_intermediate_dim_decode[chri], activation=None, name='decoder_y_'+str(nlayer+1)+'_'+str(chri))(dic_l2_decode[chri]);
                        dic_l2_decode[chri] = tf.contrib.layers.layer_norm(inputs=dic_l2_decode[chri], center=True, scale=True);
                        dic_l2_decode[chri] = tf.nn.leaky_relu(dic_l2_decode[chri])
                        dic_l2_decode[chri] = tf.nn.dropout(dic_l2_decode[chri], rate=self.dropout_rate);
                    dic_l1_decode[chri] = tf.layers.Dense(len(chr_list[chri]), activation=tf.nn.sigmoid, name='decoder_y_output'+str(chri))(dic_l2_decode[chri]);
                    py.append(dic_l1_decode[chri])

                py_concat = tf.concat(py, 1)
                return py_concat;
            
            
        def translator_rnatoatac(encoded_data, trans_ver, reuse=tf.AUTO_REUSE):
            """
            translate from scRNA to scATAC
            """
            with tf.variable_scope('translator_xy', reuse=tf.AUTO_REUSE):
                if trans_ver == 'linear':
                    translator_output = tf.layers.Dense(self.embed_dim_y, activation=None, name='translator_xy_1')(encoded_data);
                elif trans_ver == '1l':
                    translator_output = tf.layers.Dense(self.embed_dim_y, activation=tf.nn.leaky_relu, name='translator_xy_1')(encoded_data);
                elif trans_ver == '2l':
                    l1 = tf.layers.Dense(self.embed_dim_y, activation=tf.nn.leaky_relu, name='translator_xy_1')(encoded_data);
                    l2 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True)
                    translator_output = tf.layers.Dense(self.embed_dim_y, activation=tf.nn.leaky_relu, name='translator_xy_2')(l2);    
                return translator_output;


        def translator_atactorna(encoded_data, trans_ver, reuse=tf.AUTO_REUSE):
            """
            translate from scATAC to scRNA
            """
            with tf.variable_scope('translator_yx', reuse=tf.AUTO_REUSE):
                if trans_ver == 'linear':
                    translator_output = tf.layers.Dense(self.embed_dim_x, activation=None, name='translator_yx_1')(encoded_data);
                elif trans_ver == '1l':
                    translator_output = tf.layers.Dense(self.embed_dim_x, activation=tf.nn.leaky_relu, name='translator_yx_1')(encoded_data);
                elif trans_ver == '2l':
                    l1 = tf.layers.Dense(self.embed_dim_x, activation=tf.nn.leaky_relu, name='translator_yx_1')(encoded_data);
                    l2 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True)
                    translator_output = tf.layers.Dense(self.embed_dim_x, activation=tf.nn.leaky_relu, name='translator_yx_2')(l2);
                return translator_output;

        self.libsize_x = tf.reduce_sum(self.input_x, 1)
        self.libsize_y = tf.reduce_sum(self.input_y, 1) / 1000
        
        self.px_z_m, self.px_z_v, self.encoded_x = encoder_rna(tf.concat([self.input_x, self.batch_x], 1), self.nlayer, self.hidden_frac);

        z = tf.truncated_normal(tf.shape(self.px_z_m), stddev=1.0)
        self.py_z_m, self.py_z_v, self.encoded_y = encoder_atac(tf.concat([self.input_y, self.batch_y], 1), self.nlayer, self.hidden_frac, self.chr_list);

        ## scRNA reconstruction
        self.px_scale, self.px_dropout, self.px_r = decoder_rna(tf.concat([self.encoded_x, self.batch_x], 1), self.nlayer, self.hidden_frac);
        if self.dispersion == 'genebatch':
            self.px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='px_r_genebatch_x')(self.batch_x)

        self.px_r = tf.clip_by_value(self.px_r, clip_value_min = -2000000, clip_value_max=15)
        self.px_r = tf.math.exp(self.px_r)
        self.reconstr_x = tf.transpose(tf.transpose(self.px_scale) *self.libsize_x)
        
        ## scRNA loss
        # reconstr loss
        self.softplus_pi = tf.nn.softplus(-self.px_dropout)
        self.log_theta_eps = tf.log(self.px_r + 1e-8)
        self.log_theta_mu_eps = tf.log(self.px_r + self.reconstr_x + 1e-8)
        self.pi_theta_log = -self.px_dropout + tf.multiply(self.px_r, (self.log_theta_eps - self.log_theta_mu_eps))

        self.case_zero = tf.nn.softplus(self.pi_theta_log) - self.softplus_pi
        self.mul_case_zero = tf.multiply(tf.dtypes.cast(self.input_x < 1e-8, tf.float32), self.case_zero)

        self.case_non_zero = (
            -self.softplus_pi
            + self.pi_theta_log
            + tf.multiply(self.input_x, (tf.log(self.reconstr_x + 1e-8) - self.log_theta_mu_eps))
            + tf.lgamma(self.input_x + self.px_r)
            - tf.lgamma(self.px_r)
            - tf.lgamma(self.input_x + 1)
        )
        self.mul_case_non_zero = tf.multiply(tf.dtypes.cast(self.input_x > 1e-8, tf.float32), self.case_non_zero)

        self.res = self.mul_case_zero + self.mul_case_non_zero
        self.reconstr_loss_x = - tf.reduce_mean(tf.reduce_sum(self.res, axis=1))

        # KL loss
        self.kld_loss_x = tf.reduce_mean(0.5*(tf.reduce_sum(-tf.math.log(self.px_z_v) + self.px_z_v + tf.math.square(self.px_z_m) -1, axis=1))) * self.kl_weight

        ## scATAC reconstruction
        self.py = decoder_atac(tf.concat([self.encoded_y, self.batch_y], 1), self.nlayer, self.hidden_frac, self.chr_list);
        self.reconstr_y = tf.transpose(tf.transpose(self.py) * self.libsize_y)

        ## scATAC loss
        # reconstruction
        bce = tf.keras.losses.BinaryCrossentropy()
        self.reconstr_loss_y = bce(self.input_y, self.reconstr_y) * self.input_dim_y
        self.kld_loss_y = tf.reduce_mean(0.5*(tf.reduce_sum(-self.py_z_v + tf.math.exp(self.py_z_v) + tf.math.square(self.py_z_m)-1, axis=1))) * self.kl_weight

        ## translation on co-assays
        self.translator_encoded_x = translator_atactorna(self.encoded_y, trans_ver = self.trans_ver)
        self.translator_encoded_y = translator_rnatoatac(self.encoded_x, trans_ver = self.trans_ver)

        ## translate to scRNA
        self.px_scale_translator, self.px_dropout_translator, self.px_r_translator = decoder_rna(tf.concat([self.translator_encoded_x, self.batch_x], 1), self.nlayer, self.hidden_frac);
        if self.dispersion == 'genebatch':
            self.px_r_translator = tf.layers.Dense(self.input_dim_x, activation=None, name='translator_yx_px_r_genebatch')(self.batch_y)

        self.px_r_translator = tf.clip_by_value(self.px_r_translator, clip_value_min = -2000000, clip_value_max=15)
        self.px_r_translator = tf.math.exp(self.px_r_translator)
        self.translator_reconstr_x = tf.transpose(tf.transpose(self.px_scale_translator) *self.libsize_x)
        
        ## loss
        self.softplus_pi_translator = tf.nn.softplus(-self.px_dropout_translator)  #  uses log(sigmoid(x)) = -softplus(-x)
        self.log_theta_eps_translator = tf.log(self.px_r_translator + 1e-8)
        self.log_theta_mu_eps_translator = tf.log(self.px_r_translator + self.translator_reconstr_x + 1e-8)
        self.pi_theta_log_translator = -self.px_dropout_translator + tf.multiply(self.px_r_translator, (self.log_theta_eps_translator - self.log_theta_mu_eps_translator))

        self.case_zero_translator = tf.nn.softplus(self.pi_theta_log_translator) - self.softplus_pi_translator
        self.mul_case_zero_translator = tf.multiply(tf.dtypes.cast(self.input_x < 1e-8, tf.float32), self.case_zero_translator)

        self.case_non_zero_translator = (
            -self.softplus_pi_translator
            + self.pi_theta_log_translator
            + tf.multiply(self.input_x, (tf.log(self.translator_reconstr_x + 1e-8) - self.log_theta_mu_eps_translator))
            + tf.lgamma(self.input_x + self.px_r_translator)
            - tf.lgamma(self.px_r_translator)
            - tf.lgamma(self.input_x + 1)
        )
        self.mul_case_non_zero_translator = tf.multiply(tf.dtypes.cast(self.input_x > 1e-8, tf.float32), self.case_non_zero_translator)

        self.res_translator = self.mul_case_zero_translator + self.mul_case_non_zero_translator
        self.translator_loss_x = - tf.reduce_mean(tf.reduce_sum(self.res_translator, axis=1))
        
        ## translate to scATAC
        self.py_translator = decoder_atac(tf.concat([self.translator_encoded_y, self.batch_y], 1), self.nlayer, self.hidden_frac, self.chr_list);
        self.translator_reconstr_y = tf.transpose(tf.transpose(self.py_translator) *self.libsize_y)
        
        ## loss
        self.translator_loss_y = bce(self.input_y, self.translator_reconstr_y)* self.input_dim_y;

        ## optimizers
        self.train_vars_x = [var for var in tf.trainable_variables() if '_x' in var.name];
        self.train_vars_y = [var for var in tf.trainable_variables() if '_y' in var.name];
        self.train_vars_trans_x = [var for var in tf.trainable_variables() if 'translator_yx' in var.name];
        self.train_vars_trans_y = [var for var in tf.trainable_variables() if 'translator_xy' in var.name];
        self.loss_x = self.reconstr_loss_x + self.kl_weight_x * self.kld_loss_x
        self.loss_y = self.reconstr_loss_y + self.kl_weight_y * self.kld_loss_y
        self.optimizer_x = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=0.01).minimize(self.loss_x, var_list=self.train_vars_x );
        self.optimizer_y = tf.train.AdamOptimizer(learning_rate=self.learning_rate_y, epsilon=0.00001).minimize(self.loss_y, var_list=self.train_vars_y );
        self.optimizer_trans_x = tf.train.AdamOptimizer(learning_rate=self.learning_rate_yx).minimize(self.translator_loss_x, var_list=self.train_vars_trans_x );
        self.optimizer_trans_y = tf.train.AdamOptimizer(learning_rate=self.learning_rate_xy).minimize(self.translator_loss_y, var_list=self.train_vars_trans_y );

        self.sess = tf.Session();
        self.sess.run(tf.global_variables_initializer());

    def train(self, data_x, batch_x, data_y, batch_y, data_x_val, batch_x_val, data_y_val, batch_y_val, data_x_co, batch_x_co, data_y_co, batch_y_co, nepoch_warmup_x, nepoch_warmup_y, patience, nepoch_klstart_x, nepoch_klstart_y, output_model,  batch_size, nlayer, save_model=False):
        """
        train in four steps, in each step, part of neural network is optimized meanwhile other layers are frozen.
        early stopping based on tolerance (patience) and maximum epochs defined in each step
        sep_train_index: 1: train scATAC autoencoder; 2: train scRNA autoencoder; 3: translate scATAC to scRNA; 4: translate scRNA to scATAC
        n_iter_step1: document the niter for scATAC autoencoder, once it reaches nepoch_klstart_y, KL will start to warm up
        n_iter_step2: document the niter for scRNA autoencoder, once it reaches nepoch_klstart_x, KL will start to warm up

        """
        val_reconstr_y_loss_list = [];
        val_kl_y_loss_list = [];
        val_reconstr_x_loss_list = [];
        val_kl_x_loss_list = [];
        val_translat_y_loss_list = [];
        val_translat_x_loss_list = [];
        last_improvement=0
        n_iter_step1 = 0 # keep track of the number of epochs for nepoch_klstart_y
        n_iter_step2 = 0 # # keep track of the number of epochs for nepoch_klstart_x

        iter_list1 = []
        iter_list2 = []
        iter_list3 = []
        iter_list4 = []
        loss_val_check_list = []
        my_epochs_max = {}
        my_epochs_max[1] = 500 #min(round(10000000/data_x.shape[0]), max_epoch)
        my_epochs_max[2] = 500 #min(round(10000000/data_y.shape[0]), max_epoch)
        my_epochs_max[3] = 100
        my_epochs_max[4] = 100
        saver = tf.train.Saver()
        sep_train_index = 1
        for iter in range(1, 2000):
            print('iter '+str(iter))
            sys.stdout.flush()
            if sep_train_index == 1:
                if n_iter_step1 < nepoch_klstart_y:
                    kl_weight_y_update = 0
                else:
                    kl_weight_y_update = min(1.0, (n_iter_step1-nepoch_klstart_y)/float(nepoch_warmup_y)) #50
                iter_list1.append(iter)
                iter_list = iter_list1
                for batch_id in range(0, data_y.shape[0]//batch_size +1):
                    data_y_i = data_y[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y.shape[0]),].todense()
                    batch_y_i = batch_y[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y.shape[0]),].todense()
                    self.sess.run(self.optimizer_y, feed_dict={self.input_x: np.zeros(shape=(data_y_i.shape[0], data_x.shape[1]), dtype=np.int32), self.input_y: data_y_i, self.batch_x: np.zeros(shape=(batch_y_i.shape[0], batch_x.shape[1]), dtype=np.int32), self.batch_y: batch_y_i, self.kl_weight_x: 0.0, self.kl_weight_y: kl_weight_y_update});
                    
                        
                n_iter_step1 +=1
                loss_reconstruct_y_val = []
                #loss_kl_y_val = []
                for batch_id in range(0, data_y_val.shape[0]//batch_size +1):
                    data_y_vali = data_y_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_val.shape[0]),].todense()
                    batch_y_vali = batch_y_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_val.shape[0]),].todense()
                    loss_reconstruct_y_val_i, loss_kl_y_val_i = self.get_losses_atac(np.zeros(shape=(data_y_vali.shape[0], data_x.shape[1]), dtype=np.int32), data_y_vali, np.zeros(shape=(batch_y_vali.shape[0], batch_x.shape[1]), dtype=np.int32), batch_y_vali, 0.0, kl_weight_y_update);
                    loss_reconstruct_y_val.append(loss_reconstruct_y_val_i)
                    #loss_kl_y_val.append(loss_kl_y_val_i)
                
                loss_val_check = np.nanmean(np.array(loss_reconstruct_y_val))
                val_reconstr_y_loss_list.append(loss_val_check)
                #val_kl_y_loss_list.append(np.nanmean(np.array(loss_kl_y_val)))
                
            if sep_train_index == 2:
                if n_iter_step2 < nepoch_klstart_x:
                    kl_weight_x_update = 0
                else:
                    kl_weight_x_update = min(1.0, (n_iter_step2-nepoch_klstart_x)/float(nepoch_warmup_x))
                
                iter_list2.append(iter)
                iter_list = iter_list2
                for batch_id in range(0, data_x.shape[0]//batch_size +1):
                    data_x_i = data_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),].todense()
                    batch_x_i = batch_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),].todense()
                    self.sess.run(self.optimizer_x, feed_dict={self.input_x: data_x_i, self.input_y: np.zeros(shape=(data_x_i.shape[0], data_y.shape[1]), dtype=np.int32), self.batch_x: batch_x_i, self.batch_y: np.zeros(shape=(batch_x_i.shape[0], batch_y.shape[1]), dtype=np.int32), self.kl_weight_x: kl_weight_x_update, self.kl_weight_y: 0.0});
                        
                n_iter_step2 +=1
                loss_reconstruct_x_val = []
                #loss_kl_x_val = []
                for batch_id in range(0, data_x_val.shape[0]//batch_size +1):
                    data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    loss_reconstruct_x_val_i, loss_kl_x_val_i = self.get_losses_rna(data_x_vali, np.zeros(shape=(data_x_vali.shape[0], data_y.shape[1]), dtype=np.int32), batch_x_vali, np.zeros(shape=(batch_x_vali.shape[0], batch_y.shape[1]), dtype=np.int32), kl_weight_x_update, 0.0);
                    loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                    #loss_kl_x_val.append(loss_kl_x_val_i)

                loss_val_check = np.nanmean(np.array(loss_reconstruct_x_val))
                val_reconstr_x_loss_list.append(loss_val_check)
                #val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))

                if np.isnan(loss_reconstruct_x_val).any():
                    break
            if sep_train_index == 3:
                iter_list3.append(iter)
                iter_list = iter_list3
                for batch_id in range(0, data_x_co.shape[0]//batch_size +1):
                    data_x_i = data_x_co[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_co.shape[0]),].todense()
                    data_y_i = data_y_co[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_co.shape[0]),].todense()
                    batch_x_i = batch_x_co[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_co.shape[0]),].todense()
                    batch_y_i = batch_y_co[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_co.shape[0]),].todense()
                    self.sess.run(self.optimizer_trans_x, feed_dict={self.input_x: data_x_i, self.input_y: data_y_i, self.batch_x: batch_x_i, self.batch_y:batch_y_i, self.kl_weight_x: 0.0, self.kl_weight_y: 0.0});

                loss_translator_x_val = []
                for batch_id in range(0, data_x_val.shape[0]//batch_size +1):
                    data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    data_y_vali = data_y_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_val.shape[0]),].todense()
                    batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    batch_y_vali = batch_y_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_val.shape[0]),].todense()
                    loss_translator_x_val_i = self.get_losses_translatetorna(data_x_vali, data_y_vali, batch_x_vali, batch_y_vali);
                    loss_translator_x_val.append(loss_translator_x_val_i)
                
                loss_val_check = np.nanmean(np.array(loss_translator_x_val))
                val_translat_x_loss_list.append(loss_val_check)


            if sep_train_index == 4:
                iter_list4.append(iter)
                iter_list = iter_list4
                for batch_id in range(0, data_x_co.shape[0]//batch_size +1):
                    data_x_i = data_x_co[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_co.shape[0]),].todense()
                    data_y_i = data_y_co[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_co.shape[0]),].todense()
                    batch_x_i = batch_x_co[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_co.shape[0]),].todense()
                    batch_y_i = batch_y_co[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_co.shape[0]),].todense()
                    self.sess.run(self.optimizer_trans_y, feed_dict={self.input_x: data_x_i, self.input_y: data_y_i, self.batch_x: batch_x_i, self.batch_y:batch_y_i, self.kl_weight_x: 0.0, self.kl_weight_y: 0.0});

                loss_translator_y_val = []
                for batch_id in range(0, data_x_val.shape[0]//batch_size +1):
                    data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    data_y_vali = data_y_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_val.shape[0]),].todense()
                    batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    batch_y_vali = batch_y_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_y_val.shape[0]),].todense()
                    loss_translator_y_val_i = self.get_losses_translatetoatac(data_x_vali, data_y_vali, batch_x_vali, batch_y_vali);
                    loss_translator_y_val.append(loss_translator_y_val_i)
                
                loss_val_check = np.nanmean(np.array(loss_translator_y_val))
                val_translat_y_loss_list.append(loss_val_check)

            
            if ((iter + 1) % 1 == 0): # check every epoch
                print('loss_val_check: '+str(loss_val_check))
                loss_val_check_list.append(loss_val_check)
                try:
                    loss_val_check_best
                except NameError:
                    loss_val_check_best = loss_val_check
                if loss_val_check < loss_val_check_best:
                    #save_sess = self.sess
                    saver.save(self.sess, output_model+'_step'+str(sep_train_index)+'/mymodel')
                    loss_val_check_best = loss_val_check
                    last_improvement = 0
                else:
                    last_improvement +=1

                if len(loss_val_check_list) > 1:
                    ## decide on early stopping 
                    stop_decision = last_improvement > patience
                    if stop_decision or len(iter_list) == my_epochs_max[sep_train_index]-1:
                        tf.reset_default_graph()
                        saver = tf.train.import_meta_graph(output_model+'_step'+str(sep_train_index)+'/mymodel.meta')
                        saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'_step'+str(sep_train_index)+'/'))
                        print('step'+str(sep_train_index)+' reached minimum, switching to next')
                        last_improvement = 0
                        loss_val_check_list = []
                        del loss_val_check_best
                        sep_train_index +=1
                        if sep_train_index > 4:
                            break

        return iter_list1, iter_list2, iter_list3, iter_list4, val_reconstr_y_loss_list, val_kl_y_loss_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_translat_y_loss_list, val_translat_x_loss_list


    def load(self, output_model):
        """
        load pre-trained model
        """
        saver = tf.train.Saver()
        if os.path.exists(output_model+'_step4/'):
            print('== load existing model from '+ output_model+'_step4/')
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(output_model+'_step4/mymodel.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'_step4/'))

    def predict_embedding(self, data_x, data_y, batch_x, batch_y):
        """
        return scRNA and scATAC projections on VAE embedding layers 
        """
        return self.sess.run([self.encoded_x, self.translator_encoded_x, self.encoded_y, self.translator_encoded_y], feed_dict={self.input_x: data_x, self.input_y: data_y, self.batch_x: batch_x, self.batch_y: batch_y});
    
    def predict_reconstruction(self, data_x, data_y, batch_x, batch_y):
        """
        return reconstructed and translated scRNA and scATAC profiles (incorporating sequencing-depth)
        """
        return self.sess.run([self.translator_reconstr_x, self.translator_reconstr_y, self.reconstr_x, self.reconstr_y], feed_dict={self.input_x: data_x, self.input_y: data_y, self.batch_x: batch_x, self.batch_y: batch_y});
    
    def predict_rnanorm_translation(self, input_dim_x, data_y, batch_y):
        """
        return scRNA profile (normalized) translated from input scATAC, use batch_y as batch_x to make prediction on the same batch
        """
        return self.sess.run(self.px_scale_translator, feed_dict={self.input_x: np.zeros(shape=(data_y.shape[0], input_dim_x), dtype=np.int32), self.input_y: data_y, self.batch_x: batch_y, self.batch_y: batch_y});
        
    def predict_atacnorm_translation(self, data_x, batch_x, input_dim_y):
        """
        return scATAC profile (normalized) translated from input scRNA, use batch_x as batch_y to make prediction on the same batch
        """
        return self.sess.run(self.py_translator, feed_dict={self.input_x: data_x, self.input_y: np.zeros(shape=(data_x.shape[0], input_dim_y), dtype=np.int32), self.batch_x: batch_x, self.batch_y: batch_x});
    
    def get_losses_rna(self, data_x, data_y, batch_x, batch_y, kl_weight_x, kl_weight_y):
        """
        return scRNA reconstruction loss
        """
        return self.sess.run([self.reconstr_loss_x, self.kld_loss_x], feed_dict={self.input_x: data_x, self.input_y: data_y, self.batch_x: batch_x, self.batch_y: batch_y, self.kl_weight_x: kl_weight_x, self.kl_weight_y: kl_weight_y});

    def get_losses_atac(self, data_x, data_y, batch_x, batch_y, kl_weight_x, kl_weight_y):
        """
        return scATAC reconstruction loss
        """
        return self.sess.run([self.reconstr_loss_y, self.kld_loss_y], feed_dict={self.input_x: data_x, self.input_y: data_y, self.batch_x: batch_x, self.batch_y: batch_y, self.kl_weight_x: kl_weight_x, self.kl_weight_y: kl_weight_y});

    def get_losses_translatetoatac(self, data_x, data_y, batch_x, batch_y):
        """
        return scATAC prediction (translated from scRNA) loss
        """
        return self.sess.run(self.translator_loss_y, feed_dict={self.input_x: data_x, self.input_y: data_y, self.batch_x: batch_x, self.batch_y: batch_y});

    def get_losses_translatetorna(self, data_x, data_y, batch_x, batch_y):
        """
        return scRNA prediction (translated from scATAC) loss
        """
        return self.sess.run(self.translator_loss_x, feed_dict={self.input_x: data_x, self.input_y: data_y, self.batch_x: batch_x, self.batch_y: batch_y});

    def restore(self, restore_folder):
        """
        Restore the tensorflow graph stored in restore_folder.
        """
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(restore_folder+'/mymodel.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(restore_folder+'/'))

