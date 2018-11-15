# SOURCE: https://github.com/clinicalml/cfrnet, MIT-License
from __future__ import print_function

import sys
import numpy as np
from util import *
import tensorflow as tf
from perfect_match.models.pehe_loss import pehe_loss
from perfect_match.models.model_eval import ModelEvaluation


class CFRNet(object):
    """
    cfr_net implements the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976
    This file contains the class cfr_net as well as helper functions.
    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """
    def __init__(self, input_dim, num_units, num_layers=1, nonlinearity="elu",
                 weight_initialisation_std=0.1, **kwargs):
        self.variables = {}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.weight_decay_loss = 0
        self.imbalance_loss_weight_param = kwargs["imbalance_loss_weight"]
        self.benchmark = kwargs["benchmark"]
        self.with_pehe_loss = kwargs["with_pehe_loss"]
        self.num_treatments = kwargs["num_treatments"]
        self.nonlinearity = get_nonlinearity_by_name(nonlinearity)

        self._build_graph(input_dim, num_units,
                          num_representation_layers=num_layers,
                          num_regression_layers=num_layers,
                          weight_initialisation_std=weight_initialisation_std)

    def load(self, path):
        saver = tf.train.Saver(var_list=self.variables)
        saver.restore(self.sess, path)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.weight_decay_loss += wd * tf.nn.l2_loss(var)
        return var

    def _build_graph(self, input_dim, num_units,
                     num_representation_layers, num_regression_layers, weight_initialisation_std,
                     reweight_sample=False, loss_function="l2",
                     imbalance_penalty_function="wass", rbf_sigma=0.1,
                     wass_lambda=10.0, wass_iterations=10, wass_bpt=True):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):
        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """
        ''' Initialize input placeholders '''
        self.x = tf.placeholder("float", shape=[None, input_dim], name='x')
        self.t = tf.placeholder("float", shape=[None, 1], name='t')
        self.y_ = tf.placeholder("float", shape=[None, 1], name='y_')

        ''' Parameter placeholders '''
        self.imbalance_loss_weight = tf.placeholder("float", name='r_alpha')
        self.l2_weight = tf.placeholder("float", name='r_lambda')
        self.dropout_representation = tf.placeholder("float", name='dropout_in')
        self.dropout_regression = tf.placeholder("float", name='dropout_out')
        self.p_t = tf.placeholder("float", name='p_treated')

        dim_input = input_dim
        dim_in = num_units
        dim_out = num_units

        weights_in, biases_in = [], []

        if num_representation_layers == 0:
            dim_in = dim_input
        if num_regression_layers == 0:
            dim_out = dim_in

        ''' Construct input/representation layers '''
        h_rep, weights_in, biases_in = build_mlp(self.x, num_representation_layers, dim_in,
                                                 self.dropout_representation, self.nonlinearity,
                                                 weight_initialisation_std=weight_initialisation_std)

        # Normalize representation.
        h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))

        ''' Construct ouput layers '''
        y, y_concat, weights_out, weights_pred = self._build_output_graph(h_rep_norm, self.t, dim_in, dim_out,
                                                                          self.dropout_regression,
                                                                          num_regression_layers,
                                                                          weight_initialisation_std)

        ''' Compute sample reweighting '''
        if reweight_sample:
            w_t = self.t/(2*self.p_t)
            w_c = (1-self.t)/(2*(1-self.p_t))
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if self.with_pehe_loss:
            risk = pred_error = tf.reduce_mean(sample_weight*tf.square(self.y_ - y)) + \
                                pehe_loss(self.y_, y_concat, self.t, self.x, self.num_treatments) / 10.
        elif loss_function == 'log':
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            res = self.y_*tf.log(y) + (1.0-self.y_)*tf.log(1.0-y)

            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        else:
            risk = tf.reduce_mean(sample_weight*tf.square(self.y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(self.y_ - y)))

        ''' Regularization '''
        for i in range(0, num_representation_layers):
            self.weight_decay_loss += tf.nn.l2_loss(weights_in[i])

        p_ipm = 0.5

        if self.imbalance_loss_weight_param == 0.0:
            imb_dist = tf.reduce_mean(self.t)
            imb_error = 0
        elif imbalance_penalty_function == 'mmd2_rbf':
            imb_dist = mmd2_rbf(h_rep_norm, self.t, p_ipm, rbf_sigma)
            imb_error = self.imbalance_loss_weight * imb_dist
        elif imbalance_penalty_function == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm, self.t, p_ipm)
            imb_error = self.imbalance_loss_weight * mmd2_lin(h_rep_norm, self.t, p_ipm)
        elif imbalance_penalty_function == 'mmd_rbf':
            imb_dist = tf.abs(mmd2_rbf(h_rep_norm, self.t, p_ipm, rbf_sigma))
            imb_error = safe_sqrt(tf.square(self.imbalance_loss_weight) * imb_dist)
        elif imbalance_penalty_function == 'mmd_lin':
            imb_dist = mmd2_lin(h_rep_norm, self.t, p_ipm)
            imb_error = safe_sqrt(tf.square(self.imbalance_loss_weight) * imb_dist)
        elif imbalance_penalty_function == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm, self.t, p_ipm, sq=True,
                                            its=wass_iterations, lam=wass_lambda, backpropT=wass_bpt)
            imb_error = self.imbalance_loss_weight * imb_dist
            self.imb_mat = imb_mat  # FOR DEBUG
        elif imbalance_penalty_function == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm, self.t, p_ipm, sq=True,
                                            its=wass_iterations, lam=wass_lambda, backpropT=wass_bpt)
            imb_error = self.imbalance_loss_weight * imb_dist
            self.imb_mat = imb_mat  # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm, p_ipm, self.t)
            imb_error = self.imbalance_loss_weight * imb_dist

        ''' Total error '''
        tot_error = risk
        if self.imbalance_loss_weight_param != 0.0:
            tot_error = tot_error + imb_error
        tot_error = tot_error + self.l2_weight*self.weight_decay_loss

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm

    def _build_output(self, h_input, dim_in, dim_out, dropout_regression, num_regression_layers, weight_initialisation_std):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*num_regression_layers)

        weights_out = []; biases_out = []

        for i in range(0, num_regression_layers):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                                     stddev=weight_initialisation_std/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlinearity(z))
            h_out[i+1] = tf.nn.dropout(h_out[i+1], 1.0 - dropout_regression)

        weights_pred = self._create_variable(tf.random_normal([dim_out, 1],
                                                              stddev=weight_initialisation_std/np.sqrt(dim_out)),
                                             'w_pred')
        
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        self.weight_decay_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dropout_representation, dropout_regression,
                            num_regression_layers, weight_initialisation_std):
        ''' Construct output/regression layers '''
        i0 = tf.to_int32(tf.where(t < 1)[:, 0])
        i1 = tf.to_int32(tf.where(t > 0)[:, 0])

        rep0 = tf.gather(rep, i0)
        rep1 = tf.gather(rep, i1)

        y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in,
                                                             dropout_representation, dropout_regression,
                                                             num_regression_layers,
                                                             weight_initialisation_std)
        y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in,
                                                             dropout_representation, dropout_regression,
                                                             num_regression_layers,
                                                             weight_initialisation_std)

        y = tf.dynamic_stitch([i0, i1], [y0, y1])
        weights_out = weights_out0 + weights_out1
        weights_pred = weights_pred0 + weights_pred1

        y_concat = tf.concat([y0, y1], axis=0)

        return y, y_concat, weights_out, weights_pred

    def train(self, train_generator, train_steps, val_generator, val_steps, num_epochs,
              learning_rate, learning_rate_decay=0.97, iterations_per_decay=100,
              dropout=0.0, imbalance_loss_weight=0.0, l2_weight=0.0, checkpoint_path="",
              early_stopping_patience=12, early_stopping_on_pehe=False):
        saver = tf.train.Saver(var_list=self.variables, max_to_keep=0)
        global_step = tf.Variable(0, trainable=False, dtype="int64")

        lr = tf.train.exponential_decay(learning_rate, global_step,
                                        iterations_per_decay, learning_rate_decay, staircase=True)

        opt = tf.train.AdamOptimizer(lr)
        train_step = opt.minimize(self.tot_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())

        best_val_loss, num_epochs_without_improvement = np.finfo(float).max, 0
        for epoch_idx in range(num_epochs):
            train_losses = self.run_generator(train_generator, train_steps, dropout,
                                              imbalance_loss_weight, l2_weight, train_step)

            val_losses = self.run_generator(val_generator, val_steps, 0, 0, 0)

            if early_stopping_on_pehe:
                score_dict = ModelEvaluation.evaluate_counterfactual(self,
                                                                     val_generator,
                                                                     val_steps,
                                                                     self.benchmark,
                                                                     set_name="val",
                                                                     with_print=True,
                                                                     stateful_benchmark=True)

                current_val_loss = score_dict["cf_pehe_nn"]
            else:
                current_val_loss = val_losses[1]
            do_save = current_val_loss < best_val_loss
            if do_save:
                num_epochs_without_improvement = 0
                best_val_loss = current_val_loss
                saver.save(self.sess, checkpoint_path)
            else:
                num_epochs_without_improvement += 1

            self.print_losses(epoch_idx, num_epochs, train_losses, val_losses, do_save)

            if num_epochs_without_improvement >= early_stopping_patience:
                break

    def print_losses(self, epoch_idx, num_epochs, train_losses, val_losses, did_save=False):
        print("Epoch [{:04d}/{:04d}] {:} TRAIN: {:.3f} MSE: {:.3f} IMB: {:.3f} VAL: {:.3f} vPRED {:.3f} vIMB {:.3f}"
              .format(
                  epoch_idx, num_epochs,
                  "xx" if did_save else "::",
                  train_losses[0], train_losses[1], train_losses[2],
                  val_losses[0], val_losses[1], val_losses[2],
              ),
              file=sys.stderr)

    def run_generator(self, generator, steps, dropout, imbalance_loss_weight, l2_weight, train_step=None):
        losses = []
        for iter_idx in range(steps):
            (x_batch, t_batch), y_batch = next(generator)
            t_batch = np.expand_dims(t_batch, axis=-1)
            y_batch = np.expand_dims(y_batch, axis=-1)

            feed_dict = {
                self.x: x_batch,
                self.t: t_batch,
                self.y_: y_batch,
                self.dropout_regression: dropout,
                self.dropout_representation: dropout,
                self.imbalance_loss_weight: imbalance_loss_weight,
                self.l2_weight: l2_weight,
                self.p_t: 0.5
            }

            if train_step is not None:
                self.sess.run(train_step, feed_dict=feed_dict)

            losses.append(self.sess.run([self.tot_loss, self.pred_loss, self.imb_dist],
                                        feed_dict=feed_dict))
        return np.mean(losses, axis=0)

    def predict(self, x):
        y_pred = self.sess.run(self.output, feed_dict={
            self.x: x[0],
            self.t: np.expand_dims(x[1], axis=-1),
            self.dropout_representation: 0.0,
            self.dropout_regression: 0.0,
            self.l2_weight: 0.0
        })
        return y_pred
