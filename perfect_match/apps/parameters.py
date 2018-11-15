"""
Copyright (C) 2018  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function

import os
from argparse import ArgumentParser, Action, ArgumentTypeError


class ReadableDir(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise ArgumentTypeError("readable_dir:{} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise ArgumentTypeError("readable_dir:{} is not a readable dir".format(prospective_dir))


def parse_parameters():
    parser = ArgumentParser(description='Implicit ensemble.')
    parser.add_argument("--dataset", action=ReadableDir, required=True,
                        help="Folder containing the data set to be loaded.")
    parser.add_argument("--seed", type=int, default=909,
                        help="Seed for the random number generator.")
    parser.add_argument("--output_directory", default="./models",
                        help="Base directory of all output files.")
    parser.add_argument("--model_name", default="forecast.h5.npz",
                        help="Base directory of all output files.")
    parser.add_argument("--load_existing", default="",
                        help="Existing model to load.")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Number of processes to use where available for multitasking.")
    parser.add_argument("--learning_rate", default=0.0001, type=float,
                        help="Learning rate to use for training.")
    parser.add_argument("--l2_weight", default=0.0, type=float,
                        help="L2 weight decay used on neural network weights.")
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size to use for training.")
    parser.add_argument("--early_stopping_patience", type=int, default=12,
                        help="Number of stale epochs to wait before terminating training")
    parser.add_argument("--num_units", type=int, default=8,
                        help="Number of neurons to use in DNN layers.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers to use in DNNs.")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="Value of the dropout parameter used in training in the network.")
    parser.add_argument("--imbalance_loss_weight", default=0.0, type=float,
                        help="Value of the imbalance penalty weight for balancing the learnt representation.")
    parser.add_argument("--fraction_of_data_set", type=float, default=1,
                        help="Fraction of time_series to use for folds.")
    parser.add_argument("--validation_set_fraction", type=float, default=0.27,
                        help="Fraction of time_series to hold out for the validation set.")
    parser.add_argument("--test_set_fraction", type=float, default=0.1,
                        help="Fraction of time_series to hold out for the test set.")
    parser.add_argument("--num_hyperopt_runs", type=int, default=35,
                        help="Number of hyperopt runs to perform.")
    parser.add_argument("--hyperopt_offset", type=int, default=0,
                        help="Offset at which to start the hyperopt runs.")
    parser.add_argument("--tcga_num_features", type=int, default=50,
                        help="Number of features to use from TCGA dataset.")
    parser.add_argument("--experiment_index", type=int, default=0,
                        help="Index into repeatable experiments' datasets.")
    parser.add_argument("--num_treatments", type=int, default=2,
                        help="Number of treatments to use when benchmark supports variable numbers of treatments.")
    parser.add_argument("--num_randomised_neighbours", type=int, default=6,
                        help="Number of neighbours to use for randomised match assignment in PM.")
    parser.add_argument("--strength_of_assignment_bias", type=int, default=10,
                        help="Strenght of assignment bias (kappa) to use for semi-synthetic datasets.")
    parser.add_argument("--propensity_batch_probability", type=float, default=1.0,
                        help="Fraction of batch samples matched with their propensity matched nearest neighbor.")
    parser.add_argument("--ganite_weight_alpha", type=float, default=1.0,
                        help="Supervised loss weight alpha for the counterfactual block when using method = GANITE.")
    parser.add_argument("--ganite_weight_beta", type=float, default=1.0,
                        help="Supervised loss weight beta for the ITE block when using method = GANITE.")

    parser.add_argument("--benchmark", default="tcga",
                        help="Benchmark dataset to use. One of ['news', 'tcga', 'ihdp', 'jobs'].")
    parser.add_argument("--method", default="ols1",
                        help="Method to use. One of "
                             "['knn', 'ols1', 'ols2', 'cf', 'rf', 'bart', 'nn', 'nn+', 'xgb', 'gp', 'psm', 'ganite'].")

    parser.set_defaults(with_rnaseq=False)
    parser.add_argument("--with_rnaseq", dest='with_rnaseq', action='store_true',
                        help="Whether or not to use RNASeq data.")
    parser.set_defaults(use_tarnet=True)
    parser.add_argument("--do_not_use_tarnet", dest='use_tarnet', action='store_false',
                        help="Whether or not to use the TARNET architecture.")
    
    parser.set_defaults(do_train=False)
    parser.add_argument("--do_train", dest='do_train', action='store_true',
                        help="Whether or not to train a model.")
    parser.set_defaults(do_hyperopt=False)
    parser.add_argument("--do_hyperopt", dest='do_hyperopt', action='store_true',
                        help="Whether or not to perform hyperparameter optimisation.")
    parser.set_defaults(do_evaluate=False)
    parser.add_argument("--do_evaluate", dest='do_evaluate', action='store_true',
                        help="Whether or not to evaluate a model.")
    parser.set_defaults(hyperopt_against_eval_set=False)
    parser.add_argument("--hyperopt_against_eval_set", dest='hyperopt_against_eval_set', action='store_true',
                        help="Whether or not to evaluate hyperopt runs against the evaluation set.")
    parser.set_defaults(copy_to_local=False)
    parser.add_argument("--copy_to_local", dest='copy_to_local', action='store_true',
                        help="Whether or not to copy the dataset to a local cache before training.")
    parser.set_defaults(do_hyperopt_on_lsf=False)
    parser.add_argument("--do_hyperopt_on_lsf", dest='do_hyperopt_on_lsf', action='store_true',
                        help="Whether or not to perform hyperparameter optimisation split into multiple jobs on LSF.")
    parser.set_defaults(do_merge_lsf=False)
    parser.add_argument("--do_merge_lsf", dest='do_merge_lsf', action='store_true',
                        help="Whether or not to merge LSF hyperopt runs.")
    parser.set_defaults(with_tensorboard=False)
    parser.add_argument("--with_tensorboard", dest='with_tensorboard', action='store_true',
                        help="Whether or not to serve tensorboard data.")
    parser.set_defaults(with_propensity_dropout=False)
    parser.add_argument("--with_propensity_dropout", dest='with_propensity_dropout', action='store_true',
                        help="Whether or not to use propensity dropout.")
    parser.set_defaults(with_propensity_batch=False)
    parser.add_argument("--with_propensity_batch", dest='with_propensity_batch', action='store_true',
                        help="Whether or not to use propensity batching.")
    parser.set_defaults(early_stopping_on_pehe=False)
    parser.add_argument("--early_stopping_on_pehe", dest='early_stopping_on_pehe', action='store_true',
                        help="Whether or not to use early stopping on nearest-neighbour PEHE.")
    parser.set_defaults(with_pehe_loss=False)
    parser.add_argument("--with_pehe_loss", dest='with_pehe_loss', action='store_true',
                        help="Whether or not to use the PEHE objective.")
    parser.set_defaults(match_on_covariates=False)
    parser.add_argument("--match_on_covariates", dest='match_on_covariates', action='store_true',
                        help="Whether or not to match on covariates (alternative is to match on propensity score).")

    parser.set_defaults(save_predictions=True)
    parser.add_argument("--do_not_save_predictions", dest='save_predictions', action='store_false',
                        help="Whether or not to save predictions.")
    parser.add_argument("--save_predictions", dest='save_predictions', action='store_true',
                        help="Whether or not to save predictions.")

    return vars(parser.parse_args())


def clip_percentage(value):
    return max(0., min(1., float(value)))
