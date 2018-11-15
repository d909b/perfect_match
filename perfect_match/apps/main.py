#!/usr/bin/env python2
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
import sys
import numpy as np
import pandas as pd
from os.path import join

# Configure tensorflow not to use the entirety of GPU memory.
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.callbacks import TensorBoard
from perfect_match.apps.util import time_function
from perfect_match.models.model_eval import ModelEvaluation
from perfect_match.apps.evaluate import EvaluationApplication
from perfect_match.data_access.generator import make_keras_generator
from perfect_match.models.baselines.psm import PSM
from perfect_match.models.baselines.psm_pbm import PSM_PBM
from perfect_match.models.baselines.ganite import GANITE
from perfect_match.models.baselines.knn import KNearestNeighbours
from perfect_match.models.baselines.causal_forest import CausalForest
from perfect_match.models.baselines.random_forest import RandomForest
from perfect_match.models.baselines.neural_network import NeuralNetwork
from perfect_match.models.baselines.gaussian_process import GaussianProcess
from perfect_match.models.baselines.tf_neural_network import NeuralNetwork as TFNeuralNetwork
from perfect_match.models.baselines.gradientboosted import GradientBoostedTrees
from perfect_match.models.baselines.bart import BayesianAdditiveRegressionTrees
from perfect_match.models.baselines.ordinary_least_squares import OrdinaryLeastSquares1, OrdinaryLeastSquares2
from perfect_match.apps.parameters import clip_percentage, parse_parameters
from perfect_match.data_access.patient_generator import make_generator, get_last_row_id


class MainApplication(EvaluationApplication):
    def __init__(self, args):
        super(MainApplication, self).__init__(args)

    def setup(self):
        super(MainApplication, self).setup()

    def make_train_generator(self, randomise=True, stratify=True):
        seed = int(np.rint(self.args["seed"]))
        validation_fraction = clip_percentage(self.args["validation_set_fraction"])
        test_fraction = clip_percentage(self.args["test_set_fraction"])

        train_generator, train_steps = make_generator(self.args,
                                                      self.benchmark,
                                                      is_validation=False,
                                                      is_test=False,
                                                      validation_fraction=validation_fraction,
                                                      test_fraction=test_fraction,
                                                      seed=seed,
                                                      randomise=randomise,
                                                      stratify=stratify)
        return train_generator, train_steps

    def make_validation_generator(self, randomise=False):
        seed = int(np.rint(self.args["seed"]))
        validation_fraction = clip_percentage(self.args["validation_set_fraction"])
        test_fraction = clip_percentage(self.args["test_set_fraction"])

        val_generator, val_steps = make_generator(self.args,
                                                  self.benchmark,
                                                  is_validation=True,
                                                  is_test=False,
                                                  validation_fraction=validation_fraction,
                                                  test_fraction=test_fraction,
                                                  seed=seed,
                                                  randomise=randomise)
        return val_generator, val_steps

    def make_test_generator(self, randomise=False, do_not_sample_equalised=False):
        seed = int(np.rint(self.args["seed"]))
        validation_fraction = clip_percentage(self.args["validation_set_fraction"])
        test_fraction = clip_percentage(self.args["test_set_fraction"])

        test_generator, test_steps = make_generator(self.args,
                                                    self.benchmark,
                                                    is_validation=False,
                                                    is_test=True,
                                                    validation_fraction=validation_fraction,
                                                    test_fraction=test_fraction,
                                                    seed=seed,
                                                    randomise=randomise)
        return test_generator, test_steps

    def get_best_model_path(self):
        return join(self.args["output_directory"], "model.npz")

    def get_prediction_path(self):
        return join(self.args["output_directory"], "predictions.csv")

    def get_hyperopt_parameters(self):
        hyper_params = {}

        base_params = {
        }

        hyper_params.update(base_params)
        method = self.args["method"]
        if EvaluationApplication.method_is_neural_network(method):
            if self.args["benchmark"] == "ihdp":
                nn_params = {
                    "batch_size": (16, 100, 200, 500) if not self.args["with_propensity_batch"] else (4, 8, 50, 100,),
                    "num_layers": (1, 2, 3),
                    "num_units": (50, 100, 200,),
                }
                hyper_params.update(nn_params)
            elif self.args["benchmark"] == "jobs":
                nn_params = {
                    "batch_size": (50,),
                    "num_units": (60,),
                }
                hyper_params.update(nn_params)
            else:
                nn_params = {
                    "batch_size": (50,),
                    "num_layers": (2, 3) if not self.args["with_propensity_dropout"] else (3,),
                    "num_units": (40, 60, 80),
                }
                hyper_params.update(nn_params)

        if float(self.args["imbalance_loss_weight"]) != 0.0:
            hyper_params["imbalance_loss_weight"] = (0.1, 1.0, 10.0)

        if method == "ganite":
            hyper_params["ganite_weight_alpha"] = (0.1, 1, 10)
            hyper_params["ganite_weight_beta"] = (0.1, 1, 10)

        return hyper_params

    @time_function("time_steps")
    def time_steps(self, generator, num_steps=10):
        for _ in range(num_steps):
            _ = next(generator)

    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        print("INFO: Started training feature extraction.", file=sys.stderr)

        with_tensorboard = self.args["with_tensorboard"]

        n_jobs = int(np.rint(self.args["n_jobs"]))
        num_epochs = int(np.rint(self.args["num_epochs"]))
        learning_rate = float(self.args["learning_rate"])
        l2_weight = float(self.args["l2_weight"])
        batch_size = int(np.rint(self.args["batch_size"]))
        early_stopping_patience = int(np.rint(self.args["early_stopping_patience"]))
        early_stopping_on_pehe = self.args["early_stopping_on_pehe"]
        imbalance_loss_weight = float(self.args["imbalance_loss_weight"])
        num_layers = int(np.rint(self.args["num_layers"]))
        num_units = int(np.rint(self.args["num_units"]))
        dropout = float(self.args["dropout"])
        method = self.args["method"]
        method_type = MainApplication.get_method_name_map()[method]
        best_model_path = self.get_best_model_path()
        with_propensity_dropout = self.args["with_propensity_dropout"]
        with_pehe_loss = self.args["with_pehe_loss"]
        use_tarnet = self.args["use_tarnet"]
        match_on_covariates = self.args["match_on_covariates"]
        num_randomised_neighbours = int(np.rint(self.args["num_randomised_neighbours"]))
        propensity_batch_probability = float(self.args["propensity_batch_probability"])
        strength_of_assignment_bias = int(np.rint(self.args["strength_of_assignment_bias"]))
        ganite_weight_alpha = float(self.args["ganite_weight_alpha"])
        ganite_weight_beta = float(self.args["ganite_weight_beta"])

        network_params = {
            "with_propensity_dropout": with_propensity_dropout,
            "imbalance_loss_weight": imbalance_loss_weight,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_on_pehe": early_stopping_on_pehe,
            "num_layers": num_layers,
            "num_units": num_units,
            "dropout": dropout,
            "batch_size": batch_size,
            "num_treatments": self.benchmark.get_num_treatments(),
            "input_dim": self.benchmark.get_input_shapes(self.args)[0],
            "output_dim": self.benchmark.get_output_shapes(self.args)[0],
            "best_model_path": best_model_path,
            "l2_weight": l2_weight,
            "learning_rate": learning_rate,
            "with_tensorboard": with_tensorboard,
            "n_jobs": n_jobs,
            "benchmark": self.benchmark,
            "with_pehe_loss": with_pehe_loss,
            "use_tarnet": use_tarnet,
            "strength_of_assignment_bias": strength_of_assignment_bias,
            "ganite_weight_alpha": ganite_weight_alpha,
            "ganite_weight_beta": ganite_weight_beta,
            "propensity_batch_probability": propensity_batch_probability,
            "match_on_covariates": match_on_covariates,
            "num_randomised_neighbours": num_randomised_neighbours,
        }

        num_losses = 1

        train_generator, train_steps = make_keras_generator(self.args,
                                                            train_generator,
                                                            train_steps,
                                                            batch_size=batch_size
                                                              if EvaluationApplication.method_is_neural_network(method)
                                                              else train_steps,
                                                            num_losses=num_losses,
                                                            benchmark=self.benchmark,
                                                            is_train=True)

        inner_val_generator, inner_val_steps = val_generator, val_steps
        val_generator, val_steps = make_keras_generator(self.args,
                                                        inner_val_generator,
                                                        inner_val_steps,
                                                        batch_size=inner_val_steps,
                                                        num_losses=num_losses,
                                                        benchmark=self.benchmark)

        assert train_steps > 0, "You specified a batch_size that is bigger than the size of the train set."
        assert val_steps > 0, "You specified a batch_size that is bigger than the size of the validation set."

        if with_tensorboard:
            tb_folder = join(self.args["output_directory"], "tensorboard")
            tmp_generator, tmp_steps = make_keras_generator(self.args,
                                                            inner_val_generator,
                                                            inner_val_steps,
                                                            batch_size=2,
                                                            num_losses=num_losses,
                                                            benchmark=self.benchmark)
            tb = [MainApplication.build_tensorboard(tmp_generator, tb_folder)]
        else:
            tb = []

        network_params["tensorboard_callback"] = tb

        model = method_type()
        model.build(**network_params)

        if self.args["load_existing"]:
            print("INFO: Loading existing model from", self.args["load_existing"], file=sys.stderr)
            model.load(self.args["load_existing"])

        if self.args["do_train"]:
            if EvaluationApplication.method_is_neural_network(self.args["method"]) and\
               self.args["with_propensity_batch"]:
                adjusted_train_steps = train_steps / self.benchmark.get_num_treatments()
            else:
                adjusted_train_steps = train_steps
            model.fit_generator(train_generator=train_generator,
                                train_steps=adjusted_train_steps,
                                num_epochs=num_epochs,
                                val_generator=val_generator,
                                val_steps=val_steps,
                                batch_size=batch_size)

            model.save(best_model_path)

            ModelEvaluation.evaluate(model, train_generator, train_steps, "train")
            ModelEvaluation.evaluate(model, val_generator, val_steps, "validation")

        return model

    def evaluate_model(self, model, test_generator, test_steps, with_print=True, set_name="test"):
        if with_print:
            print("INFO: Started evaluation.", file=sys.stderr)

        test_generator, test_steps = make_keras_generator(self.args,
                                                          test_generator,
                                                          test_steps,
                                                          batch_size=test_steps,
                                                          num_losses=1,
                                                          benchmark=self.benchmark)

        auc_score = ModelEvaluation.evaluate(model, test_generator,
                                             test_steps, set_name, with_print=with_print)
        cf_score = ModelEvaluation.evaluate_counterfactual(model, test_generator,
                                                           test_steps, self.benchmark,
                                                           set_name + "_cf", with_print=with_print)
        auc_score.update(cf_score)
        return auc_score

    def save_predictions(self, model):
        print("INFO: Saving model predictions.", file=sys.stderr)

        fraction_of_data_set = clip_percentage(self.args["fraction_of_data_set"])
        file_path = self.get_prediction_path()

        generators = [self.make_train_generator, self.make_validation_generator, self.make_test_generator]

        predictions = []
        for generator_fun in generators:
            generator, steps = generator_fun(randomise=False)
            generator, steps = make_keras_generator(self.args,
                                                    generator,
                                                    steps,
                                                    batch_size=1,
                                                    num_losses=1,
                                                    benchmark=self.benchmark)
            steps = int(np.rint(steps * fraction_of_data_set))

            for step in range(steps):
                x, y = next(generator)

                last_id = get_last_row_id()

                predictions.append([last_id, np.squeeze(model.predict(x))])

        df = pd.DataFrame(predictions, columns=["recordId", "prediction"])
        df.to_csv(file_path)

        print("INFO: Saved model predictions to", file_path, file=sys.stderr)

    @staticmethod
    def build_tensorboard(tmp_generator, tb_folder):
        for a_file in os.listdir(tb_folder):
            file_path = join(tb_folder, a_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e, file=sys.stderr)

        tb = TensorBoard(tb_folder, write_graph=False, histogram_freq=1, write_grads=True, write_images=False)
        x, y = next(tmp_generator)

        tb.validation_data = x
        tb.validation_data[1] = np.expand_dims(tb.validation_data[1], axis=-1)
        if isinstance(y, list):
            num_targets = len(y)
            tb.validation_data += [y[0]] + y[1:]
        else:
            tb.validation_data += [y]
            num_targets = 1

        tb.validation_data += [np.ones(x[0].shape[0])] * num_targets + [0.0]
        return tb

    @staticmethod
    def get_method_name_map():
        return {
            'knn': KNearestNeighbours,
            'ols1': OrdinaryLeastSquares1,
            'ols2': OrdinaryLeastSquares2,
            'cf': CausalForest,
            'rf': RandomForest,
            'bart': BayesianAdditiveRegressionTrees,
            'nn': TFNeuralNetwork,
            'nn+': NeuralNetwork,
            'xgb': GradientBoostedTrees,
            'gp': GaussianProcess,
            'psm': PSM,
            'psmpbm': PSM_PBM,
            'ganite': GANITE,
        }


if __name__ == '__main__':
    app = MainApplication(parse_parameters())
    app.run()
