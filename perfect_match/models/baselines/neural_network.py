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

import sys
from keras.callbacks import EarlyStopping

from perfect_match.models.model_builder import ModelBuilder
from perfect_match.models.baselines.baseline import Baseline
from perfect_match.models.cf_early_stopping import CounterfactualEarlyStopping
from perfect_match.models.model_factory import ModelFactoryCheckpoint, ModelFactory


class NeuralNetwork(Baseline):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.callbacks = []

    def load(self, path):
        weight_list = ModelFactory.load_weights(path)
        self.model.set_weights(weight_list)

    def _build(self, **kwargs):
        self.kwargs = kwargs
        self.best_model_path = kwargs["best_model_path"]
        if kwargs["use_tarnet"]:
            return ModelBuilder.build_tarnet(**kwargs)
        else:
            return ModelBuilder.build_simple(**kwargs)

    def make_callbacks(self, val_generator, val_steps, **kwargs):
        with_propensity_dropout = kwargs["with_propensity_dropout"]
        early_stopping_patience = kwargs["early_stopping_patience"]
        early_stopping_on_pehe = kwargs["early_stopping_on_pehe"]
        best_model_path = kwargs["best_model_path"]
        tb = kwargs["tensorboard_callback"]

        monitor_name = "val_dynamic_stitch_loss" if with_propensity_dropout else "val_loss"
        monitor_mode = "min"

        if early_stopping_on_pehe:
            print("INFO: Using early stopping on nearest neighbour PEHE.", file=sys.stderr)
            early_stopping = CounterfactualEarlyStopping(patience=early_stopping_patience,
                                                         val_generator=val_generator,
                                                         val_steps=val_steps,
                                                         benchmark=self.kwargs["benchmark"],
                                                         model=self.model,
                                                         mode=monitor_mode,
                                                         min_delta=0.0001)
        else:
            print("INFO: Using early stopping on the main loss.", file=sys.stderr)
            early_stopping = EarlyStopping(patience=early_stopping_patience,
                                           monitor=monitor_name,
                                           mode=monitor_mode,
                                           min_delta=0.0001)

        callbacks = [
            early_stopping,
            ModelFactoryCheckpoint(filepath=best_model_path,
                                   save_best_only=True,
                                   save_weights_only=True,
                                   monitor=monitor_name,
                                   mode=monitor_mode),
        ] + tb
        return callbacks

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size):
        # Save once in case training does not converge.
        ModelFactory.save_weights(self.model, self.best_model_path)

        self.model.fit_generator(train_generator,
                                 train_steps,
                                 epochs=num_epochs,
                                 validation_data=val_generator,
                                 validation_steps=val_steps,
                                 callbacks=self.make_callbacks(val_generator,
                                                               val_steps,
                                                               **self.kwargs),
                                 verbose=2,
                                 workers=0)

        print("INFO: Resetting to best encountered model at", self.best_model_path, ".", file=sys.stderr)

        # Reset to the best model observed in training.
        weights = ModelFactory.load_weights(self.best_model_path)
        self.model.set_weights(weights)

        # Sanity check.
        # print("Sanity check (train):", self.model.evaluate_generator(train_generator, train_steps), file=sys.stderr)
        # print("Sanity check (val):", self.model.evaluate_generator(val_generator, val_steps), file=sys.stderr)
