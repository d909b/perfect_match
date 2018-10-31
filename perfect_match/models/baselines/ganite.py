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
from perfect_match.models.baselines.baseline import Baseline
from perfect_match.models.baselines.ganite_package.ganite_model import GANITEModel


class GANITE(Baseline):
    def __init__(self):
        super(GANITE, self).__init__()
        self.callbacks = []

    def load(self, path):
        self.model.load(path)

    def _build(self, **kwargs):
        self.best_model_path = kwargs["best_model_path"]
        self.learning_rate = kwargs["learning_rate"]
        self.dropout = kwargs["dropout"]
        self.l2_weight = kwargs["l2_weight"]
        self.num_units = kwargs["num_units"]
        self.num_layers = kwargs["num_layers"]
        self.num_treatments = kwargs["num_treatments"]
        self.imbalance_loss_weight = kwargs["imbalance_loss_weight"]
        self.early_stopping_patience = kwargs["early_stopping_patience"]
        self.early_stopping_on_pehe = kwargs["early_stopping_on_pehe"]
        self.input_dim = kwargs["input_dim"]
        self.output_dim = kwargs["output_dim"]
        self.ganite_weight_alpha = kwargs["ganite_weight_alpha"]
        self.ganite_weight_beta = kwargs["ganite_weight_beta"]
        return GANITEModel(self.input_dim,
                           self.output_dim,
                           num_units=self.num_units,
                           dropout=self.dropout,
                           l2_weight=self.l2_weight,
                           learning_rate=self.learning_rate,
                           num_layers=self.num_layers,
                           num_treatments=self.num_treatments,
                           with_bn=False,
                           nonlinearity="elu",
                           alpha=self.ganite_weight_alpha,
                           beta=self.ganite_weight_beta)

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size):
        # num_epochs = int(np.ceil(3000 / batch_size))
        self.model.train(train_generator,
                         train_steps,
                         num_epochs=num_epochs,
                         learning_rate=self.learning_rate,
                         val_generator=val_generator,
                         val_steps=val_steps,
                         dropout=self.dropout,
                         l2_weight=self.l2_weight,
                         imbalance_loss_weight=self.imbalance_loss_weight,
                         checkpoint_path=self.best_model_path,
                         early_stopping_patience=self.early_stopping_patience,
                         early_stopping_on_pehe=self.early_stopping_on_pehe)
