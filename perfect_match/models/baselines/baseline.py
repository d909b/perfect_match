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
import numpy as np
import pandas as pd
from functools import partial
from perfect_match.models.model_factory import ModelFactory


class Baseline(object):
    def __init__(self):
        self.model = None

    @staticmethod
    def to_data_frame(x):
        return pd.DataFrame(data=x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

    def _build(self, **kwargs):
        return None

    def build(self, **kwargs):
        self.model = self._build(**kwargs)

    def preprocess(self, x):
        return x

    def postprocess(self, y):
        return y

    def load(self, path):
        pass

    def save(self, path):
        pass

    def predict_for_model(self, model, x):
        if hasattr(self.model, "predict_proba"):
            return self.postprocess(model.predict_proba(self.preprocess(x)))
        else:
            return self.postprocess(model.predict(self.preprocess(x)))

    def predict(self, x):
        return self.predict_for_model(self.model, x)

    def fit_generator_for_model(self, model, train_generator, train_steps, val_generator, val_steps, num_epochs):
        x, y = self.collect_generator(train_generator, train_steps)
        model.fit(x, y)

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size):
        self.fit_generator_for_model(self.model, train_generator, train_steps, val_generator, val_steps, num_epochs)

    def collect_generator(self, generator, generator_steps):
        all_outputs = []
        for _ in range(generator_steps):
            generator_output = next(generator)
            x, y = generator_output[0], generator_output[1]
            all_outputs.append((self.preprocess(x), y))
        return map(partial(np.concatenate, axis=0), zip(*all_outputs))


class PickleableMixin(object):
    def load(self, path):
        self.model = ModelFactory.load_object(path)

    def save(self, path):
        ModelFactory.save_object(self.model, path)
