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
from sklearn.ensemble import GradientBoostingClassifier
from perfect_match.models.baselines.baseline import Baseline, PickleableMixin


class GradientBoostedTrees(PickleableMixin, Baseline):
    def __init__(self):
        super(GradientBoostedTrees, self).__init__()

    def _build(self, **kwargs):
        num_units = int(np.rint(kwargs["num_units"]))
        num_layers = int(np.rint(kwargs["num_layers"]))
        return GradientBoostingClassifier(n_estimators=num_units, max_depth=num_layers)

    def preprocess(self, x):
        return np.concatenate([x[0], np.atleast_2d(np.expand_dims(x[1], axis=-1))], axis=-1)

    def postprocess(self, y):
        return y[:, -1]
