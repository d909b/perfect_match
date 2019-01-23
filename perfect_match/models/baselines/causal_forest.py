"""
Copyright (C) 2019  anonymised author, anonymised institution

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
from functools import partial
from perfect_match.models.baselines.baseline import Baseline, PickleableMixin


class CausalForest(PickleableMixin, Baseline):
    def __init__(self):
        super(CausalForest, self).__init__()
        self.bart = None

    def install_grf(self):
        from rpy2.robjects.packages import importr
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        import rpy2.robjects as robjects

        robjects.r.options(download_file_method='curl')

        package_names = ["grf"]
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=0)
        utils.chooseCRANmirror(ind=0)

        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))

        return importr("grf")

    def _build(self, **kwargs):
        from rpy2.robjects import numpy2ri
        grf = self.install_grf()

        self.grf = grf
        numpy2ri.activate()

        return None

    def predict_for_model(self, model, x):
        import rpy2.robjects as robjects
        r = robjects.r
        result = r.predict(self.model, self.preprocess(x))
        return np.array(result[0])

    def fit_generator_for_model(self, model, train_generator, train_steps, val_generator, val_steps, num_epochs):
        from rpy2.robjects.vectors import StrVector, FactorVector, FloatVector, IntVector
        all_outputs = []
        for _ in range(train_steps):
            generator_output = next(train_generator)
            x, y = generator_output[0], generator_output[1]
            all_outputs.append((self.preprocess(x), x[1], y))
        x, t, y = map(partial(np.concatenate, axis=0), zip(*all_outputs))

        self.model = self.grf.causal_forest(x,
                                            FloatVector([float(yy) for yy in y]),
                                            FloatVector([float(tt) for tt in t]), seed=909)

    def preprocess(self, x):
        return x[0]

    def postprocess(self, y):
        return y[:, -1]
