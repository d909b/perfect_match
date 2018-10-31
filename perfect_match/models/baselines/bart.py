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
import numpy as np
from perfect_match.models.baselines.baseline import Baseline, PickleableMixin


class BayesianAdditiveRegressionTrees(PickleableMixin, Baseline):
    def __init__(self):
        super(BayesianAdditiveRegressionTrees, self).__init__()
        self.bart = None

    def install_bart(self):
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.packages import importr
        from rpy2.robjects.vectors import StrVector
        import rpy2.robjects as robjects

        robjects.r.options(download_file_method='curl')

        # install.packages("rJava")
        rj = importr("rJava", robject_translations={'.env': 'rj_env'})
        rj._jinit(parameters="-Xmx16g", force_init=True)
        print("rJava heap size is", np.array(rj._jcall(rj._jnew("java/lang/Runtime"), "J", "maxMemory"))[0] / 1e9,
              "GB.", file=sys.stderr)

        package_names = ["bartMachine"]
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=0)
        utils.chooseCRANmirror(ind=0)

        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))

        return importr("bartMachine")

    def _build(self, **kwargs):
        from rpy2.robjects import numpy2ri, pandas2ri
        n_jobs = int(np.rint(kwargs["n_jobs"]))

        bart = self.install_bart()
        bart.set_bart_machine_num_cores(n_jobs)

        self.bart = bart
        numpy2ri.activate()
        pandas2ri.activate()

        return None

    def predict_for_model(self, model, x):
        import rpy2.robjects as robjects
        r = robjects.r
        return np.array(r.predict(self.model, Baseline.to_data_frame(self.preprocess(x))))

    def fit_generator_for_model(self, model, train_generator, train_steps, val_generator, val_steps, num_epochs):
        from rpy2.robjects.vectors import StrVector, IntVector, FactorVector, FloatVector
        x, y = self.collect_generator(train_generator, train_steps)

        self.model = self.bart.bartMachine(X=Baseline.to_data_frame(x),
                                           y=FloatVector([yy for yy in y]),
                                           mem_cache_for_speed=False,
                                           seed=909,
                                           run_in_sample=False)

    def preprocess(self, x):
        return np.concatenate([x[0], np.atleast_2d(np.expand_dims(x[1], axis=-1))], axis=-1)

    def postprocess(self, y):
        return y[:, -1]
