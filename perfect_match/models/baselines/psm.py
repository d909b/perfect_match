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
import pandas as pd
from perfect_match.models.baselines.neural_network import NeuralNetwork


class PSM(NeuralNetwork):
    def __init__(self):
        super(PSM, self).__init__()

    def install_matchit(self):
        from rpy2.robjects.packages import importr
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        import rpy2.robjects as robjects

        package_names = ["MatchIt"]

        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            robjects.r.options(download_file_method='curl')
            utils = rpackages.importr('utils')
            utils.chooseCRANmirror(ind=0)
            utils.chooseCRANmirror(ind=0)
            utils.install_packages(StrVector(names_to_install))

        return importr("MatchIt")

    def _build(self, **kwargs):
        from rpy2.robjects import numpy2ri, pandas2ri
        match_it = self.install_matchit()

        self.num_treatments = kwargs["num_treatments"]
        self.batch_size = kwargs["batch_size"]
        self.match_it = match_it
        numpy2ri.activate()
        pandas2ri.activate()

        return super(PSM, self)._build(**kwargs)

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size):
        train_generator, train_steps = self.get_matched_generator(train_generator, train_steps)
        super(PSM, self).fit_generator(train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size)

    def get_matched_generator(self, train_generator, train_steps):
        from rpy2.robjects import pandas2ri
        from rpy2.robjects import Formula

        all_x, all_y = [], []
        for _ in range(train_steps):
            x, y = next(train_generator)
            all_x.append(x)
            all_y.append(y)

        x, t, y = np.concatenate(map(lambda x: x[0], all_x), axis=0), \
                  np.concatenate(map(lambda x: x[1], all_x), axis=0), \
                  np.concatenate(all_y, axis=0)

        num_features = x.shape[-1]

        formula = Formula('t ~ ' + "+".join(["x" + str(i) for i in range(num_features)]))
        env = formula.environment
        env['x'] = x
        env['t'] = t

        t_indices = map(lambda t_idx: np.where(t == t_idx)[0], range(self.num_treatments))
        t_lens = map(lambda x: len(x), t_indices)

        undersample = True
        base_treatment_idx = np.argmin(t_lens) if undersample else np.argmax(t_lens)
        base_indices = t_indices[base_treatment_idx]

        outer_x, outer_t, outer_y = [], [], []
        for treatment_idx in range(self.num_treatments):
            if treatment_idx == base_treatment_idx:
                inner_x, inner_t, inner_y = x[base_indices], t[base_indices], y[base_indices]
            else:
                other_indices = t_indices[treatment_idx]

                this_x = np.concatenate([x[base_indices], x[other_indices]], axis=0)
                this_t = np.concatenate([[0]*len(base_indices), [1]*len(other_indices)], axis=0)
                this_y = np.concatenate([y[base_indices], y[other_indices]], axis=0)

                data = pd.DataFrame(data=np.column_stack([this_x, this_t, this_y]),
                                    index=np.arange(len(this_x)),
                                    columns=["x" + str(i) for i in range(num_features)] + ["t", "y"])

                try:
                    out_data = pandas2ri.ri2py_dataframe(
                        self.match_it.match_data(self.match_it.matchit(formula, data=data, method="nearest"))
                    ).values
                    inner_x, inner_t, inner_y = out_data[:, :-4], out_data[:, -4], out_data[:, -3]
                except:
                    print("WARN: MatchIt failed.", file=sys.stderr)
                    inner_x, inner_t, inner_y = x[other_indices[0:1]], np.array([1]), y[other_indices[0:1]]

                other_indices = np.where(inner_t == 1)[0]
                inner_x, inner_t, inner_y = inner_x[other_indices], \
                                            inner_t[other_indices]*treatment_idx, \
                                            inner_y[other_indices]
            outer_x.append(inner_x)
            outer_t.append(inner_t)
            outer_y.append(inner_y)

        outer_x, outer_t, outer_y = np.concatenate(outer_x, axis=0), \
                                    np.concatenate(outer_t, axis=0), \
                                    np.concatenate(outer_y, axis=0)

        def outer_generator():
            while True:
                indices = np.random.permutation(out_data.shape[0])
                for idx in range(len(indices)):
                    yield outer_x[idx], outer_t[idx], outer_y[idx]

        def inner_generator(wrapped_generator):
            while True:
                batch_data = zip(*map(lambda _: next(wrapped_generator), range(self.batch_size)))
                yield [np.array(batch_data[0]), np.array(batch_data[1])], np.array(batch_data[2])

        new_generator = inner_generator(outer_generator())
        train_steps = max(outer_x.shape[0] // self.batch_size, 1)

        return new_generator, train_steps
