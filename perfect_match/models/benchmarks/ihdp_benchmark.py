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
import math
import numpy as np
from perfect_match.data_access.ihdp.data_access import DataAccess


class IHDPBenchmark(object):
    def __init__(self, data_dir,
                 **kwargs):
        data_dir = kwargs["output_directory"]
        self.data_access = DataAccess(data_dir, kwargs["seed"], kwargs["experiment_index"])
        self.assignment_cache = {}
        self.assign_counterfactuals = False
        self.num_treatments = 2
        self.seed = kwargs["seed"]
        self.random_generator = None

    @staticmethod
    def get_db_file_name():
        return DataAccess.DB_FILE_NAME

    def filter(self, patients):
        return patients

    def set_assign_counterfactuals(self, value):
        self.assign_counterfactuals = value

    def get_num_treatments(self):
        return self.num_treatments

    def get_data_access(self):
        return self.data_access

    def get_input_shapes(self, args):
        return (25,)

    def get_output_shapes(self, args):
        return (1,)

    def initialise(self, args):
        data_dir = args["output_directory"]
        self.data_access = DataAccess(data_dir, args["seed"], args["experiment_index"])
        self.random_generator = np.random.RandomState(self.seed)

    def fit(self, generator, steps, batch_size):
        pass

    def get_assignment(self, id, x):
        if id not in self.assignment_cache:
            assigned_treatment, assigned_y = self._assign(id)
            self.assignment_cache[id] = assigned_treatment, assigned_y

        assigned_treatment, assigned_y = self.assignment_cache[id]

        if self.assign_counterfactuals:
            return assigned_treatment, assigned_y
        else:
            return assigned_treatment, assigned_y[assigned_treatment]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def _assign(self, id):
        treatment_chosen = self.data_access.get_row(DataAccess.TABLE_IHDP, id, columns="t")[0]
        y = np.array(self.data_access.get_row(DataAccess.TABLE_IHDP, id, columns="y0,y1"))
        return treatment_chosen, y
