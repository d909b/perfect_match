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
from perfect_match.apps.parameters import clip_percentage
from sklearn.metrics.pairwise import euclidean_distances
from perfect_match.data_access.news.data_access import DataAccess
from perfect_match.models.benchmarks.tcga_benchmark import TCGABenchmark


class NewsBenchmark(object):
    def __init__(self, data_dir,
                 response_mean_of_mean=0.45, response_std_of_mean=0.15,
                 response_mean_of_std=0.1, response_std_of_std=0.05,
                 strength_of_assignment_bias=10, epsilon_std=0.15,
                 num_samples=5000, num_treatments=16,
                 **kwargs):
        self.centroids = None
        self.data_access = DataAccess(data_dir)
        self.assignment_cache = {}
        self.assign_counterfactuals = False
        self.num_treatments = num_treatments
        self.response_mean_of_mean = response_mean_of_mean
        self.response_std_of_mean = response_std_of_mean
        self.response_mean_of_std = response_mean_of_std
        self.response_std_of_std = response_std_of_std
        self.strength_of_assignment_bias = strength_of_assignment_bias
        self.epsilon_std = epsilon_std
        self.seed = kwargs["seed"]
        self.random_generator = None
        self.num_samples = num_samples

    @staticmethod
    def get_db_file_name():
        return DataAccess.DB_FILE_NAME

    def filter(self, patients):
        return self.random_generator.choice(patients, size=self.num_samples, replace=False)

    def set_assign_counterfactuals(self, value):
        self.assign_counterfactuals = value

    def get_num_treatments(self):
        return self.num_treatments

    def get_data_access(self):
        return self.data_access

    def get_input_shapes(self, args):
        return (self.data_access.get_news_dimension(),)

    def get_output_shapes(self, args):
        return (1,)

    def initialise(self, args, seed=909):
        self.random_generator = np.random.RandomState(seed)
        self.centroids = None

    def fit(self, generator, steps, batch_size):
        from perfect_match.data_access.generator import get_last_id_set

        centroids_tmp = []
        centroid_indices = sorted(self.random_generator.permutation(steps*batch_size)[:self.num_treatments + 1])

        current_idx = 0
        while len(centroid_indices) != 0:
            x, _ = next(generator)
            ids = get_last_id_set()

            while len(centroid_indices) != 0 and centroid_indices[0] <= current_idx + len(x[0]):
                next_index = centroid_indices[0]
                del centroid_indices[0]

                is_last_treatment = len(centroid_indices) == 0
                if is_last_treatment:
                    # Last treatment is control = worse expected outcomes.
                    response_mean_of_mean = 1 - self.response_mean_of_mean
                else:
                    response_mean_of_mean = self.response_mean_of_mean

                response_mean = clip_percentage(self.random_generator.normal(response_mean_of_mean,
                                                                             self.response_std_of_mean))
                response_std = clip_percentage(self.random_generator.normal(self.response_mean_of_std,
                                                                            self.response_std_of_std))

                id, data = self.data_access.get_entry_with_id(ids[next_index])
                z = data["z"]

                centroids_tmp.append((z, response_mean, response_std))
            current_idx += len(x[0])
        self.centroids = centroids_tmp
        self.assignment_cache = {}

    def get_assignment(self, id, x):
        if self.centroids is None:
            return 0, 0

        if id not in self.assignment_cache:
            id, data = self.data_access.get_entry_with_id(id)
            z = data["z"]
            assigned_treatment, assigned_y = self._assign(z)
            self.assignment_cache[id] = assigned_treatment, assigned_y

        assigned_treatment, assigned_y = self.assignment_cache[id]

        if self.assign_counterfactuals:
            return assigned_treatment, assigned_y
        else:
            return assigned_treatment, assigned_y[assigned_treatment]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def _assign(self, x):
        # Assignment should be biased towards treatments that help more.
        assert self.centroids is not None, "Must call __fit__ before __assign__."

        distances = self.get_centroid_weights(x)

        expected_responses = []
        for treatment in range(self.num_treatments + 1):
            _, response_mean, response_std = self.centroids[treatment]
            y_this_treatment = self.random_generator.normal(response_mean, response_std)
            expected_responses.append(
                clip_percentage(y_this_treatment + self.random_generator.normal(0.0, self.epsilon_std))
            )
        expected_responses = np.array(expected_responses)

        y = []
        control_response, control_distance = expected_responses[-1], distances[-1]
        for treatment_idx in range(self.num_treatments):
            this_response, this_distance = expected_responses[treatment_idx], distances[treatment_idx]
            y.append(this_response * (this_distance + control_distance))
        y = np.array(y)

        # Invert the expected responses, because a lower percentage of recurrence/death is a better outcome.
        treatment_chosen = self.random_generator.choice(self.num_treatments,
                                                        p=TCGABenchmark.stable_softmax(
                                                            self.strength_of_assignment_bias * y)
                                                        )

        return treatment_chosen, 50*y

    def get_centroid_weights(self, x):
        similarities = map(
            lambda centroid: euclidean_distances(self.data_access.standardise_entry(x).reshape((1, -1)),
                                                 centroid.reshape((1, -1))),
            map(lambda x: x[0], self.centroids)
        )
        return np.squeeze(similarities)
