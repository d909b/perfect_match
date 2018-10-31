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
from sklearn.metrics.pairwise import cosine_similarity
from perfect_match.data_access.twins.data_access import DataAccess
from perfect_match.models.benchmarks.tcga_benchmark import TCGABenchmark


class TwinsBenchmark(object):
    def __init__(self, data_dir, is_v2=False, is_binary=True,
                 response_mean_of_mean=0.45, response_std_of_mean=0.15,
                 response_mean_of_std=0.1, response_std_of_std=0.05,
                 strength_of_assignment_bias=10, epsilon_std=0.15,
                 **kwargs):
        self.centroids = None
        self.data_access = DataAccess(data_dir)
        self.assignment_cache = {}
        self.assign_counterfactuals = False
        self.is_v2 = is_v2
        self.num_treatments = 2 if is_binary else 4
        self.response_mean_of_mean = response_mean_of_mean
        self.response_std_of_mean = response_std_of_mean
        self.response_mean_of_std = response_mean_of_std
        self.response_std_of_std = response_std_of_std
        self.strength_of_assignment_bias = strength_of_assignment_bias
        self.epsilon_std = epsilon_std
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
        # 0 = lighter, male, 1 = heavier, male, 2 = lighter, female, 3 = heavier, female
        return self.num_treatments

    def get_data_access(self):
        return self.data_access

    def get_input_shapes(self, args):
        return (self.data_access.get_pairs_dimension(),)

    def get_output_shapes(self, args):
        return (1,)

    def initialise(self, args):
        self.random_generator = np.random.RandomState(909)
        if not self.is_v2:
            self.centroids = self.random_generator.uniform(-0.1, 0.1, size=(self.data_access.get_pairs_dimension(), 1))
        else:
            self.centroids = None

    def fit(self, generator, steps, batch_size):
        if self.is_v2:
            centroids_tmp = []
            centroid_indices = sorted(self.random_generator.permutation(steps*batch_size)[:self.num_treatments + 1])

            current_idx = 0
            while len(centroid_indices) != 0:
                x, _ = next(generator)
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
                    centroids_tmp.append((x[0][next_index], response_mean, response_std))
                current_idx += len(x[0])
            self.centroids = centroids_tmp
            self.assignment_cache = {}

    def get_assignment(self, id, x):
        if self.centroids is None:
            return 0, 0

        if id not in self.assignment_cache:
            assigned_treatment, assigned_y = self._assign(x)
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

        if not self.is_v2:
            # TODO: Assignment and Y are independent = no assignment bias.
            expected_responses = np.dot(self.centroids.T, x[7:]) + self.random_generator.normal(0, 0.1)

            treatment_chosen = self.random_generator.binomial(1, p=TwinsBenchmark.sigmoid(expected_responses))

            lighter_sex, heavier_sex = x[3], x[4]

            if self.num_treatments == 2:
                outcomes = [x[5], x[6]]
            else:
                outcomes = [
                    None, None, None, None
                ]

                if lighter_sex == DataAccess.GENDER_MALE:
                    outcomes[0] = x[5]
                else:
                    outcomes[2] = x[5]

                if heavier_sex == DataAccess.GENDER_MALE:
                    outcomes[1] = x[6]
                else:
                    outcomes[3] = x[6]

                if (treatment_chosen == 0 and lighter_sex == DataAccess.GENDER_FEMALE) or\
                   (treatment_chosen == 1 and heavier_sex == DataAccess.GENDER_FEMALE):
                    treatment_chosen += 2

            return treatment_chosen, outcomes
        else:
            distances = self.get_centroid_weights(x)

            y = []
            control_distance = distances[-1]
            for treatment_idx in range(self.num_treatments):
                this_distance = distances[treatment_idx]
                y.append(50*(this_distance + control_distance) + self.random_generator.normal(0.0, 1.0))
            y = np.array(y)

            # Invert the expected responses, because a lower percentage of recurrence/death is a better outcome.
            choice_percentage = TCGABenchmark.stable_softmax(self.strength_of_assignment_bias * distances[:-1])
            treatment_chosen = self.random_generator.choice(self.num_treatments, p=choice_percentage)

            return treatment_chosen, y

    def get_centroid_weights(self, x):
        similarities = map(
            lambda centroid: cosine_similarity(self.data_access.standardise_entry(
                                                   np.array(x[7:], dtype="float32")
                                               ).reshape((1, -1)),
                                               centroid.reshape((1, -1))),
            map(lambda x: x[0], self.centroids)
        )
        return np.squeeze(similarities)
