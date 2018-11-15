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

from mahalanobis_batch import MahalanobisBatch
from propensity_batch import PropensityBatch


class BatchAugmentation(object):
    def make_propensity_lists(self, train_ids, benchmark_implementation, **kwargs):
        match_on_covariates = kwargs["match_on_covariates"]
        if match_on_covariates:
            self.batch_augmentation = MahalanobisBatch()
        else:
            self.batch_augmentation = PropensityBatch()
        self.batch_augmentation.make_propensity_lists(train_ids, benchmark_implementation)

    def enhance_batch_with_propensity_matches(self, benchmark, treatment_data, input_data, batch_y,
                                              match_probability=1.0, num_randomised_neighbours=6):
        if self.batch_augmentation is not None:
            return self.batch_augmentation.enhance_batch_with_propensity_matches(benchmark, treatment_data,
                                                                                 input_data, batch_y,
                                                                                 match_probability,
                                                                                 num_randomised_neighbours)
        else:
            raise Exception("Batch augmentation mode must be set.")
