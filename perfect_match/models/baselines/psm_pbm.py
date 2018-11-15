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

import numpy as np
from perfect_match.models.baselines.neural_network import NeuralNetwork
from perfect_match.data_access.batch_augmentation import BatchAugmentation
from perfect_match.data_access.generator import get_last_id_set


class PSM_PBM(NeuralNetwork):
    def __init__(self):
        super(PSM_PBM, self).__init__()

    def _build(self, **kwargs):
        self.args = kwargs
        self.num_treatments = kwargs["num_treatments"]
        self.batch_size = kwargs["batch_size"]
        self.benchmark = kwargs["benchmark"]
        self.batch_augmentation = BatchAugmentation()
        self.propensity_batch_probability = float(kwargs["propensity_batch_probability"])
        self.num_randomised_neighbours = int(np.rint(kwargs["num_randomised_neighbours"]))

        return super(PSM_PBM, self)._build(**kwargs)

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size):
        train_generator, train_steps = self.get_matched_generator(train_generator, train_steps)
        super(PSM_PBM, self).fit_generator(train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size)

    def get_matched_generator(self, train_generator, train_steps):
        all_x, all_y, all_ids = [], [], []
        for _ in range(train_steps):
            x, y = next(train_generator)
            all_x.append(x)
            all_y.append(y)
            all_ids.append(get_last_id_set())

        x, t, y, ids = np.concatenate(map(lambda x: x[0], all_x), axis=0), \
                       np.concatenate(map(lambda x: x[1], all_x), axis=0), \
                       np.concatenate(all_y, axis=0), \
                       np.concatenate(all_ids, axis=0)

        t_indices = map(lambda t_idx: np.where(t == t_idx)[0], range(self.num_treatments))
        t_lens = map(lambda x: len(x), t_indices)

        self.batch_augmentation.make_propensity_lists(ids, self.benchmark, **self.args)

        undersample = True
        base_treatment_idx = np.argmin(t_lens) if undersample else np.argmax(t_lens)
        base_indices = t_indices[base_treatment_idx]
        inner_x, inner_t, inner_y = x[base_indices], t[base_indices], y[base_indices]

        outer_x, outer_t, outer_y = \
                self.batch_augmentation.enhance_batch_with_propensity_matches(self.benchmark,
                                                                              inner_t,
                                                                              inner_x,
                                                                              inner_y,
                                                                              self.propensity_batch_probability,
                                                                              self.num_randomised_neighbours)

        def outer_generator():
            while True:
                indices = np.random.permutation(outer_x.shape[0])
                for idx in range(len(indices)):
                    yield outer_x[idx], outer_t[idx], outer_y[idx]

        def inner_generator(wrapped_generator):
            while True:
                batch_data = zip(*map(lambda _: next(wrapped_generator), range(self.batch_size)))
                yield [np.array(batch_data[0]), np.array(batch_data[1])], np.array(batch_data[2])

        new_generator = inner_generator(outer_generator())
        train_steps = max(outer_x.shape[0] // self.batch_size, 1)

        return new_generator, train_steps
