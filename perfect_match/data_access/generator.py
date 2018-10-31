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
from keras.utils import to_categorical
from perfect_match.data_access.patient_generator import get_last_row_id


LAST_ID_SET = None


def get_last_id_set():
    return LAST_ID_SET


def make_keras_generator(args, wrapped_generator, num_steps,
                         batch_size=1, num_losses=1,
                         benchmark=None, is_train=False):
    method = args["method"]
    with_propensity_dropout = args["with_propensity_dropout"]
    num_steps = num_steps // batch_size

    def generator():
        global LAST_ID_SET
        while True:
            batch_data, ids = zip(*map(lambda _: (next(wrapped_generator), get_last_row_id()),
                                       range(batch_size)))

            LAST_ID_SET = ids

            batch_x, batch_y = benchmark.get_data_access().prepare_batch(args, batch_data, benchmark, is_train)

            if num_losses != 1:
                batch_y = batch_y * num_losses

            if with_propensity_dropout and (method == "nn" or method == "nn+"):
                batch_y = [to_categorical(batch_x[1], num_classes=benchmark.get_num_treatments()), batch_y]

            yield batch_x, batch_y

    return generator(), num_steps


def wrap_generator_with_constant_y(wrapped_generator, y):

    def generator():
        while True:
            x, _ = next(wrapped_generator)
            yield x, np.array([y] * len(x[0]))
    return generator()
