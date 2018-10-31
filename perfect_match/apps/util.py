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
import time
import numpy as np
from tensorflow.python.client import device_lib


def random_cycle_generator(num_origins, seed=505):
    while 1:
        random_generator = np.random.RandomState(seed)
        samples = random_generator.permutation(num_origins)
        for sample in samples:
            yield sample


def resample_with_replacement_generator(array):
    while 1:
        for _ in range(len(array)):
            sample_idx = np.random.randint(0, len(array))
            yield array[sample_idx]


def get_num_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def error(*msg):
    log(log_level="ERROR", *msg)


def log(log_level="INFO", *msg):
    print(log_level, ":", *msg, file=sys.stderr)


def report_duration(task, duration):
    log(task, "took", duration, "seconds.")


def time_function(task_name):
    def time_function(func):
        def func_wrapper(*args, **kargs):
            t_start = time.time()
            return_value = func(*args, **kargs)
            t_dur = time.time() - t_start
            report_duration(task_name, t_dur)
            return return_value
        return func_wrapper
    return time_function
