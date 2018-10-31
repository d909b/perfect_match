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
from itertools import cycle

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from perfect_match.apps.util import random_cycle_generator, resample_with_replacement_generator, log

LAST_ROW_ID = None


def get_last_row_id():
    return LAST_ROW_ID


def report_distribution(data, labels, num_classes, set_name):
    counts = np.zeros((num_classes,))
    for i in range(num_classes):
        counts[i] = np.sum(labels == i) / float(len(labels))
    log("INFO: Using", set_name, "set (n=", len(data), ") with distribution", counts)


def make_generator(args, benchmark, is_validation=False, is_test=False,
                   validation_fraction=0.2, test_fraction=0.2, seed=909, randomise=True,
                   stratify=True, resample_with_replacement=False):
    fraction_of_data_set = args["fraction_of_data_set"]

    patients = benchmark.get_data_access().get_labelled_patients()
    patients = benchmark.filter(patients)

    num_patients = len(patients)

    if fraction_of_data_set < 1.0:
        num_patients = int(np.rint(num_patients * fraction_of_data_set))
        patients = np.random.permutation(patients)[:num_patients]

    num_validation_patients = int(np.floor(num_patients * validation_fraction))
    num_test_patients = int(np.floor(num_patients * test_fraction))

    split_indices = benchmark.get_data_access().get_split_indices()
    if stratify:
        labels, num_labels = benchmark.get_data_access().get_labels(args, map(lambda x: (x,), patients), benchmark)
        if split_indices[0] is None:
            test_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test_patients, random_state=seed)
            rest_indices, test_indices = next(test_sss.split(patients, labels))
        else:
            rest_indices, test_indices = split_indices

        val_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation_patients, random_state=seed)
        train_indices, val_indices = next(val_sss.split(patients[rest_indices], labels[rest_indices]))

        if is_test:
            report_distribution(patients[rest_indices][train_indices],
                                labels[rest_indices][train_indices],
                                num_labels, "train")
            report_distribution(patients[rest_indices][val_indices],
                                labels[rest_indices][val_indices],
                                num_labels, "validation")
            report_distribution(patients[test_indices],
                                labels[test_indices],
                                num_labels, "test")
    else:
        if split_indices[0] is None:
            indices = np.random.permutation(num_patients)
            rest_indices, test_indices = indices[num_test_patients:], indices[:num_test_patients]
        else:
            rest_indices, test_indices = split_indices

        remaining_indices = np.random.permutation(len(rest_indices))
        train_indices, val_indices = remaining_indices[num_validation_patients:], \
                                     remaining_indices[:num_validation_patients]

    if is_test:
        patients = patients[test_indices]
    elif is_validation:
        patients = patients[rest_indices][val_indices]
    else:
        patients = patients[rest_indices][train_indices]
        if stratify and args["with_propensity_batch"]:
            benchmark.get_data_access().make_propensity_lists(patients, benchmark, **args)

    num_steps = len(patients)

    def generator():
        global LAST_ROW_ID

        if resample_with_replacement:
            id_generator = resample_with_replacement_generator(patients)
        else:
            if randomise:
                id_generator = random_cycle_generator(patients)
            else:
                id_generator = cycle(patients)

        while True:
            next_patient_id = next(id_generator)
            patient_id, result = benchmark.get_data_access().get_entry_with_id(next_patient_id, args)

            LAST_ROW_ID = patient_id
            yield result

    return generator(), num_steps
