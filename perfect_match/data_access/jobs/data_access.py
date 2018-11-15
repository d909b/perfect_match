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

import os
import sys
import sqlite3
import numpy as np
from os.path import join
from perfect_match.data_access.batch_augmentation import BatchAugmentation


class DataAccess(BatchAugmentation):
    DB_FILE_NAME = "jobs.db"
    TABLE_JOBS = "jobs"

    def __init__(self, data_dir, seed, experiment_index):
        self.data_dir = data_dir
        self.x, self.y0, self.y1, self.z, self.e, self.train_indices, self.test_indices =\
            self.generate_new_dataset(seed, experiment_index)
        self.db = None
        self.connect()
        self.setup_schema()
        self.initialise_data()

    def get_split_indices(self):
        return None, None

    def generate_new_dataset(self, seed, experiment_index):
        this_directory = os.path.dirname(os.path.realpath(__file__))
        train_file_path = join(this_directory, "jobs_DW_bin.train.npz")
        test_file_path = join(this_directory, "jobs_DW_bin.test.npz")

        print("INFO: Using Jobs set", experiment_index, ".", file=sys.stderr)
        train_set = np.load(train_file_path)
        test_set = np.load(test_file_path)

        def get_field(name, squeeze=False):
            return np.concatenate([train_set[name],
                                   test_set[name]], axis=0)

        n_train, n_test = train_set["x"].shape[0], test_set["x"].shape[0]
        train_indices = np.arange(0, n_train)
        test_indices = np.arange(n_train, n_train+n_test)
        y_f = get_field("yf")
        t = np.squeeze(get_field("t").astype(int))
        y = np.zeros((n_train+n_test, 2))
        y[t == 0, 0:1] = y_f[t == 0]
        y[t == 1, 1:2] = y_f[t == 1]

        e = np.squeeze(get_field("e"), axis=-1).astype(int)
        x = np.squeeze(get_field("x"), axis=-1)
        x = np.column_stack([np.expand_dims(np.arange(n_train+n_test)+1, axis=-1), x])

        return x, y[:, 0], y[:, 1], \
               t, e, train_indices, test_indices

    def connect(self):
        db_file = join(self.data_dir, DataAccess.DB_FILE_NAME)
        if os.path.isfile(db_file):
            os.remove(db_file)

        self.db = sqlite3.connect(db_file,
                                  check_same_thread=False,
                                  detect_types=sqlite3.PARSE_DECLTYPES)

        # Disable journaling.
        self.db.execute("PRAGMA journal_mode = OFF;")
        self.db.execute("PRAGMA page_size = 16384;")

    def initialise_data(self):
        with self.db:
            self.insert_many(DataAccess.TABLE_JOBS, zip(self.x[:, 0], self.x[:, 1:],
                                                        self.y0, self.y1, self.z,
                                                        self.e))

    def setup_schema(self):
        self.setup_jobs()
        self.db.commit()

    def setup_jobs(self):
        self.db.execute(("CREATE TABLE IF NOT EXISTS {table_name}"
                         "("
                         "id INT NOT NULL PRIMARY KEY, "
                         "x ARRAY, "
                         "y0 FLOAT, "
                         "y1 FLOAT, "
                         "t INT, "
                         "e INT "
                         ");").format(table_name=DataAccess.TABLE_JOBS))

    def insert_many(self, table_name, values):
        self.db.executemany("INSERT INTO {table_name} VALUES ({question_marks});"
                            .format(table_name=table_name,
                                    question_marks=",".join(["?"] * len(values[0]))),
                            values)

    def insert_ihdp(self, values):
        self.insert_many(DataAccess.TABLE_JOBS, values)

    def get_column(self, table_name, ids, column_name):
        tmp_name = "tmp_ids"
        self.create_temporary_table(tmp_name, ids)
        return_value = self.db.execute("SELECT {column_name} "
                                       "FROM {table_name} "
                                       "WHERE rowid IN (SELECT id FROM {tmp_table}) "
                                       "ORDER BY rowid;"
                                       .format(column_name=column_name,
                                               table_name=table_name,
                                               tmp_table=tmp_name)).fetchall()
        self.drop_temporary_table(tmp_name)
        return return_value

    def get_num_rows(self, table_name):
        # NOTE: This query assumes that there has never been any deletions in the time series table.
        return self.db.execute("SELECT MAX(_ROWID_) FROM {} LIMIT 1;".format(table_name)) \
            .fetchone()[0]

    def get_row(self, table_name, id, columns="x", with_rowid=False):
        if with_rowid:
            columns = "rowid, " + columns

        query = "SELECT " \
                "{columns} " \
                "FROM {table_name} " \
                "WHERE rowid = ?;".format(table_name=table_name,
                                          columns=columns)
        return self.db.execute(query, (id,)).fetchone()

    def get_rows(self, train_ids, columns="rowid, x"):
        tmp_name = "tmp_pairs"
        self.create_temporary_table(tmp_name, map(lambda x: (x,), train_ids))

        ihdp = self.db.execute("SELECT {columns} "
                               "FROM {table_pairs} "
                               "WHERE rowid IN (SELECT id FROM {tmp_table});"
                               .format(columns=columns,
                                       table_pairs=DataAccess.TABLE_JOBS,
                                       tmp_table=tmp_name)).fetchall()

        self.drop_temporary_table(tmp_name)

        if columns == "rowid, x":
            ids = np.array(map(lambda x: x[0], ihdp))
            ihdp_data = map(lambda x: x[1], ihdp)
            ihdp_data = np.array(ihdp_data)
            return ihdp_data, ids, ihdp_data
        else:
            return ihdp

    def get_labelled_patients(self):
        return np.arange(1, self.get_num_rows(DataAccess.TABLE_JOBS) + 1)

    def create_temporary_table(self, table_name, values):
        self.db.execute("CREATE TEMP TABLE {table_name} (id INT);".format(table_name=table_name))
        if len(values) != 0:
            self.db.executemany("INSERT INTO {table_name} VALUES (?);".format(table_name=table_name), values)
        return table_name

    def drop_temporary_table(self, table_name):
        self.db.execute("drop table {tmp_table_name};".format(tmp_table_name=table_name))

    def get_labels(self, args, ids, benchmark):
        assignments = []
        for id in ids:
            ihdp = self.get_row(DataAccess.TABLE_JOBS, id[0], with_rowid=True)
            assignment = benchmark.get_assignment(id[0], ihdp[1])[0]
            assignments.append(assignment)
        assignments = np.array(assignments)
        num_labels = benchmark.get_num_treatments()
        return assignments, num_labels

    def get_entry_with_id(self, id, args):
        ihdp = self.get_row(DataAccess.TABLE_JOBS, id, with_rowid=True)

        patient_id = ihdp[0]
        result = {"id": patient_id, "x": ihdp[1]}

        return patient_id, result

    def standardise_entry(self, entry):
        return entry

    def prepare_batch(self, args, batch_data, benchmark, is_train=False):
        ids = np.array(map(lambda x: x["id"], batch_data))
        ihdp_data = map(lambda x: x["x"], batch_data)

        assignments = map(benchmark.get_assignment, ids, ihdp_data)
        treatment_data, batch_y = zip(*assignments)
        treatment_data = np.array(treatment_data)

        if args["with_propensity_batch"] and is_train:
            propensity_batch_probability = float(args["propensity_batch_probability"])
            num_randomised_neighbours = int(np.rint(args["num_randomised_neighbours"]))
            ihdp_data, treatment_data, batch_y = self.enhance_batch_with_propensity_matches(benchmark,
                                                                                            treatment_data,
                                                                                            ihdp_data,
                                                                                            batch_y,
                                                                                            propensity_batch_probability,
                                                                                            num_randomised_neighbours)

        input_data = np.asarray(ihdp_data).astype(np.float32)

        batch_y = np.array(batch_y)
        batch_x = [
            input_data,
            treatment_data,
        ]
        return batch_x, batch_y
