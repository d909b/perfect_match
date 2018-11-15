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
import sqlite3
import numpy as np
from os.path import join
from perfect_match.data_access.batch_augmentation import BatchAugmentation


class DataAccess(BatchAugmentation):
    DB_FILE_NAME = "news.db"
    TABLE_NEWS = "news"

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.db = None
        self.connect()
        self.setup_schema()
        self.cache_rowid = {}
        self.cache_no_rowid = {}

    def get_split_indices(self):
        return None, None

    def connect(self):
        self.db = sqlite3.connect(join(self.data_dir, DataAccess.DB_FILE_NAME),
                                  check_same_thread=False,
                                  detect_types=sqlite3.PARSE_DECLTYPES)

        # Disable journaling.
        self.db.execute("PRAGMA journal_mode = OFF;")
        self.db.execute("PRAGMA page_size = 16384;")

    def setup_schema(self):
        self.setup_news()

        self.db.commit()

    def setup_news(self):
        self.db.execute(("CREATE TABLE IF NOT EXISTS {table_name}"
                         "("
                         "x ARRAY, "
                         "z ARRAY "
                         ");").format(table_name=DataAccess.TABLE_NEWS))

    def insert_many(self, table_name, values):
        self.db.executemany("INSERT INTO {table_name} VALUES ({question_marks});"
                            .format(table_name=table_name,
                                    question_marks=",".join(["?"] * len(values[0]))),
                            values)

    def insert_news(self, values):
        self.insert_many(DataAccess.TABLE_NEWS, values)

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

    def get_row(self, table_name, id, with_rowid=False):
        cache = self.cache_rowid if with_rowid else self.cache_no_rowid
        if id in cache:
            return_value = cache[id]
            return return_value

        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns

        query = "SELECT " \
                "{columns} " \
                "FROM {table_name} " \
                "WHERE rowid = ?;".format(table_name=table_name,
                                          columns=columns)
        return_value = self.db.execute(query, (id,)).fetchone()
        cache[id] = return_value
        return return_value

    def get_rows(self, train_ids):
        tmp_name = "tmp_pairs"
        self.create_temporary_table(tmp_name, map(lambda x: (x,), train_ids))

        news = self.db.execute("SELECT rowid, * "
                               "FROM {table_pairs} "
                               "WHERE rowid IN (SELECT id FROM {tmp_table});"
                               .format(table_pairs=DataAccess.TABLE_NEWS,
                                       tmp_table=tmp_name)).fetchall()

        self.drop_temporary_table(tmp_name)

        ids = np.array(map(lambda x: x[0], news))
        news_data = map(lambda x: x[1], news)
        news_data = np.array(news_data)
        return news_data, ids, news_data

    def get_row_by_id(self, table_name, id, with_rowid=False):
        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns

        query = "SELECT " \
                "{columns} " \
                "FROM {table_name} " \
                "WHERE id = ?;".format(table_name=table_name,
                                       columns=columns)
        return self.db.execute(query, (id,)).fetchone()

    def get_labelled_patients(self):
        return np.arange(self.get_num_rows(DataAccess.TABLE_NEWS)) + 1

    def create_temporary_table(self, table_name, values):
        self.db.execute("CREATE TEMP TABLE {table_name} (id INT);".format(table_name=table_name))
        if len(values) != 0:
            self.db.executemany("INSERT INTO {table_name} VALUES (?);".format(table_name=table_name), values)
        return table_name

    def drop_temporary_table(self, table_name):
        self.db.execute("drop table {tmp_table_name};".format(tmp_table_name=table_name))

    def get_news_dimension(self):
        news = self.db.execute("SELECT * FROM {table_name} WHERE rowid = 1;"
                               .format(table_name=DataAccess.TABLE_NEWS)).fetchone()
        return news[0].shape[0]

    def get_labels(self, args, ids, benchmark):
        assignments = []
        for id in ids:
            news = self.get_row(DataAccess.TABLE_NEWS, id[0])
            assignment = benchmark.get_assignment(id[0], news[0])[0]
            assignments.append(assignment)
        assignments = np.array(assignments)
        num_labels = benchmark.get_num_treatments()
        return assignments, num_labels

    def get_entry_with_id(self, id, args={}):
        news = self.get_row(DataAccess.TABLE_NEWS, id, with_rowid=True)

        patient_id = news[0]
        result = {"id": patient_id, "x": news[1], "z": news[2]}

        return patient_id, result

    def standardise_entry(self, entry):
        return entry

    def prepare_batch(self, args, batch_data, benchmark, is_train=False):
        ids = np.array(map(lambda x: x["id"], batch_data))
        news_data = map(lambda x: x["x"], batch_data)

        assignments = map(benchmark.get_assignment, ids, news_data)
        treatment_data, batch_y = zip(*assignments)
        treatment_data = np.array(treatment_data)

        if args["with_propensity_batch"] and is_train:
            propensity_batch_probability = float(args["propensity_batch_probability"])
            num_randomised_neighbours = int(np.rint(args["num_randomised_neighbours"]))
            news_data, treatment_data, batch_y = self.enhance_batch_with_propensity_matches(benchmark,
                                                                                            treatment_data,
                                                                                            news_data,
                                                                                            batch_y,
                                                                                            propensity_batch_probability,
                                                                                            num_randomised_neighbours)

        input_data = np.asarray(news_data).astype(np.float32)

        batch_y = np.array(batch_y)
        batch_x = [
            input_data,
            treatment_data,
        ]
        return batch_x, batch_y
