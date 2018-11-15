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
import os
import io
import sqlite3
import datetime
import numpy as np
from os.path import join
from pandas import read_csv
from perfect_match.data_access.batch_augmentation import BatchAugmentation


def adapt_array(arr):
    """
    SOURCE: http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)
sqlite3.register_converter("DATE", lambda x: datetime.datetime.fromtimestamp(float(x)/1000))


class DataAccess(BatchAugmentation):
    DB_FILE_NAME = "tcga.db"
    MIN_FILE_NAME = "min_val.npy"
    MAX_FILE_NAME = "max_val.npy"

    TABLE_CLINICAL = "clinical"
    TABLE_RNASEQ = "rnaseq"
    TABLE_METHYLATION = "methylation"
    TABLE_SNP = "snp"
    TABLE_SURGERIES = "surgeries"

    GENDER_MALE = 0
    GENDER_FEMALE = 1

    RADIATION_THERAPY_NO = 0
    RADIATION_THERAPY_YES = 1

    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.tcga_num_features = int(np.rint(kwargs["tcga_num_features"]))
        self.db = None
        this_directory = os.path.dirname(os.path.realpath(__file__))
        min_path = os.path.join(this_directory, DataAccess.MIN_FILE_NAME)
        max_path = os.path.join(this_directory, DataAccess.MAX_FILE_NAME)
        if os.path.exists(min_path) and \
           os.path.exists(max_path):
            self.min_val, self.max_val = np.load(min_path)[:-1], np.load(max_path)[:-1]
        else:
            self.min_val, self.max_val = None, None
        self.connect()
        self.setup_schema()

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
        self.setup_clinical()
        self.setup_rnaseq()
        self.setup_methylation()
        self.setup_snp()

        self.db.commit()

    def setup_clinical(self):
        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name}"
                        "("
                        "id TEXT NOT NULL PRIMARY KEY, "
                        "age INT, "
                        "gender INT, "
                        "icd10_diagnosis TEXT, "
                        "dataset_name TEXT, "
                        "days_to_death INT, "
                        "days_to_recurrence INT, "
                        "days_to_surgery INT, "
                        "did_radiation_therapy INT"
                        ");"
                        .format(table_name=DataAccess.TABLE_CLINICAL))

    def setup_rnaseq(self):
        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name}"
                        "("
                        "id TEXT NOT NULL PRIMARY KEY, "
                        "data ARRAY, "
                        "clinical_id TEXT NOT NULL, "
                        "FOREIGN KEY(clinical_id) REFERENCES {clinical_table_name}(id)"
                        ");"
                        .format(table_name=DataAccess.TABLE_RNASEQ,
                                clinical_table_name=DataAccess.TABLE_CLINICAL))

    def setup_methylation(self):
        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name}"
                        "("
                        "id TEXT NOT NULL PRIMARY KEY, "
                        "data ARRAY, "
                        "clinical_id TEXT NOT NULL, "
                        "FOREIGN KEY(clinical_id) REFERENCES {clinical_table_name}(id)"
                        ");"
                        .format(table_name=DataAccess.TABLE_METHYLATION,
                                clinical_table_name=DataAccess.TABLE_CLINICAL))

    def setup_snp(self):
        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name}"
                        "("
                        "id TEXT NOT NULL PRIMARY KEY, "
                        "data ARRAY, "
                        "clinical_id TEXT NOT NULL, "
                        "FOREIGN KEY(clinical_id) REFERENCES {clinical_table_name}(id)"
                        ");"
                        .format(table_name=DataAccess.TABLE_SNP,
                                clinical_table_name=DataAccess.TABLE_CLINICAL))

    def insert_many(self, table_name, values):
        self.db.executemany("INSERT INTO {table_name} VALUES ({question_marks});"
                            .format(table_name=table_name,
                                    question_marks=",".join(["?"] * len(values[0]))),
                            values)

    def insert_clinical(self, values):
        self.insert_many(DataAccess.TABLE_CLINICAL, values)

    def insert_rnaseq(self, values):
        self.insert_many(DataAccess.TABLE_RNASEQ, values)

    def insert_methylation(self, values):
        self.insert_many(DataAccess.TABLE_METHYLATION, values)

    def insert_snp(self, values):
        self.insert_many(DataAccess.TABLE_SNP, values)

    def get_dataset_names(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "dataset_name")

    def get_days_to_death(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "days_to_death")

    def get_days_to_recurrence(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "days_to_recurrence")

    def get_days_to_surgery(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "days_to_surgery")

    def get_did_radiation_therapy(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "did_radiation_therapy")

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
        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns

        query = "SELECT " \
                "{columns} " \
                "FROM {table_name} " \
                "WHERE rowid = ?;".format(table_name=table_name,
                                          columns=columns)

        if not isinstance(id, tuple):
            id = (id,)

        return self.db.execute(query, id).fetchone()

    def get_rows(self, train_ids):
        tmp_name = "tmp_data"
        self.create_temporary_table(tmp_name, map(lambda x: (x,), train_ids))

        patients = self.db.execute("SELECT rowid, * "
                                   "FROM {table_pairs} "
                                   "WHERE rowid IN (SELECT id FROM {tmp_table});"
                                   .format(table_pairs=DataAccess.TABLE_CLINICAL,
                                           tmp_table=tmp_name)).fetchall()

        self.drop_temporary_table(tmp_name)

        patient_rowids = map(lambda x: x[0], patients)
        patient_ids = map(lambda x: x[1], patients)

        tmp_name = "patient_ids"
        self.create_temporary_table(tmp_name, map(lambda x: (x,), patient_ids))

        rnaseq_data = self.db.execute("SELECT * " \
                                      "FROM {table_name} " \
                                      "WHERE clinical_id IN (SELECT id FROM {tmp_table});"
                                      .format(table_name=DataAccess.TABLE_RNASEQ,
                                              tmp_table=tmp_name)).fetchall()

        self.drop_temporary_table(tmp_name)

        id_seq_map = {}
        for sample in rnaseq_data:
            clinical_id = sample[-1]
            id_seq_map[clinical_id] = sample

        rnaseq_data = map(lambda id: id_seq_map[id], patient_ids)
        rnaseq_data = np.array(map(lambda x: x[1], rnaseq_data))
        rnaseq_data = (rnaseq_data - self.min_val) / (self.max_val - self.min_val + 0.00001)

        return rnaseq_data, patient_rowids, rnaseq_data

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

    def get_rows_by_clinical_id(self, table_name, id, with_rowid=False):
        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns

        query = "SELECT " \
                "{columns} " \
                "FROM {table_name} " \
                "WHERE clinical_id = ?;".format(table_name=table_name,
                                                columns=columns)
        return self.db.execute(query, (id,)).fetchone()

    def get_labelled_patients(self):
        return_value = self.db.execute("SELECT {table_name}.rowid "
                                       "FROM {table_name} "
                                       "WHERE {table_name}.id IN (SELECT clinical_id FROM {other_table}) "
                                       "ORDER BY {table_name}.rowid;"
                                       .format(table_name=DataAccess.TABLE_CLINICAL,
                                               other_table=DataAccess.TABLE_RNASEQ)).fetchall()

        return np.squeeze(return_value)

    def create_temporary_table(self, table_name, values):
        self.db.execute("CREATE TEMP TABLE {table_name} (id INT);".format(table_name=table_name))
        if len(values) != 0:
            self.db.executemany("INSERT INTO {table_name} VALUES (?);".format(table_name=table_name), values)
        return table_name

    def drop_temporary_table(self, table_name):
        self.db.execute("drop table {tmp_table_name};".format(tmp_table_name=table_name))

    def get_rnaseq_dimension(self):
        rnaseq = self.db.execute("SELECT data FROM {table_name} WHERE rowid = 1;"
                                 .format(table_name=DataAccess.TABLE_RNASEQ)).fetchone()[0]

        return rnaseq.shape[0]

    @staticmethod
    def get_clinical_data(filepath):
        clinical_data = read_csv(filepath, sep="\t", header=0, index_col=0).T
        return clinical_data

    @staticmethod
    def get_rnaseq_data(filepath):
        rnaseq_data = read_csv(filepath, sep="\t", header=0, index_col=0,
                               skiprows=lambda i: i == 1).T
        return rnaseq_data

    @staticmethod
    def binarize_days(days):
        days_copy = np.copy(days)
        days_copy[days_copy > 0] = 1
        days_copy[days_copy <= 0] = 0
        return days_copy

    def get_labels(self, args, patients, benchmark):
        if benchmark is not None:
            tcga_num_features = int(np.rint(args["tcga_num_features"]))

            assignments = []
            for id in patients:
                entry = self.get_entry_with_id(id[0], {"with_rnaseq": True})[1]
                rnaseq_data = np.array(entry["rnaseq"][1])
                rnaseq_data = (rnaseq_data - self.min_val) / (self.max_val - self.min_val + 0.00001)

                if tcga_num_features > 0:
                    rnaseq_data = rnaseq_data[:tcga_num_features]

                assignment = benchmark.get_assignment(id, rnaseq_data)[0]
                assignments.append(assignment)
            assignments = np.array(assignments)

            num_labels = benchmark.get_num_treatments()
            return assignments, num_labels
        else:
            names = self.get_dataset_names(patients)
            unique_names = set(names)
            num_names = len(unique_names)
            name_mapping = dict(zip(unique_names, range(len(unique_names))))

            deaths = DataAccess.binarize_days(self.get_days_to_death(patients))
            recurrences = DataAccess.binarize_days(self.get_days_to_recurrence(patients))
            surgeries = DataAccess.binarize_days(self.get_days_to_surgery(patients))
            radiations = DataAccess.binarize_days(self.get_did_radiation_therapy(patients))

            # Binarize treatment combinations into classes.
            labels = deaths * 2 ** 0 + recurrences * 2 ** 1 + surgeries * 2 ** 2 + radiations * 2 ** 3
            num_labels = 2 ** 4 - 1
            return labels, num_labels

    def get_entry_with_id(self, id, args={"with_rnaseq": True}):
        with_rnaseq = args["with_rnaseq"]

        patient = self.get_row(DataAccess.TABLE_CLINICAL, id, with_rowid=True)

        patient_rowid = patient[0]
        patient_id = patient[1]

        result = {"clinical": patient}

        if with_rnaseq:
            result["rnaseq"] = self.get_rows_by_clinical_id(DataAccess.TABLE_RNASEQ, patient_id)

        return patient_rowid, result

    def prepare_batch(self, args, batch_data, benchmark, is_train=False):
        tcga_num_features = self.tcga_num_features

        CLINICAL_INDEX_RADIATION = -1
        CLINICAL_INDEX_SURGERY = -2
        INDEX_DEATH, INDEX_RECURRENCE = 6, 7

        patient_ids = np.array(map(lambda x: x["clinical"][0], batch_data))
        rnaseq_data = np.array(map(lambda x: x["rnaseq"][1], batch_data))
        rnaseq_data = (rnaseq_data - self.min_val) / (self.max_val - self.min_val + 0.00001)

        if benchmark is None:
            treatment_data = np.array(map(lambda x: x["clinical"][CLINICAL_INDEX_RADIATION], batch_data)) * 2 ** 0 + \
                             np.array(map(lambda x: x["clinical"][CLINICAL_INDEX_SURGERY] != -1, batch_data)) * 2 ** 1
            batch_y = map(lambda x: x["clinical"][INDEX_DEATH] != -1, batch_data)
        else:
            assignments = map(benchmark.get_assignment, patient_ids, rnaseq_data)
            treatment_data, batch_y = zip(*assignments)
            treatment_data = np.array(treatment_data)

        if tcga_num_features > 0:
            rnaseq_data = benchmark.select_features(rnaseq_data)

        if args["with_propensity_batch"] and is_train:
            propensity_batch_probability = float(args["propensity_batch_probability"])
            num_randomised_neighbours = int(np.rint(args["num_randomised_neighbours"]))
            rnaseq_data, treatment_data, batch_y = self.enhance_batch_with_propensity_matches(benchmark,
                                                                                              treatment_data,
                                                                                              rnaseq_data,
                                                                                              batch_y,
                                                                                              propensity_batch_probability,
                                                                                              num_randomised_neighbours)

        batch_y = np.array(batch_y)
        batch_x = [
            rnaseq_data,
            treatment_data,
        ]
        return batch_x, batch_y
