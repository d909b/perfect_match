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

import sqlite3
import numpy as np
from os.path import join
from pandas import read_csv
from perfect_match.data_access.batch_augmentation import BatchAugmentation


class DataAccess(BatchAugmentation):
    DB_FILE_NAME = "twins.db"

    TABLE_PAIRS = "pairs"

    GENDER_MALE = 0
    GENDER_FEMALE = 1

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.db = None
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
        self.setup_pairs()

        self.db.commit()

    def setup_pairs(self):
        columns = ""
        for idx, fields in enumerate(DataAccess.get_ordered_fields()):
            name = fields[1]
            if idx < 3:
                columns += name + "_0 INT,"
                columns += name + "_1 INT,"
            else:
                columns += name + " INT,"
        columns = columns[:-1]
        self.db.execute(("CREATE TABLE IF NOT EXISTS {table_name}"
                         "("
                         "id INT NOT NULL PRIMARY KEY, " + columns +
                         ");").format(table_name=DataAccess.TABLE_PAIRS))

    def insert_many(self, table_name, values):
        self.db.executemany("INSERT INTO {table_name} VALUES ({question_marks});"
                            .format(table_name=table_name,
                                    question_marks=",".join(["?"] * len(values[0]))),
                            values)

    def insert_clinical(self, values):
        self.insert_many(DataAccess.TABLE_PAIRS, values)

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
        return self.db.execute(query, (id,)).fetchone()

    def get_rows(self, train_ids):
        tmp_name = "tmp_pairs"
        self.create_temporary_table(tmp_name, map(lambda x: (x,), train_ids))

        pairs = self.db.execute("SELECT * "
                                "FROM {table_pairs} "
                                "WHERE rowid IN (SELECT id FROM {tmp_table});"
                                .format(table_pairs=DataAccess.TABLE_PAIRS,
                                        tmp_table=tmp_name)).fetchall()

        self.drop_temporary_table(tmp_name)

        input_data = np.array(pairs)
        ids, pair_data = input_data[:, 0], input_data[:, 7:]
        return input_data, ids, pair_data

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

    def get_rows_by_clinical_id(self, table_name, id):
        columns = "*"

        query = "SELECT " \
                "{columns} " \
                "FROM {table_name} " \
                "WHERE clinical_id = ?;".format(table_name=table_name,
                                                columns=columns)
        return self.db.execute(query, (id,)).fetchone()

    def get_labelled_patients(self):
        return np.arange(self.get_num_rows(DataAccess.TABLE_PAIRS)) + 1

    def create_temporary_table(self, table_name, values):
        self.db.execute("CREATE TEMP TABLE {table_name} (id INT);".format(table_name=table_name))
        if len(values) != 0:
            self.db.executemany("INSERT INTO {table_name} VALUES (?);".format(table_name=table_name), values)
        return table_name

    def drop_temporary_table(self, table_name):
        self.db.execute("drop table {tmp_table_name};".format(tmp_table_name=table_name))

    def get_pairs_dimension(self):
        pair = self.db.execute("SELECT * FROM {table_name} WHERE rowid = 1;"
                               .format(table_name=DataAccess.TABLE_PAIRS)).fetchone()
        return len(pair) - 7

    @staticmethod
    def get_ordered_fields(clean=False):
        if clean:
            factor_list = [8, 9]
            adequacy_missing = 4
        else:
            factor_list = 9
            adequacy_missing = None
        identity = lambda x: x
        minus_one = lambda x: x-1
        convert_gender = lambda x: DataAccess.GENDER_MALE if x == 1 else DataAccess.GENDER_FEMALE
        factor_fun = lambda x: 1 if x == 1 else 0
        divide_by = lambda val: lambda x: float(x) / float(val)

        return [
            ("DBIRWT", "birth_weight", None, identity, identity),
            ("CSEX", "child_sex", None, convert_gender, identity),
            ("AGED", "days_age_at_death", None, lambda x: 0 if np.isnan(x) else 1, identity),
            ("DTOTORD", "number_previous_births", 99, identity, divide_by(10)),
            ("DMAR", "marital_status", None, minus_one, identity),
            ("DMAGE", "mother_age", None, minus_one, divide_by(50)),
            ("DMEDUC", "mother_education", 99, minus_one, divide_by(17)),
            #("MRACE", "mother_race", None, minus_one),
            #("FRACE", "father_race", 99, minus_one),
            ("PLDEL", "place_of_delivery", 9, minus_one, identity),
            ("RESSTATB", "residence_state", None, minus_one, identity),
            # ("BRSTATE_REG", "residence_state_region", None),
            # ("DLIVORD", "number_previous_births_live", 99),
            ("GESTAT", "number_gestation_weeks", 99, identity, divide_by(47)),
            #("GESTAT10", "number_gestation_weeks_coded", 10),
            ("ADEQUACY", "adequacy_of_care", adequacy_missing, minus_one, identity),
            ("MPCB", "month_of_pregnancy_care_began", 99, minus_one, divide_by(9)),
            ("NPREVIST", "number_prenatal_visits", 99, identity, divide_by(49)),
            # ("DISLLB", "interval_since_last_live_birth", 999),
            ("ANEMIA", "anemia", factor_list, factor_fun, identity),
            ("CARDIAC", "cardiac", factor_list, factor_fun, identity),
            ("LUNG", "lung", factor_list, factor_fun, identity),
            ("DIABETES", "diabetes", factor_list, factor_fun, identity),
            ("HERPES", "herpes", factor_list, factor_fun, identity),
            ("HYDRA", "hydra", factor_list, factor_fun, identity),
            ("HEMO", "Hemoqlobinopathy", factor_list, factor_fun, identity),
            ("CHYPER", "hypertension_chronic", factor_list, factor_fun, identity),
            ("PHYPER", "hypertension_pregnancy", factor_list, factor_fun, identity),
            ("ECLAMP", "eclampsia", factor_list, factor_fun, identity),
            ("INCERVIX", "incompetent_cervix", factor_list, factor_fun, identity),
            ("PRE4000", "previous_infant_less_than_4000", factor_list, factor_fun, identity),
            ("PRETERM", "previous_preterm", factor_list, factor_fun, identity),
            ("RENAL", "renal", factor_list, factor_fun, identity),
            ("RH", "rh_sensitisation", factor_list, factor_fun, identity),
            ("UTERINE", "uterine_bleeding", factor_list, factor_fun, identity),
            ("OTHERMR", "other", factor_list, factor_fun, identity),
            # ("TOBACCO", "tobacco_use", 9),
            ("CIGAR", "num_cigarettes_per_day", 99, identity, divide_by(98)),  # < 98
            # ("ALCOHOL", "alcohol_use", 9),
            ("DRINK", "number_of_drinks", 99, identity, divide_by(98)),  # < 98
            ("WTGAIN", "num_pounds_gained", 99, identity, divide_by(98))  # < 98
        ]

    @staticmethod
    def get_pairs_data(filepath):
        pairs_data = read_csv(filepath)
        return pairs_data

    def get_labels(self, args, ids, benchmark):
        assignments = []
        for id in ids:
            pair = self.get_row(DataAccess.TABLE_PAIRS, id[0])
            assignment = benchmark.get_assignment(id, pair)[0]
            assignments.append(assignment)

        assignments = [t - 2 if t >= 2 else t for t in assignments]

        # get assignments from benchmark first - then select the correct "child_sex"
        sex_0 = np.squeeze(self.get_column(DataAccess.TABLE_PAIRS, ids, "child_sex_0"), axis=-1)
        sex_1 = np.squeeze(self.get_column(DataAccess.TABLE_PAIRS, ids, "child_sex_1"), axis=-1)
        dead_0 = np.squeeze(self.get_column(DataAccess.TABLE_PAIRS, ids, "days_age_at_death_0"), axis=-1)
        dead_1 = np.squeeze(self.get_column(DataAccess.TABLE_PAIRS, ids, "days_age_at_death_1"), axis=-1)
        sex = np.squeeze([np.stack([sex_0, sex_1]).T[idx, t] for idx, t in enumerate(assignments)])
        dead = np.squeeze([np.stack([dead_0, dead_1]).T[idx, t] for idx, t in enumerate(assignments)])

        num_labels = 2**2 - 1
        return sex * 2 ** 0 + dead * 2 ** 1, num_labels

    def get_entry_with_id(self, id, args):
        pair = self.get_row(DataAccess.TABLE_PAIRS, id)

        patient_id = pair[0]
        result = {"pair": pair}

        return patient_id, result

    def standardise_entry(self, entry):
        standardisers = map(lambda x: x[4], self.get_ordered_fields()[3:])
        for i in range(len(entry)):
            entry[i] = standardisers[i](entry[i])
        return entry

    def prepare_batch(self, args, batch_data, benchmark, is_train=False):
        pair_data = np.array(map(lambda x: x["pair"], batch_data))
        ids, input_data = pair_data[:, 0], pair_data[:, 7:]

        assignments = map(benchmark.get_assignment, ids, pair_data)
        treatment_data, batch_y = zip(*assignments)
        treatment_data = np.array(treatment_data)

        if args["with_propensity_batch"] and is_train:
            propensity_batch_probability = float(args["propensity_batch_probability"])
            num_randomised_neighbours = int(np.rint(args["num_randomised_neighbours"]))
            input_data, treatment_data, batch_y = self.enhance_batch_with_propensity_matches(benchmark,
                                                                                             treatment_data,
                                                                                             input_data,
                                                                                             batch_y,
                                                                                             propensity_batch_probability,
                                                                                             num_randomised_neighbours)

        input_data = input_data.astype(np.float32)
        input_data = np.array(map(self.standardise_entry, input_data))

        batch_y = np.array(batch_y)
        batch_x = [
            input_data,
            treatment_data,
        ]
        return batch_x, batch_y
