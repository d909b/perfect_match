#!/usr/bin/env python2
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
import numpy as np

BATCH_NUMBER = "1"

default_params = "--dataset={DATASET_PATH} " \
                 "--with_rnaseq " \
                 "--do_train " \
                 "--do_evaluate " \
                 "--num_hyperopt_runs={NUM_HYPEROPT_RUNS} " \
                 "--do_hyperopt " \
                 "--fraction_of_data_set=1.00 " \
                 "--num_units=60 " \
                 "--num_layers=3 " \
                 "--seed={i} " \
                 "--num_epochs={NUM_EPOCHS} " \
                 "--learning_rate=0.001 " \
                 "--dropout=0.0 " \
                 "--batch_size=50 " \
                 "--output_directory={OUTPUT_FOLDER}/{NAME}/run_{i} " \
                 "--l2_weight=0.000 " \
                 "--imbalance_loss_weight=0.0 " \
                 "--benchmark={DATASET} " \
                 "--num_treatments={NUM_TREATMENTS} " \
                 "--strength_of_assignment_bias={KAPPA} " \
                 "--tcga_num_features={TCGA_FEATURES} " \
                 "--early_stopping_patience={EARLY_STOPPING_PATIENCE} " \
                 "--do_not_save_predictions " \
                 "--propensity_batch_probability={PBM_PROBABILITY} " \
                 "--experiment_index={i} " \
                 "--num_randomised_neighbours=6 "

command_params_pbm = "--method={MODEL_TYPE} " \
                     "--with_propensity_batch " \
                     "--imbalance_loss_weight=0.0 "

command_params_pbm_no_tarnet = "--method={MODEL_TYPE} " \
                               "--with_propensity_batch " \
                               "--imbalance_loss_weight=0.0 " \
                               "--do_not_use_tarnet "

command_params_pbm_mahalanobis = "--method={MODEL_TYPE} " \
                                 "--with_propensity_batch " \
                                 "--imbalance_loss_weight=0.0 " \
                                 "--match_on_covariates "

command_params_psm = "--method=psm " \
                     "--imbalance_loss_weight=0.0 "

command_params_psmpbm = "--method=psmpbm " \
                        "--imbalance_loss_weight=0.0 "

command_params_ganite = "--method=ganite " \
                        "--imbalance_loss_weight=0.0 "

command_params_tarnet = "--method=nn+ " \
                        "--imbalance_loss_weight=0.0 "

command_params_cfrnet = "--method=nn+ " \
                        "--imbalance_loss_weight=1.0 "

command_params_rf = "--method=rf " \
                    "--imbalance_loss_weight=0.0 "

command_params_cf = "--method=cf " \
                    "--imbalance_loss_weight=0.0 "

command_params_bart = "--method=bart " \
                      "--imbalance_loss_weight=0.0 "

command_params_knn = "--method=knn " \
                     "--imbalance_loss_weight=0.0 "

command_params_tarnetpd = "--method=nn+ " \
                          "--with_propensity_dropout " \
                          "--imbalance_loss_weight=0.0 "

command_params_mse = " "
command_params_pehe = "--early_stopping_on_pehe "

command_template = "mkdir -p {OUTPUT_FOLDER}/{NAME}/run_{i}/ && " \
                   "CUDA_VISIBLE_DEVICES='' {SUB_COMMAND} "


def model_is_pbm_variant(model_type):
    return model_type == "pbm" or model_type == "pbm_mahal" or model_type == "pbm_no_tarnet"


def dataset_is_binary_and_has_counterfactuals():
    return DATASET == "ihdp"


def get_dataset_params(DATASET):
    num_tcga_features = None
    if DATASET == "ihdp":  # IHDP - Table 3 (IHDP)
        num_hyperopt_runs = 30
        num_epochs = 400
        early_stopping_patience = 30
        num_repeats = 1000
        treatment_set = [2]
        kappa_set = [0]
        model_set = [
            "pbm", "pbm_mahal", "pbm_no_tarnet",
            "knn", "psm", "psmpbm",
            "rf", "cf", "bart",
            "ganite", "tarnetpd", "tarnet", "cfrnet"
        ]
        es_set = ["mse"]*len(model_set)
        pbm_percentages = [1.0]*len(es_set)
    elif DATASET == "jobs":  # Jobs - Table 3 (Jobs)
        num_hyperopt_runs = 30
        num_epochs = 400
        early_stopping_patience = 30
        num_repeats = 10
        treatment_set = [2]
        kappa_set = [0]
        model_set = ["pbm", "pbm_mahal", "pbm_no_tarnet", "psm", "psmpbm", "tarnetpd"]
        es_set = ["mse"]*len(model_set)
        pbm_percentages = [1.0]*len(es_set)
    elif DATASET == "tcga":  # Influence of higher hidden confounding - Fig. 4
        num_hyperopt_runs = 5
        num_epochs = 100
        early_stopping_patience = 30
        num_repeats = 5
        max_tcga_features = 20531
        num_tcga_features = np.rint(np.arange(0.1, 1.0, 0.1) * max_tcga_features).tolist()
        treatment_set = [8]*len(num_tcga_features)
        kappa_set = [10]*len(num_tcga_features)
        model_set = ["pbm", "tarnetpd", "tarnet", "cfrnet"]
        es_set = ["mse"]*len(model_set)
        pbm_percentages = [1.0]*len(es_set)
    else:  # case: News
        if DATASET == "news_matching_percentage":  # PBM matching percentage influence - Fig. 2
            num_hyperopt_runs = 10
            num_epochs = 100
            early_stopping_patience = 30
            num_repeats = 50
            treatment_set = [8]
            kappa_set = [10]
            pbm_percentages = np.arange(0.1, 1.0, 0.1).tolist()
            model_set = ["pbm"]*len(pbm_percentages)
            es_set = ["mse"]*len(pbm_percentages)
            DATASET = "news"
        elif DATASET == "news_treatment_assignment":  # Treatment assignment bias influence Kappa News-8 - Fig. 3
            num_hyperopt_runs = 10
            num_epochs = 100
            early_stopping_patience = 30
            num_repeats = 50
            treatment_set = [8, 8, 8, 8, 8, 8, 8]
            kappa_set = [5, 7, 10, 12, 15, 17, 20]
            model_set = ["pbm", "tarnetpd", "tarnet", "cfrnet", "cf"]
            es_set = ["mse"]*len(model_set)
            pbm_percentages = [1.0]*len(es_set)
            DATASET = "news"
        else:  # News-k - Table 3 (News-2) and Table 4 (News-4, News-8, News-16)
            num_hyperopt_runs = 10
            num_epochs = 100
            early_stopping_patience = 30
            num_repeats = 50
            treatment_set = [2, 4, 8, 16]
            kappa_set = [10, 10, 10, 7]
            model_set = [
                "pbm", "pbm_mahal", "pbm_no_tarnet",
                "knn", "psm", "psmpbm",
                "rf", "cf", "bart",
                "ganite", "tarnetpd", "tarnet", "cfrnet"
            ]
            es_set = ["mse"]*len(model_set)
            pbm_percentages = [1.0]*len(es_set)

    if num_tcga_features is None:
        num_tcga_features = [0]*len(kappa_set)

    return DATASET, num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, treatment_set, kappa_set, \
           model_set, es_set, pbm_percentages, num_tcga_features


def run(DATASET, DATASET_PATH, OUTPUT_FOLDER, SUB_COMMAND, LOG_FILE):
    DATASET, num_hyperopt_runs, num_epochs, early_stopping_patience, num_repeats, \
    treatment_set, kappa_set, model_set, es_set, pbm_percentages, num_tcga_features \
        = get_dataset_params(DATASET)

    for num_treatments, kappa, tcga_features in zip(treatment_set, kappa_set, num_tcga_features):
        for model_type, early_stopping_type, pbm_percentage in zip(model_set, es_set, pbm_percentages):
            if model_type == "pbm":
                command_params = command_params_pbm
            elif model_type == "ganite":
                command_params = command_params_ganite
            elif model_type == "pbm_mahal":
                command_params = command_params_pbm_mahalanobis
            elif model_type == "pbm_no_tarnet":
                command_params = command_params_pbm_no_tarnet
            elif model_type == "psm":
                command_params = command_params_psm
            elif model_type == "psmpbm":
                command_params = command_params_psmpbm
            elif model_type == "knn":
                command_params = command_params_knn
            elif model_type == "tarnet":
                command_params = command_params_tarnet
            elif model_type == "tarnetpd":
                command_params = command_params_tarnetpd
            elif model_type == "cfrnet":
                command_params = command_params_cfrnet
            elif model_type == "cf":
                command_params = command_params_cf
            elif model_type == "rf":
                command_params = command_params_rf
            elif model_type == "bart":
                command_params = command_params_bart
            else:
                command_params = command_params_tarnet

            if model_is_pbm_variant(model_type):
                if dataset_is_binary_and_has_counterfactuals():
                    command_params = command_params.format(MODEL_TYPE="nn")
                else:
                    command_params = command_params.format(MODEL_TYPE="nn+")

            if early_stopping_type == "pehe":
                command_early_stopping = command_params_pehe
            else:
                command_early_stopping = command_params_mse

            name = "pm_{DATASET}{NUM_TREATMENTS}a{KAPPA}k{PBM_P}{TCGA}_{MODEL_TYPE}_{EARLY_STOPPING_TYPE}_{BATCH_NUMBER}" \
                .format(DATASET=DATASET,
                        KAPPA=kappa,
                        PBM_P="{0:.2f}".format(pbm_percentage) + "p" if pbm_percentage != 1.0 else "",
                        TCGA="{0:d}".format(int(tcga_features)) + "f" if tcga_features != 0 else "",
                        NUM_TREATMENTS=num_treatments,
                        BATCH_NUMBER=BATCH_NUMBER,
                        MODEL_TYPE=model_type,
                        EARLY_STOPPING_TYPE=early_stopping_type)

            for i in range(0, num_repeats):
                local_log_file = LOG_FILE.format(NAME=name, i=i)

                print((command_template + default_params + command_params + command_early_stopping + "&> {LOG_FILE}")
                      .format(SUB_COMMAND=SUB_COMMAND,
                              LOG_FILE=local_log_file,
                              NAME=name,
                              DATASET=DATASET,
                              DATASET_PATH=DATASET_PATH,
                              OUTPUT_FOLDER=OUTPUT_FOLDER,
                              KAPPA=kappa,
                              TCGA_FEATURES=int(tcga_features),
                              NUM_TREATMENTS=num_treatments,
                              NUM_HYPEROPT_RUNS=num_hyperopt_runs,
                              NUM_EPOCHS=num_epochs,
                              PBM_PROBABILITY=pbm_percentage,
                              EARLY_STOPPING_PATIENCE=early_stopping_patience,
                              i=i))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("USAGE: ./run_all_experiments.py {PATH_TO_FOLDER_CONTAINING_MAIN.PY} {DATASET_NAME} {DATABASE_PATH} {OUTPUT_FOLDER}\n"
              "       e.g. ./run_all_experiments.py ./ news ./data ./results\n"
              "       where \n"
              "         PATH_TO_FOLDER_CONTAINING_MAIN.PY is the path to the directory that contains main.py\n"
              "         DATASET_NAME is one of (news, tcga, ihdp, jobs)\n"
              "         DATABASE_PATH is the path to the directory containing tcga.db and news.db \n"
              "                       (See README.md on where to download tcga.db and news.db)\n"
              "         OUTPUT_FOLDER is the path to the directory to which you want to save experiment results.\n",
              file=sys.stderr)
    else:
        # Path where the python executable file is located.
        BINARY_FOLDER = sys.argv[1]

        # Dataset to use. One of: (news, tcga, ihdp, jobs).
        DATASET = sys.argv[2]

        # Path where the SQLite databases for each dataset (tcga.db, news.db) are located.
        DATASET_PATH = sys.argv[3]

        # Folder to write output files and intermediary models to.
        OUTPUT_FOLDER = sys.argv[4]

        # Python command to execute.
        SUB_COMMAND = "python {BINARY}".format(BINARY=os.path.join(BINARY_FOLDER, "main.py"))

        # Folder to write the log file to.
        # Do not change if you want to use the run_results.sh script for result parsing.
        LOG_FILE = os.path.join(OUTPUT_FOLDER, "{NAME}/run_{i}/run.txt")

        run(DATASET, DATASET_PATH, OUTPUT_FOLDER, SUB_COMMAND, LOG_FILE)
