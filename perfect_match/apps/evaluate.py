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

import glob
import os
import subprocess
import sys
import time
import numpy as np
import keras.backend as K
from shutil import copy
from os.path import join
from datetime import datetime
from perfect_match.apps.parameters import parse_parameters
from perfect_match.data_access.generator import make_keras_generator
from perfect_match.models.benchmarks.ihdp_benchmark import IHDPBenchmark
from perfect_match.models.benchmarks.tcga_benchmark import TCGABenchmark
from perfect_match.models.benchmarks.news_benchmark import NewsBenchmark
from perfect_match.models.benchmarks.jobs_benchmark import JobsBenchmark
from perfect_match.models.benchmarks.twins_benchmark import TwinsBenchmark


class EvaluationApplication(object):
    def __init__(self, args):
        self.args = args
        print("INFO: Args are:", self.args, file=sys.stderr)
        print("INFO: Running at", str(datetime.now()), file=sys.stderr)

        data_dir = self.args["dataset"]
        copy_to_local = self.args["copy_to_local"]
        do_hyperopt_on_lsf = self.args["do_hyperopt_on_lsf"]
        benchmark_type = EvaluationApplication.get_benchmark_name_map()[self.args["benchmark"]]
        benchmark_type.get_db_file_name()
        if copy_to_local and not do_hyperopt_on_lsf:
            tmp_dir = os.environ.get("TMPDIR")
            if tmp_dir is None:
                print("ERROR: Couldn't copy to local because TMPDIR was not set.", file=sys.stderr)
            else:
                print("INFO: Copying dataset from", data_dir, "to local directory", tmp_dir, ".",
                      file=sys.stderr)
                copy(join(data_dir, benchmark_type.get_db_file_name()), tmp_dir)
                data_dir = tmp_dir
                self.args["dataset"] = data_dir

        self.best_score_index = 0
        self.best_score = np.finfo(float).max
        self.best_params = ""
        self.best_model_name = "best_model.npy"
        self.benchmark = benchmark_type(self.args["dataset"], **self.args)
        self.setup()

    @staticmethod
    def get_benchmark_name_map():
        return {
            "tcga": TCGABenchmark,
            "twins": TwinsBenchmark,
            "news": NewsBenchmark,
            "ihdp": IHDPBenchmark,
            "jobs": JobsBenchmark
        }

    @staticmethod
    def method_is_neural_network(method):
        return method == "nn+" or method == "nn" or method == "psm" or method == "ganite"

    def setup(self):
        seed = self.args["seed"]
        print("INFO: Seed is", seed, file=sys.stderr)

        import tensorflow as tf
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # Configure tensorflow not to use the entirety of GPU memory at start.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

    def store_cache(self):
        print("INFO: Nothing to store.", file=sys.stderr)

    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        pass

    def evaluate_model(self, model, test_generator, test_steps, with_print=True, set_name=""):
        pass

    def make_train_generator(self):
        pass

    def make_validation_generator(self):
        pass

    def make_test_generator(self):
        pass

    def get_hyperopt_parameters(self):
        return {}

    def get_best_model_path(self):
        return ""

    def get_prediction_path(self):
        return ""

    def save_predictions(self, model):
        return

    def run(self):
        if self.args["do_merge_lsf"]:
            self.merge_lsf_runs()
        elif self.args["do_hyperopt"]:
            return self.run_hyperopt()
        else:
            return self.run_single()

    def run_single(self, evaluate_against="test"):
        print("INFO: Run with args:", self.args, file=sys.stderr)

        save_predictions = self.args["save_predictions"]

        self.benchmark.initialise(self.args)
        benchmark_generator, orig_benchmark_steps = self.make_train_generator(stratify=False)
        benchmark_generator, benchmark_steps = make_keras_generator(self.args,
                                                                    benchmark_generator,
                                                                    orig_benchmark_steps,
                                                                    batch_size=orig_benchmark_steps,
                                                                    num_losses=1,
                                                                    benchmark=self.benchmark)

        self.benchmark.fit(benchmark_generator, benchmark_steps, orig_benchmark_steps)

        train_generator, train_steps = self.make_train_generator()
        val_generator, val_steps = self.make_validation_generator()
        test_generator, test_steps = self.make_test_generator()

        print("INFO: Built generators with", train_steps,
              "training samples, ", val_steps,
              "validation samples and", test_steps, "test samples.",
              file=sys.stderr)

        model = self.train_model(train_generator,
                                 train_steps,
                                 val_generator,
                                 val_steps)

        if self.args["do_evaluate"]:
            if evaluate_against == "val":
                eval_generator, eval_steps = val_generator, val_steps
            else:
                eval_generator, eval_steps = test_generator, test_steps

            eval_score = self.evaluate_model(model, eval_generator, eval_steps, set_name=evaluate_against)
        else:
            eval_score = None

        if save_predictions:
            self.save_predictions(model)

        test_score = self.evaluate_model(model, test_generator, test_steps,
                                         with_print=evaluate_against == "val", set_name="test")
        return eval_score, test_score

    def bsub_with_args(self, args, dependencies=None, is_gpu=False):
        this_directory = os.path.dirname(os.path.abspath(__file__))
        log_file = join(args["output_directory"], "run.txt")

        def arg_for_key_value(k, v):
            if v is True:
                return "--" + k
            elif v is False:
                return ""
            else:
                return "--" + k + "=" + str(args[k])

        gpu_requested = "-R \"rusage[ngpus_excl_p=1]\"" if is_gpu else ""
        extra_resources = "-R \"rusage[mem=16000,scratch=11000]\" {gpu_requested} -n 4 -W 120:00"\
            .format(gpu_requested=gpu_requested)
        sub_command = "python " + join(this_directory, "main.py")
        arguments = " ".join([arg_for_key_value(k, args[k]) for k in args])
        dependencies = "" if dependencies is None else "-w \"" + " && ".join(map(str, dependencies)) + "\""
        bash_profile = "~/.gpu_bash_profile" if is_gpu else "~/.bash_profile"

        command = "bsub {extra_resources} {dependencies} -N -o {log_file} /bin/bash -c " \
                  "\"source ~/venv/bin/activate && " \
                  "source {bash_profile} && cd ~/bin/ && " \
                  "{sub_command} {arguments}\"".format(sub_command=sub_command,
                                                       arguments=arguments,
                                                       log_file=log_file,
                                                       extra_resources=extra_resources,
                                                       dependencies=dependencies,
                                                       bash_profile=bash_profile)

        output = subprocess.check_output(command, shell=True)
        job_id = output[output.find("<") + 1:output.find(">")]
        return job_id

    def run_singe_on_lsf(self, initial_args, hyperopt_offset):
        run_directory = join(initial_args["output_directory"], "run_" + str(hyperopt_offset))

        if not os.path.isdir(run_directory):
            # Create a sub directory for the LSF job.
            os.mkdir(run_directory)

        initial_args = dict(initial_args)
        new_args = {
            "hyperopt_offset": hyperopt_offset,
            "num_hyperopt_runs": hyperopt_offset+1,
            "do_hyperopt_on_lsf": False,
            "output_directory": run_directory
        }
        initial_args.update(new_args)
        return self.bsub_with_args(initial_args)

    def post_merge_job(self, initial_args, job_ids):
        initial_args = dict(initial_args)
        new_args = {
            "do_merge_lsf": True,
            "copy_to_local": False
        }
        initial_args.update(new_args)
        return self.bsub_with_args(initial_args, dependencies=job_ids, is_gpu=False)

    def merge_lsf_runs(self):
        def find_nth(haystack, needle, n):
            parts = haystack.split(needle, n + 1)
            if len(parts) <= n + 1:
                return -1
            return len(haystack) - len(parts[-1]) - len(needle)

        output_directory = self.args["output_directory"]
        filenames = sorted(glob.glob(join(output_directory, "*/run.txt")))
        with open(join(output_directory, "summary.txt"), 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    contents = infile.read()
                    contents = contents + contents
                    num_run_results = contents.count("Best_test_score")
                    if num_run_results > 1:
                        # Only merge the latest result.
                        last_idx = find_nth(contents, "Best_test_score", num_run_results - 2)
                        contents = contents[last_idx:]
                        contents = contents[contents.find("}")+1:]  # Remove the last line too.
                    outfile.write(contents)

    @staticmethod
    def get_random_hyperopt_parameters(initial_args, hyperopt_parameters, hyperopt_index):
        new_params = dict(initial_args)
        for k, v in hyperopt_parameters.iteritems():
            if isinstance(v, list):
                min_val, max_val = v
                new_params[k] = np.random.uniform(min_val, max_val)
            elif isinstance(v, tuple):
                choice = np.random.choice(v)
                new_params[k] = choice
        # new_params["experiment_index"] = hyperopt_index
        return new_params

    @staticmethod
    def print_run_results(args, hyperopt_parameters, run_index, score, run_time):
        message = "Hyperopt run [" + str(run_index) + "]:"
        best_params_message = ""
        for k in hyperopt_parameters:
            best_params_message += k + "=" + "{:.4f}".format(args[k]) + ", "
        best_params_message += "time={:.4f},".format(run_time) + "score={:.4f}".format(score)
        print("INFO:", message, best_params_message, file=sys.stderr)
        return best_params_message

    def run_hyperopt(self):
        num_hyperopt_runs = int(np.rint(self.args["num_hyperopt_runs"]))
        hyperopt_offset = int(np.rint(self.args["hyperopt_offset"]))
        do_hyperopt_on_lsf = self.args["do_hyperopt_on_lsf"]

        initial_args = dict(self.args)

        print("INFO: Performing hyperparameter optimisation.", file=sys.stderr)

        job_ids, score_dicts, test_score_dicts = [], [], []
        for i in range(num_hyperopt_runs):
            run_start_time = time.time()

            hyperopt_parameters = self.get_hyperopt_parameters()
            self.args = EvaluationApplication.get_random_hyperopt_parameters(initial_args,
                                                                             hyperopt_parameters,
                                                                             hyperopt_index=i)

            if i < hyperopt_offset:
                # Skip until we reached the hyperopt offset.
                continue

            if do_hyperopt_on_lsf:
                job_id = self.run_singe_on_lsf(initial_args, i)
                job_ids.append(job_id)
            else:
                eval_set = "test" if self.args["hyperopt_against_eval_set"] else "val"
                score_dict, test_dict = self.run_single(evaluate_against=eval_set)
                score_dicts.append(score_dict)
                test_score_dicts.append(test_dict)

                score = score_dict["cf_pehe_nn"]

                run_time = time.time() - run_start_time

                # This is necessary to avoid memory leaks when repeatedly building new models.
                K.clear_session()

                best_params_message = EvaluationApplication.print_run_results(self.args,
                                                                              hyperopt_parameters,
                                                                              i, score, run_time)
                if score < self.best_score and self.args["do_train"]:
                    self.best_score_index = i
                    self.best_score = score
                    self.best_params = best_params_message
                    best_model_path = self.get_best_model_path()
                    best_model_dir = os.path.dirname(best_model_path)
                    if os.path.isfile(best_model_path):
                        os.rename(best_model_path, join(best_model_dir, self.best_model_name))
                    if os.path.isfile(best_model_path + ".json"):
                        os.rename(best_model_path + ".json", join(best_model_dir, self.best_model_name + ".json"))

        if do_hyperopt_on_lsf:
            self.post_merge_job(initial_args, job_ids)
        else:
            print("INFO: Best[", self.best_score_index, "] config was", self.best_params, file=sys.stderr)
            self.args = initial_args

            print("INFO: Best_test_score:", test_score_dicts[self.best_score_index], file=sys.stderr)

            for key in score_dicts[0].keys():
                try:
                    values = map(lambda x: x[key], score_dicts)
                    print(key, "=", np.mean(values), np.std(values),
                          "(", np.percentile(values, 2.5), ",", np.percentile(values, 97.5), "),",
                          "median=", np.median(values),
                          "(", np.min(values), ",", np.max(values), "),",
                          file=sys.stderr)
                except:
                    print("ERROR: Could not get key", key, "for all score dicts.", file=sys.stderr)

        if len(score_dicts) != 0:
            return score_dicts[self.best_score_index]


if __name__ == "__main__":
    app = EvaluationApplication(parse_parameters())
    app.run()
