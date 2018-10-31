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
import numpy as np
from bisect import bisect_right
from perfect_match.data_access.generator import get_last_id_set
from perfect_match.models.benchmarks.ihdp_benchmark import IHDPBenchmark
from perfect_match.models.benchmarks.jobs_benchmark import JobsBenchmark
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, \
    precision_recall_curve, auc, r2_score, average_precision_score


class ModelEvaluation(object):
    @staticmethod
    def make_y(y_f, y_cf, t):
        y = np.zeros((y_f.shape[0], 2))
        y[t == 0, 0] = y_f[t == 0]
        y[t == 0, 1] = y_cf[t == 0]
        y[t == 1, 1] = y_f[t == 1]
        y[t == 1, 0] = y_cf[t == 1]
        y0, y1 = y[:, 0], y[:, 1]
        return y0, y1

    @staticmethod
    def calculate_pehe(y_true_f, y_pred_f, y_true_cf, y_pred_cf, t, mu0, mu1, x,
                       set_name="test", with_print=True, prefix="", num_neighbours=15, reject_outliers=True):
        y0, y1 = ModelEvaluation.make_y(y_pred_f, y_pred_cf, t)

        eff_pred = y1 - y0
        eff_true = mu1 - mu0

        pehe_nn = ModelEvaluation.pehe_nn(y1, y0, y_true_f, x, t,
                                          k=num_neighbours, reject_outliers=reject_outliers)
        pehe_score = np.sqrt(np.mean(np.square(eff_pred - eff_true)))
        ate = np.abs(np.mean(eff_pred) - np.mean(eff_true))

        pehe_nn_k = []
        for k in range(1, num_neighbours+1):
            pehe_nn_k.append(ModelEvaluation.pehe_nn(y1, y0, y_true_f, x, t,
                                                     k=k,
                                                     reject_outliers=False))

        if with_print:
            print("INFO: Performance on", set_name,
                  "RPEHE =", pehe_score,
                  "PEHE_NN =", pehe_nn,
                  "ATE =", ate,
                  "PEHE_NN_k =", pehe_nn_k,
                  file=sys.stderr)
        return {
            prefix + "pehe": pehe_score,
            prefix + "pehe_nn": pehe_nn,
            prefix + "ate": ate,
        }

    @staticmethod
    def calculate_est_pehe(y_true_f, y_pred_f, y_true_cf, y_pred_cf, t, x, e,
                           set_name="test", with_print=True, prefix="", num_neighbours=1, reject_outliers=False,
                           is_jobs=False):
        y0_p, y1_p = ModelEvaluation.make_y(y_pred_f, y_pred_cf, t)
        y0_t, y1_t = ModelEvaluation.make_y(y_true_f, y_true_cf, t)

        eff_pred = y1_p - y0_p
        eff_true = y1_t - y0_t

        pehe_nn = ModelEvaluation.pehe_nn(y1_p, y0_p, y_true_f, x, t,
                                          k=num_neighbours,
                                          reject_outliers=reject_outliers)
        pehe_score = np.sqrt(np.mean(np.square(eff_pred - eff_true)))
        ate = np.abs(np.mean(eff_pred) - np.mean(eff_true))

        pehe_nn_k = []
        for k in range(1, num_neighbours+1):
            pehe_nn_k.append(ModelEvaluation.pehe_nn(y1_p, y0_p, y_true_f, x, t,
                                                     k=k,
                                                     reject_outliers=False))

        if is_jobs:
            att = np.mean(y_true_f[t > 0]) - np.mean(y_true_f[(1 - t + e) > 1])
            att_pred = np.mean(eff_pred[(t + e) > 1])
            bias_att = np.abs(att_pred - att)
            policy_value = ModelEvaluation.policy_val(t[e > 0], y_true_f[e > 0], eff_pred[e > 0])
            policy_risk = 1.0 - policy_value

            if with_print:
                print("INFO: Performance on", set_name,
                      "RPEHE =", pehe_score,
                      "PEHE_NN =", pehe_nn,
                      "PEHE_NN_k_1 =", pehe_nn_k,
                      "ATE =", ate,
                      "ATT =", att,
                      "ATT_pred =", att_pred,
                      "ATT_error =", bias_att,
                      "R_POL =", policy_risk,
                      file=sys.stderr)
            return {
                prefix + "pehe": pehe_score,
                prefix + "pehe_nn": pehe_nn,
                prefix + "ate": ate,
                prefix + "att": att,
                prefix + "att_pred": att_pred,
                prefix + "att_error": bias_att,
                prefix + "policy_risk": policy_risk
            }
        else:
            if with_print:
                print("INFO: Performance on", set_name,
                      "RPEHE =", pehe_score,
                      "PEHE_NN =", pehe_nn,
                      "PEHE_NN_k_1 =", pehe_nn_k,
                      "ATE =", ate,
                      file=sys.stderr)
            return {
                prefix + "pehe": pehe_score,
                prefix + "pehe_nn": pehe_nn,
                prefix + "ate": ate,
            }

    @staticmethod
    def calculate_statistics_binary(y_true, y_pred, set_name, with_print):
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            # Choose optimal threshold based on closest-to-top-left selection on ROC curve.
            optimal_threshold_idx = np.argmin(np.linalg.norm(np.stack((fpr, tpr)).T -
                                                             np.repeat([[0., 1.]], fpr.shape[0], axis=0), axis=1))
            threshold = thresholds[optimal_threshold_idx]
            y_pred_thresholded = (y_pred > threshold).astype(np.int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresholded).ravel()
            auc_score = roc_auc_score(y_true, y_pred)

            sens_at_95spec_idx = bisect_right(fpr, 0.05)
            if sens_at_95spec_idx == 0:
                # Report 0.0 if specificity goal can not be met.
                sens_at_95spec = 0.0
            else:
                sens_at_95spec = tpr[sens_at_95spec_idx - 1]

            if auc_score < 0.5:
                print("INFO: Inverting AUC.", file=sys.stderr)
                auc_score = 1. - auc_score

            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc_score = auc(recall, precision, reorder=False)

            r2 = r2_score(y_true, y_pred)

            specificity = float(tn) / (tn + fp) if tp + fp != 0 else 0
            sensitivity = float(tp) / (tp + fn) if tp + fn != 0 else 0
            ppv = float(tp) / (tp + fp) if tp + fp != 0 else 0
            npv = float(tn) / (tn + fn) if tn + fn != 0 else 0

            f1_value = f1_score(y_true, y_pred_thresholded)

            if with_print:
                print("INFO: Performance on", set_name,
                      "AUROC =", auc_score,
                      ", with AUPRC =", auprc_score,
                      ", with r^2 =", r2,
                      ", with accuracy =", accuracy_score(y_true, y_pred_thresholded),
                      ", with mean =", np.mean(y_true),
                      ", with f1 =", f1_value,
                      ", with specificity =", specificity,
                      ", with sensitivity =", sensitivity,
                      ", with sens@95spec =", sens_at_95spec,
                      ", with PPV =", ppv,
                      ", with NPV =", npv,
                      file=sys.stderr)
            return {
                "auroc": auc_score,
                "auprc": auprc_score,
                "f1": f1_value,
                "sens@95spec": sens_at_95spec,
                "ppv": ppv,
                "npv": npv,
                "specificity": specificity,
                "sensitivity": sensitivity
            }
        except:
            print("WARN: Score calculation failed. Most likely, there was only one class present in y_true.",
                  file=sys.stderr)
            return {}

    @staticmethod
    def calculate_statistics_multiclass(y_true, y_pred, set_name, with_print):
        from keras.utils import to_categorical

        # Remove columns where all y_true are 0 - this would cause an error in calculating the statistics.
        present_columns = y_true.any(axis=0)
        y_true = y_true[:, present_columns]
        y_pred = y_pred[:, present_columns]

        # TODO: Check for all columns being not present.

        try:
            auc_score = roc_auc_score(y_true, y_pred, average="weighted")

            if auc_score < 0.5:
                print("INFO: Inverting AUC.", file=sys.stderr)
                auc_score = 1. - auc_score

            auprc_score = average_precision_score(y_true, y_pred, average="weighted")
            r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")

            y_thresh = to_categorical(np.argmax(y_pred, axis=-1), num_classes=y_pred.shape[-1])
            f1 = f1_score(y_true, y_thresh, average="weighted")

            if with_print:
                print("INFO: Performance on", set_name,
                      "AUROC (weighted) =", auc_score,
                      ", with AUPRC (weighted) =", auprc_score,
                      ", with r^2 (weighted) =", r2,
                      ", with f1 (weighted) =", f1,
                      file=sys.stderr)
            return {
                "auroc": auc_score,
                "auprc": auprc_score,
                "f1": f1,
                "r2": r2,
            }
        except:
            print("WARN: Score calculation failed. Most likely, there was only one class present in y_true.",
                  file=sys.stderr)
            return {}

    @staticmethod
    def policy_val(t, yf, eff_pred):
        # SOURCE: https://github.com/clinicalml/cfrnet, MIT-License
        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0
        treat_overlap = (policy == t) * (t > 0)
        control_overlap = (policy == t) * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value
        return policy_value

    @staticmethod
    def pdist2(X, Y):
        # SOURCE: https://github.com/clinicalml/cfrnet, MIT-License
        """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
        C = -2 * X.dot(Y.T)
        nx = np.sum(np.square(X), 1, keepdims=True)
        ny = np.sum(np.square(Y), 1, keepdims=True)
        D = (C + ny.T) + nx

        return np.sqrt(D + 1e-8)

    @staticmethod
    def cf_nn(x, t, k=5):
        # SOURCE: https://github.com/clinicalml/cfrnet, MIT-License
        It = np.array(np.where(t == 1))[0, :]
        Ic = np.array(np.where(t == 0))[0, :]

        x_c = x[Ic, :]
        x_t = x[It, :]

        D = ModelEvaluation.pdist2(x_c, x_t)

        sorted_t = np.argsort(D, 0)
        sorted_c = np.argsort(D, 1)

        nn_t, nn_c = [], []
        for i in range(k):
            nn_t.append(Ic[sorted_t[i]])
            nn_c.append(It[sorted_c[:, i]])

        return nn_t, nn_c

    @staticmethod
    def pehe_nn(y1, y0, y, x, t, k=5, nn_t=None, nn_c=None, reject_outliers=True):
        # SOURCE: https://github.com/clinicalml/cfrnet, MIT-License
        It = np.array(np.where(t == 1))[0, :]
        Ic = np.array(np.where(t == 0))[0, :]

        if nn_t is None or nn_c is None:
            k = min(len(It), len(Ic), k)
            nn_t, nn_c = ModelEvaluation.cf_nn(x, t, k)

        def do_reject_outliers(data, m=3):
            return data[abs(data - np.mean(data)) < m * np.std(data)]

        eff_nn = []
        for idx in range(len(nn_t)):
            y_cf_approx = np.copy(y)
            y_cf_approx[It] = y[nn_t[idx]]
            y_cf_approx[Ic] = y[nn_c[idx]]
            y_m0, y_m1 = ModelEvaluation.make_y(y, y_cf_approx, t)
            eff_nn.append(y_m1 - y_m0)
        eff_nn = np.mean(eff_nn, axis=0)

        eff_pred = y1 - y0
        if reject_outliers:
            delta = do_reject_outliers(eff_pred - eff_nn)
        else:
            delta = eff_pred - eff_nn
        pehe_nn = np.sqrt(np.mean(np.square(delta)))
        return pehe_nn

    @staticmethod
    def calculate_statistics_counterfactual(y_true, y_pred, set_name, with_print, prefix=""):
        mse_score = np.mean(np.square(y_true - y_pred))
        rmse_score = np.sqrt(mse_score)
        if with_print:
            print("INFO: Performance on", set_name,
                  "MSE =", mse_score,
                  "RMSE =", rmse_score,
                  file=sys.stderr)
        return {
            prefix + "mse": mse_score,
            prefix + "rmse": rmse_score,
        }

    @staticmethod
    def collect_all_outputs(model, generator, num_steps):
        all_outputs = []
        for _ in range(num_steps):
            generator_outputs = next(generator)
            if len(generator_outputs) == 3:
                batch_input, labels_batch, sample_weight = generator_outputs
            else:
                batch_input, labels_batch = generator_outputs

            all_outputs.append((model.predict(batch_input), labels_batch))
        return all_outputs

    @staticmethod
    def get_y_from_outputs(model, all_outputs, num_steps, selected_slice, selected_index):
        batch_size, y_pred, y_true = 0, [], []
        if hasattr(model, "output"):
            output_dim = model.output[selected_index].shape[-1]
        else:
            output_dim = 1

        for current_step in range(num_steps):
            model_outputs, labels_batch = all_outputs[current_step]

            if isinstance(model_outputs, list):
                model_outputs = model_outputs[selected_slice]

            if isinstance(labels_batch, list):
                labels_batch = labels_batch[selected_slice]

            if isinstance(labels_batch, list) or isinstance(labels_batch, np.ndarray):
                batch_size = len(labels_batch)
            else:
                batch_size = 1

            y_pred.append(model_outputs)
            y_true.append(labels_batch)

        if not (isinstance(y_true[0], list) or isinstance(y_true[0], np.ndarray)):
            y_true = np.array(y_true)
        else:
            y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        if output_dim != 1:
            y_true = y_true.reshape((-1, output_dim))
            y_pred = y_pred.reshape((-1, output_dim))
        else:
            y_pred = np.squeeze(y_pred)

        if (y_true.ndim == 2 and y_true.shape[-1] == 1) and \
                (y_pred.ndim == 1 and y_pred.shape[0] == y_true.shape[0]):
            y_pred = np.expand_dims(y_pred, axis=-1)

        assert y_true.shape[-1] == y_pred.shape[-1]
        assert y_true.shape[0] == y_pred.shape[0]
        assert y_true.shape[0] == num_steps * batch_size
        return y_pred, y_true, output_dim

    @staticmethod
    def evaluate(model, generator, num_steps, set_name="Test set", selected_slices=list([-1]), with_print=True):
        all_outputs = ModelEvaluation.collect_all_outputs(model, generator, num_steps)

        for i, selected_slice in enumerate(selected_slices):
            y_pred, y_true, output_dim = ModelEvaluation.get_y_from_outputs(model, all_outputs, num_steps,
                                                                            selected_slice, i)

            if output_dim == 1:
                # TODO: Switch for regression setting.
                score_dict = ModelEvaluation.calculate_statistics_binary(y_true, y_pred,
                                                                         set_name + str(i), with_print)
            else:
                score_dict = ModelEvaluation.calculate_statistics_multiclass(y_true, y_pred,
                                                                             set_name + str(i), with_print)
        return score_dict

    @staticmethod
    def evaluate_counterfactual(model, generator, num_steps, benchmark, set_name="Test set", with_print=True,
                                selected_slice=-1, stateful_benchmark=True):
        if stateful_benchmark:
            benchmark.set_assign_counterfactuals(True)

        is_ihdp = isinstance(benchmark, IHDPBenchmark)
        is_jobs = isinstance(benchmark, JobsBenchmark)

        all_outputs, all_x, all_treatments, all_mu0, all_mu1, all_e = [], [], [], [], [], []
        for _ in range(num_steps):
            generator_outputs = next(generator)
            if len(generator_outputs) == 3:
                batch_input, labels_batch, sample_weight = generator_outputs
            else:
                batch_input, labels_batch = generator_outputs

            if isinstance(labels_batch, list):
                labels_batch = labels_batch[selected_slice]

            if is_ihdp:
                id_set = get_last_id_set()
                result = np.array(benchmark.data_access.get_rows(id_set, columns="mu0, mu1"))
                all_mu0.append(result[:, 0])
                all_mu1.append(result[:, 1])
            elif is_jobs:
                id_set = get_last_id_set()
                result = np.array(benchmark.data_access.get_rows(id_set, columns="e"))
                all_e.append(result[:, 0])

            treatment_outputs = []
            for treatment_idx in range(benchmark.get_num_treatments()):
                not_none_indices = np.where(np.not_equal(labels_batch[:, treatment_idx], None))[0]
                if len(not_none_indices) == 0:
                    continue

                original_treatment = batch_input[1][not_none_indices]
                current_batch_input = [np.copy(batch_input[0]),\
                                       np.ones_like(batch_input[1])*treatment_idx]

                model_output = model.predict(current_batch_input)
                if isinstance(model_output, list):
                    model_output = model_output[selected_slice]

                none_indices = np.where(np.equal(labels_batch[:, treatment_idx], None))[0]

                if len(none_indices) != 0:
                    full_length = len(labels_batch)
                    inferred_labels = np.array([None]*full_length)
                    inferred_labels[not_none_indices] = labels_batch[not_none_indices, treatment_idx]
                    result = (model_output, inferred_labels)
                else:
                    result = (model_output, labels_batch[not_none_indices, treatment_idx])
                treatment_outputs.append(result)

            y_pred = np.column_stack(map(lambda x: x[0], treatment_outputs))
            y_true = np.column_stack(map(lambda x: x[1], treatment_outputs))

            all_outputs.append((y_pred, y_true))
            all_x.append(batch_input[0])
            all_treatments.append(batch_input[1])

        all_x = np.concatenate(all_x, axis=0)
        all_treatments = np.concatenate(all_treatments, axis=0)

        if is_ihdp:
            all_mu0 = np.concatenate(all_mu0, axis=0)
            all_mu1 = np.concatenate(all_mu1, axis=0)
        elif is_jobs:
            all_e = np.concatenate(all_e, axis=0)

        y_pred, y_true, _ = ModelEvaluation.get_y_from_outputs(model, all_outputs, num_steps,
                                                               selected_slice=-1, selected_index=0)

        y_pred_f, y_true_f = y_pred[np.arange(len(y_pred)), all_treatments], \
                             y_true[np.arange(len(y_true)), all_treatments]

        num_treatments = benchmark.get_num_treatments()
        y_pred_cf, y_true_cf = np.zeros((len(y_pred_f) * (num_treatments-1))), \
                               np.zeros((len(y_pred_f) * (num_treatments-1)))

        for treatment in range(num_treatments-1):
            for idx in range(len(y_pred_f)):
                cf_indices = np.arange(num_treatments)
                cf_indices = np.delete(cf_indices, all_treatments[idx])
                y_pred_cf[idx + len(y_pred_f)*treatment] = y_pred[idx, cf_indices[treatment]]
                y_true_cf[idx + len(y_pred_f)*treatment] = y_true[idx, cf_indices[treatment]]

        score_dict_f = ModelEvaluation.calculate_statistics_counterfactual(y_true_f, y_pred_f,
                                                                           set_name + "_f", with_print, prefix="f_")
        score_dict_cf = ModelEvaluation.calculate_statistics_counterfactual(y_true_cf, y_pred_cf,
                                                                            set_name + "_cf", with_print, prefix="cf_")

        y_true_w = np.concatenate([y_true_f, y_true_cf], axis=0)
        y_pred_w = np.concatenate([y_pred_f, y_pred_cf], axis=0)
        score_dict_w = ModelEvaluation.calculate_statistics_counterfactual(y_true_w,
                                                                           y_pred_w,
                                                                           set_name + "_w", with_print, prefix="w_")

        score_dict_f.update(score_dict_cf)
        score_dict_f.update(score_dict_w)

        if num_treatments == 2:
            if is_ihdp:
                score_dict_pehe = ModelEvaluation.calculate_pehe(y_true_f, y_pred_f, y_true_cf, y_pred_cf,
                                                                 all_treatments,
                                                                 all_mu0, all_mu1, all_x,
                                                                 set_name=set_name + "_pehe", prefix="cf_",
                                                                 with_print=with_print)
            else:
                score_dict_pehe = ModelEvaluation.calculate_est_pehe(y_true_f, y_pred_f, y_true_cf, y_pred_cf,
                                                                     all_treatments, all_x, all_e,
                                                                     set_name=set_name + "_pehe", prefix="cf_",
                                                                     with_print=with_print,
                                                                     is_jobs=is_jobs)
            score_dict_f.update(score_dict_pehe)
        else:
            list_score_dicts_pehe = []
            for i in range(num_treatments):
                for j in range(num_treatments):
                    if j >= i:
                        continue

                    # i = t0, j = t1
                    t1_indices = np.where(all_treatments == i)[0].tolist()
                    t2_indices = np.where(all_treatments == j)[0].tolist()

                    these_x = np.concatenate([all_x[t1_indices], all_x[t2_indices]], axis=0)
                    y_pred_these_treatments = np.concatenate([y_pred[t1_indices], y_pred[t2_indices]], axis=0)
                    y_true_these_treatments = np.concatenate([y_true[t1_indices], y_true[t2_indices]], axis=0)

                    these_treatments = np.concatenate([np.ones((len(t1_indices),), dtype=int)*i,
                                                       np.ones((len(t2_indices),), dtype=int)*j],
                                                      axis=0)

                    these_y_pred_f = y_pred_these_treatments[np.arange(len(y_pred_these_treatments)),
                                                             these_treatments]
                    these_y_true_f = y_true_these_treatments[np.arange(len(y_pred_these_treatments)),
                                                             these_treatments]

                    inverse_treatments = np.concatenate([np.ones((len(t1_indices),), dtype=int)*j,
                                                         np.ones((len(t2_indices),), dtype=int)*i],
                                                        axis=0)

                    these_y_pred_cf = y_pred_these_treatments[np.arange(len(y_pred_these_treatments)),
                                                              inverse_treatments]
                    these_y_true_cf = y_true_these_treatments[np.arange(len(y_pred_these_treatments)),
                                                              inverse_treatments]

                    these_treatments = np.concatenate([np.zeros((len(t1_indices),), dtype=int),
                                                       np.ones((len(t2_indices),), dtype=int)],
                                                      axis=0)

                    score_dict_pehe = ModelEvaluation.calculate_est_pehe(these_y_true_f, these_y_pred_f,
                                                                         these_y_true_cf, these_y_pred_cf,
                                                                         these_treatments, these_x, all_e,
                                                                         set_name=set_name + "_pehe", prefix="cf_",
                                                                         with_print=False)
                    list_score_dicts_pehe.append(score_dict_pehe)

            score_dict_pehe = {}
            for key in list_score_dicts_pehe[0].keys():
                all_values = [list_score_dicts_pehe[i][key] for i in range(len(list_score_dicts_pehe))]
                score_dict_pehe[key] = np.mean(all_values)
                score_dict_pehe[key + "_std"] = np.std(all_values)
            score_dict_f.update(score_dict_pehe)

            if with_print:
                print("INFO: Performance on", set_name,
                      "RPEHE =", score_dict_pehe["cf_pehe"], "+-", score_dict_pehe["cf_pehe_std"],
                      "PEHE_NN =", score_dict_pehe["cf_pehe_nn"], "+-", score_dict_pehe["cf_pehe_nn_std"],
                      "ATE =", score_dict_pehe["cf_ate"], "+-", score_dict_pehe["cf_ate_std"],
                      file=sys.stderr)

        if stateful_benchmark:
            benchmark.set_assign_counterfactuals(False)
        return score_dict_f
