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
import sys
import numpy as np
from ..apps.util import error
from os.path import join, isdir, exists
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class ModelFactory(object):
    MODEL_NAME = "model.json"
    WEIGHTS_NAME = "weights.npz"

    @staticmethod
    def save_object(object, file_path):
        pickle.dump(object, open(file_path, "w"), pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_object(file_path):
        return pickle.load(open(file_path, "r"))

    @staticmethod
    def get_file_names(folder_name):
        model_file = join(folder_name, ModelFactory.MODEL_NAME)
        weights_file = join(folder_name, ModelFactory.WEIGHTS_NAME)
        return model_file, weights_file

    @staticmethod
    def load_model(folder_name):
        model_file, weights_file = ModelFactory.get_file_names(folder_name)

        model_specification = ModelFactory.load_model_specification(model_file)
        model = model_from_json(model_specification)

        weights = ModelFactory.load_weights(weights_file)
        model.set_weights(weights)
        return model

    @staticmethod
    def save_model(model, folder_name):
        model_file, weights_file = ModelFactory.get_file_names(folder_name)

        if exists(folder_name) and not isdir(folder_name):
            error(folder_name, "exists and is not a directory. Could not save.")
            raise Exception(folder_name + "exists and is not a directory.")

        if not exists(folder_name):
            os.mkdir(folder_name)

        ModelFactory.save_model_specification(model, model_file)
        ModelFactory.save_weights(model, weights_file)

    @staticmethod
    def load_model_specification(file_path):
        with open(file_path, "r") as model_specification_file:
            content = model_specification_file.read()
            return content

    @staticmethod
    def save_model_specification(model, file_path):
        json_string = model.to_json()
        with open(file_path, "w") as model_specification_file:
            model_specification_file.write(json_string)

    @staticmethod
    def save_weights(model, file_path):
        weights = dict([(str(i), weight)
                        for i, weight in enumerate(model.get_weights())])
        np.savez(file_path, **weights)

    @staticmethod
    def load_weights(file_path):
        weights = np.load(file_path)
        num_weights = len(weights.files)
        weight_list = [weights[str(idx) + ".npy"] for idx in range(num_weights)]
        return weight_list


class ModelFactoryCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelFactoryCheckpoint, self).\
            __init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    error('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            error('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            ModelFactory.save_weights(self.model, filepath)
                        else:
                            ModelFactory.save_model(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            error('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    error('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    ModelFactory.save_weights(self.model, filepath)
                else:
                    ModelFactory.save_model(self.model, filepath)
