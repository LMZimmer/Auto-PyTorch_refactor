import json
import logging
import os as os
import random
import time
from abc import abstractmethod

import numpy as np

from sklearn.model_selection import train_test_split


class BaseClassifier():

    def __init__(self, name):

        self.configure_logging()

        self.name = name
        self.config = self.get_config()

        self.categoricals = None
        self.all_nan = None
        self.encode_dicts = None
        self.num_classes = None

    def configure_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)

    def get_config(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dirname, "../classifier_configs", self.name + ".json")
        with open(config_path, "r") as f:
            config = json.load(f)
        for k,v in config.items():
            if v=="True":
                config[k] = True
            if v=="False":
                config[k] = False
        return config

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def score(self, X_test, y_test):
        pass
