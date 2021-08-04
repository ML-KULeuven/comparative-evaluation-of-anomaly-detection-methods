import time

import numpy as np
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.cblof import CBLOF

from .benchmarkflow import BareBenchmarkFlow


class PyodFlow(BareBenchmarkFlow):
    MODEL = None

    @staticmethod
    def get_algo_config(*args, **kwargs):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs"}}

    def get_algo(self, dataset, **algo_config):
        # collect ingoing information
        print(algo_config)
        X = dataset.X
        real_contamination = dataset.contamination

        if "contamination" in algo_config:
            # algorithm needs contamination parameter
            # so set it correctly
            algo_config["contamination"] = real_contamination

        model = self.MODEL(**algo_config)

        tick = time.time()
        model.fit(X)
        tock = time.time()

        fit_time_s = tock - tick

        return model, fit_time_s

    @staticmethod
    def ask_algo(dataset, model):
        # perform duty
        tick = time.time()
        scores = model.decision_scores_
        tock = time.time()

        # note: in the case of Pyod will always be extremely low!
        predict_time_s = tock - tick

        return scores, predict_time_s


class HBOSFlow(PyodFlow):
    MODEL = HBOS
    STR = "HBOS"

    @staticmethod
    def get_algo_config(
        contamination="auto", n_bins="auto", alpha=0.1, tol=0.1, **kwargs
    ):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs"}}

    def get_algo(self, data, **config):
        # We need a tiny, tiny bit extra work here.
        X = data.X

        if "n_bins" in config and config["n_bins"] in {"sqrt_of_instances"}:
            config["n_bins"] = int(
                np.sqrt(X.shape[0])
            )  # automatically configure n_bins

        return super().get_algo(data, **config)


class IFFlow(PyodFlow):
    STR = "IF"
    MODEL = IForest

    @staticmethod
    def get_algo_config(
        contamination="auto", n_estimators=100, behaviour="new", **kwargs
    ):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs"}}


class KNNFlow(PyodFlow):
    STR = "KNN"
    MODEL = KNN

    @staticmethod
    def get_algo_config(
        contamination="auto", n_neighbors=5, method="largest", **kwargs
    ):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs"}}

    def get_algo(self, data, **config):
        # We need a tiny, tiny bit extra work here.
        X = data.X

        if "n_neighbors" in config and config["n_neighbors"] in {"rule_of_thumb"}:
            config["n_neighbors"] = int(
                max(10, int(0.03*X.shape[0]))
            )  # automatically configure n_bins

        return super().get_algo(data, **config)

class LOFFlow(PyodFlow):
    STR = "LOF"
    MODEL = LOF

    @staticmethod
    def get_algo_config(contamination="auto", n_neighbors=20, leaf_size=20, **kwargs):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs"}}

    def get_algo(self, data, **config):
        # We need a tiny, tiny bit extra work here.
        X = data.X

        if "n_neighbors" in config and config["n_neighbors"] in {"rule_of_thumb"}:
            config["n_neighbors"] = int(
                max(10, int(0.03*X.shape[0]))
            )  # automatically configure n_bins

        return super().get_algo(data, **config)

class CBLOFFlow(PyodFlow):
    STR = 'CBLOF'
    MODEL = CBLOF

    @staticmethod
    def get_algo_config(contamination="auto", n_cluster = 8, alpha = 0.9, beta = 5, random_state = None, **kwargs):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs"}}

class OCSVMFlow(PyodFlow):
    STR = 'OCSVM'
    MODEL = OCSVM

    @staticmethod
    def get_algo_config(contamination="auto", kernel = 'rbf', nu = 0.5, degree = 3, gamma = 'auto', coef0 = 0.0, schrinking = False):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs"}}
