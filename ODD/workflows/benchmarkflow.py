import warnings
from pathlib import Path

import numpy as np
from affe.flow import FlowOne
from affe.io import load_object, dump_object

from .utils import (
    get_analysis,
    get_retention,
)


# Benchmarkflow with text-based interface
class BareBenchmarkFlow(FlowOne):
    """
    This class contains the logic to run a single benchmark

    All the configuration for this benchmark is in the self.config dictionary (stored in the Flow class)
    The keys of this dictionary are:
    - 'io': all input output configuration
    - 'data': the dataset configuration
    - 'analysis': configuration about the analysis (currently UNUSED)
    - 'visuals': configuration about the visuals (currently UNUSED)
    - 'algo': the configuration of the algorithm
    """

    STR = "NULL"

    def __init__(self, experiment_identifier,  flow_identifier, out_path, root_levels_up, dataset_config, algo_config, timeout_s=60, **kwargs):

        if "algorithm_instance_name" in kwargs:
            self.algorithm_instance_name = kwargs["algorithm_instance_name"]
        else:
            self.algorithm_instance_name = None

        self.data_config = dataset_config

        config = dict(
            data=repr(dataset_config),
            algo=algo_config,
        )

        super().__init__(
            experiment_identifier, flow_identifier, root_levels_up, out_path,
            config=config,
            timeout_s=timeout_s,
        )
        self.result_path.parent.mkdir(parents=True, exist_ok = True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        return

    def __str__(self):
        param_string = ["{}={}".format(k, v) for k, v in self.config["algo"].items()]
        param_string = "(" + ", ".join(param_string) + ")"
        return self.STR + param_string

    @staticmethod
    def imports():
        import sys
        import os

        sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

        return

    @staticmethod
    def get_algo_config(**kwargs):
        raise NotImplementedError()


    def get_algo(self, data, **algo_config):
        raise NotImplementedError

    @staticmethod
    def ask_algo(data, m_algo):
        raise NotImplementedError

    @property
    def result_path(self):
        return self.out_exp_dp / 'results' / f"{self.flow_identifier}.pkl"

    @property
    def config_path(self):
        return self.out_exp_dp /'config'/f"{self.flow_identifier}.json"

    def flow(self, config):
        dataset = self.data_config.get_dataset()

        # Fit
        model, fit_time_s = self.get_algo(dataset, **config.get("algo"))

        scores, predict_time_s = self.ask_algo(dataset, model)

        # Verify outcomes!
        scores = self._clean_scores(scores)
        assert all(np.isfinite(scores)), "SCORES ARE NOT FINITE"


        # Analyse the results
        analysis = get_analysis(
            scores, fit_time_s, predict_time_s, dataset
        )

        # store the configuration, analysis and some data_config metadata/configuration

        # results = dict(analysis=analysis, metadata=metadata)
        results = dict(analysis=analysis)
        # perform duties
        results_ok = dump_object(results, self.result_path)
        config_ok = dump_object(config, self.config_path)
        # flow and log are saved by superclass

        return analysis

    def explicit_flow(self):
        assert False
        print("running", self.STR)
        self.io = self.config.get("io")
        self.dataset = self.data_config.get_dataset()

        # Fit
        if self.io.get("load_model"):
            model = load_object(self.io["model_filename_path"])
            model, fit_time_s = self.get_algo(
                self.dataset, model=model, **self.config.get("algo")
            )
        else:
            model, fit_time_s = self.get_algo(self.dataset, **self.config.get("algo"))
        self.m_algo = dict(model=model, fit_time_s=fit_time_s)

        # Predict
        scores, predict_time_s = self.ask_algo(self.dataset, model)
        self.a_algo = dict(scores=scores, predict_time_s=predict_time_s)

        if self.io.get("save_model"):
            dump_object(self.m_algo["model"], self.io["model_filename_path"])

        # Process results
        self.analysis = get_analysis(
            model,
            scores,
            fit_time_s,
            predict_time_s,
            self.dataset,
            **self.config.get("analysis"),
        )
        self.retention_ok = get_retention(
            self.io, self.dataset, self.analysis, self.config
        )

        return self.analysis

    # Custom Convenience Methods
    def get_results(self):
        return load_object(self.result_path)

    def get_results_as_record(self):
        result_file_path = self.result_path
        if not result_file_path.exists():
            return None
        algo_name = self.STR
        flow_id = self.flow_identifier
        algo_config = self.config["algo"]

        data_config = self.data_config
        data_record = data_config.get_record()
        try:
            result = load_object(str(result_file_path))
        except Exception as e:
            print(f"error with flow {self.flow_filepath}")
            return None
        # flatten the dataconfig if necessary
        # processed_data_config = _flatten_dict(data_config)

        record = dict(
            flow_id=flow_id,
            flow_identifier=self.experiment_identifier,
            algo_name=algo_name,
            algorithm_instance_name=self.algorithm_instance_name,
            **algo_config,
            **data_record,
            **result["analysis"],
            # **result["metadata"],
        )
        return record

    # Clean Scores
    @staticmethod
    def _clean_scores(scores):
        all_nonfinite = np.all(~np.isfinite(scores))
        any_nonfinite = np.any(~np.isfinite(scores))

        if all_nonfinite:
            # All non-finite, replace with zeroes and throw a warning of bullshit learning
            scores = np.zeros_like(scores)

            msg = """
            ---------------------------------------------------------
            ABSOLUTELY NOTHING WAS LEARNED, ALL SCORES ARE IDENTICAL.
            BEWARE IN INTERPRETATION!
            ---------------------------------------------------------
            """
            warnings.warn(msg)
            return scores
        elif any_nonfinite:
            msg = """
            ---------------------------------------------------------
            SOME INFINITIES ENCOUNTERED.

            SETTING THEM TO MIN AND MAX OF FINITE SCORES RESPECTIVELY
            ---------------------------------------------------------
            """
            warnings.warn(msg)

            # Some non-finite, some cleaning needed
            minimum = np.min(scores[np.isfinite(scores)])
            maximum = np.max(scores[np.isfinite(scores)])

            scores[np.isnan(scores)] = minimum
            scores[np.isneginf(scores)] = minimum
            scores[np.isposinf(scores)] = maximum
            return scores
        else:
            # All finite, no cleaning needed
            return scores
