import abc
import copy
import itertools
import warnings
import pandas as pd
import numpy as np
from typing import List
import altair as alt


def register_datasource(*keywords):
    """
    This works BUT the file containing the decorated class has to be imported in order for
    the decorator to be executed!
    So now just an easy fix and hardcoded the datasource_keywords
    """

    def decorator(inner_class):
        for keyword in keywords:
            DataSource.datasource_keywords[keyword] = inner_class
        return inner_class

    return decorator


class DataSource(abc.ABC):
    @staticmethod
    def parse_source_configuration(source_configuration_dict):
        """
        If this transform_configuration_dict resudatasets_to_use[0].lts in multiple sources
        each of these sources is yielded one after another
        """
        # import here to avoid circular dependencies
        from .outlierevaluationdata import OutlierEvaluationDataSource

        datasource_keywords = dict(
            campos = OutlierEvaluationDataSource,
        )

        for key, value in source_configuration_dict.items():
            data_source = datasource_keywords.get(key, None)
            if data_source is None:
                warnings.warn(f"Unknown source configuration key: {key}")
            else:
                yield from data_source.from_config(value)

    def __repr__(self):
        repr = "{}({})".format(self.classname, self.configstring)
        return repr

    @property
    def classname(self):
        return "{}".format(type(self).__name__)

    @property
    def configstring(self):
        return ", ".join(["{}={}".format(k, v) for k, v in self.to_config().items()])

    def to_config(self):
        config = self.get_record()

        # These are not part of the config itself,
        # but are directly related to the instance that generates the record

        config.pop("dataset_id")
        return config

    def get_record(self):
        raise NotImplementedError

    def get_dataset(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError


class DatasetConfig:
    """ TO DELETE """
    def __init__(self, data_source: DataSource, transforms=()):
        self.data_source = data_source
        self.transforms = transforms
        return

    @property
    def name(self):
        return self.data_source.name

    def get_dataset(self):
        dataset = self.data_source.get_dataset()
        for transform in self.transforms:
            dataset = transform.transform_dataset(dataset)
        dataset.config = self
        return dataset

    @property
    def n_instances(self):
        return self.get_dataset().n_instances

    @property
    def n_dimensions(self):
        return self.get_dataset().n_dimensions

    def get_record(self):
        source_record = self.data_source.get_record()
        for transform in self.transforms:
            source_record.update(transform.get_record())
        return source_record

    def to_config(self):
        source_keyword, source_config = self.data_source.to_config()
        transform_configs = []
        for transform in self.transforms:
            transform_configs.append(transform.to_config())
        return {
            "source": {source_keyword: source_config},
            "transform": {**transform_configs},
        }

    @classmethod
    def from_config(cls, config_dict):
        """
        Makes a DataConfig object based on a configuration dict
        this is assumed to be of the following form:
        {
            "source": <dictionary containing the source configurations>,
            "transform": <dictionary containing transformations>
        }

        The dictionary containing the source configurations contains key-value pairs
        Where the key is one of the keywords used by one of the subclasses of DataSource
        And the value is a dictionary containing the configuration for the corresponding datasource

        The dictionary containing the transform configurations contains key-value paris
        Where the key is one of the keywords used by one of the subclasses of DataTransform
        and the value is a dictionary containing the configuration for the corresponding datasource

        If the source_dictionary yields multiple sources and/or the transform dict yields multiple (tuples) of datatransforms
        Each source is composed with each datatransform
        """

        config_dict = copy.deepcopy(config_dict)

        assert (
            "source" in config_dict
        ), "the keyword source has to be in the configuration dict of a data config"

        source_dict = config_dict.pop("source")

        # Transformations
        transform_dict = config_dict.pop("transform", None)

        if len(config_dict) > 0:
            warnings.warn(
                f"ignored configurations in datasetconfig: {config_dict.keys()}"
            )

        source_generator = DataSource.parse_source_configuration(source_dict)
        if transform_dict is None:
            return (DatasetConfig(source) for source in source_generator)

        transform_tuple_generator = DataTransform.parse_transform_configuration(
            transform_dict
        )
        return (
            DatasetConfig(source, transformers)
            for source, transformers in itertools.product(
                source_generator, transform_tuple_generator
            )
        )

    def __repr__(self):
        return f"DataConfig({repr(self.data_source)},{repr(self.transforms)})"


class Dataset:
    def __init__(
        self, X, y, configuration=None, name=None, shuffle=True, random_state=42
    ):
        self._X = X
        self._y = y

        if shuffle:
            rng = np.random.default_rng(random_state)
            idxs = np.arange(self.n_instances)
            rng.shuffle(idxs)
            self._X = X[idxs, :]
            self._y = y[idxs]

        self.config = configuration
        if name is not None:
            self._name = name

        self._version = None
        return

    def to_pandas(self, include_labels=False, column_names=None):
        if column_names is None:
            column_names = self._default_column_names()

        df = pd.DataFrame(self.X, columns=column_names)

        if include_labels:
            df["Y"] = self.y

        return df

    def get_dataframe(self, **kwargs):
        return self.to_pandas(**kwargs)

    def plot(self, x="X0", y="X1", color="Anomaly:N"):
        return (
            alt.Chart(self.dataframe.assign(Anomaly = lambda x: x.Y == 1))
            .mark_circle(size=60)
            .encode(
                x=x,
                y=y,
                color=color,
            )
        )

    def marginal_histogram_chart(self, nb_of_bins, dimensions=None):
        import altair as alt

        charts = []
        data_df = self.to_pandas(include_labels=True)
        if dimensions is None:
            columns = data_df.columns[:-1]
        else:
            columns = data_df.columns[dimensions]
        for column in columns:
            chart = (
                alt.Chart(data_df[[column, "Y"]])
                .mark_bar()
                .encode(
                    x=alt.X(column + ":Q", bin=alt.Bin(maxbins=nb_of_bins)), y="count()"
                )
            )
            chart2 = (
                alt.Chart(data_df[[column, "Y"]])
                .mark_bar()
                .encode(
                    x=alt.X(column + ":Q", bin=alt.Bin(maxbins=nb_of_bins)),
                    y="count():Q",
                    color=alt.ColorValue("red"),
                )
                .transform_filter(alt.datum.Y == 1)
            )
            charts.append(
                alt.layer(chart, chart2).properties(title=column).interactive()
            )
        return alt.hconcat(*charts)

    def _default_column_names(self):
        return [f"X{i}" for i in range(0, self.X.shape[1])]

    def get_nominal_attribute_info(self, threshold=2):
        df = self.to_pandas()
        n_unique_values = df.nunique()
        return (n_unique_values <= threshold).to_list()

    @property
    def summary_str(self):
        return f"{self.name} #d={self.nb_dimensions} #i={self.nb_instances} #a={self.contamination:02f}%"

    @property
    def name(self):
        if hasattr(self, "_name"):
            return self._name
        elif self.config is None:
            return "None"
        else:
            return self.config.data_source.name

    @name.setter
    def name(self, value):
        assert isinstance(value, str), "Name needs to be a string."
        self._name = value
        return

    @property
    def id(self):
        return self.name

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def anomaly_labels(self):
        return self.y

    @property
    def nb_instances(self):
        return self.X.shape[0]

    @property
    def n_samples(self):
        return self.nb_instances

    @property
    def nb_samples(self):
        return self.nb_instances

    @property
    def nb_of_samples(self):
        return self.nb_instances

    @property
    def n_instances(self):
        return self.nb_instances

    @property
    def nb_dimensions(self):
        return self.X.shape[1]

    @property
    def n_dimensions(self):
        return self.nb_dimensions

    @property
    def nb_attributes(self):
        return self.nb_dimensions

    @property
    def n_attributes(self):
        return self.nb_dimensions

    @property
    def outlier_indices(self):
        return [i for i, label in enumerate(self.y) if label == 1]

    @property
    def nb_of_anomalies(self):
        return int(sum(self.y))

    @property
    def nb_anomalies(self):
        return self.nb_of_anomalies

    @property
    def n_anomalies(self):
        return self.nb_of_anomalies

    @property
    def nb_of_normal_instances(self):
        return self.nb_instances - self.nb_of_anomalies

    @property
    def n_normal_instances(self):
        return self.nb_of_normal_instances

    @property
    def contamination(self):
        return self.nb_of_anomalies / self.nb_instances

    @property
    def version(self):
        # Extract version from config/source/whatever
        if self._version is None:
            if self.config is None:
                self._version = 0
            elif isinstance(self.config, dict):
                self._version = self.config.get("version", 0)
            else:
                self._version = self.config.data_source.version

        return self._version

    @version.setter
    def version(self, value):
        assert isinstance(value, int)
        assert value >= 0
        self._version = value
        return

    @property
    def columns(self):
        return self.to_pandas().columns.tolist()

    @property
    def dataframe(self):
        return self.to_pandas(include_labels=True)
