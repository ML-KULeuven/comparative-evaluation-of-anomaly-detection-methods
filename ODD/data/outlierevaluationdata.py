"""
    All code to handle to easily handle the datasets from https://www.dbs.ifi.lmu.de/research/outlier-evaluation
    Most important things in this file are the OutlierEvaluationOldDataset class that is used to abstract from all data_config details
    And the get_standard_datasets function (which is a helper function to collect select the datasets for benchmarking

"""
import math
from pathlib import Path
import numpy as np

import pandas as pd
from ODD.data.dataset import Dataset, DataSource
from scipy.io import arff

DATASET_PATH = Path(__file__).absolute().parent.parent.parent / "data"

DATASET_DIRS = ("outlier_evaluation_semantic", "outlier_evaluation_literature")


class OutlierEvaluationDataSource(DataSource):
    keywords = ["outlier_evaluation"]

    def __init__(
        self,
        dataset_name,
        duplicates=False,
        normalized=True,
        anomaly_fraction=None,
        version=None,
        preprocessing=None,
        dataset_path=DATASET_PATH,
    ):
        path = get_dataset_path(
            dataset_name,
            duplicates,
            normalized,
            anomaly_fraction,
            version,
            preprocessing,
            dataset_path,
        )
        assert (
            path.exists()
        ), f"trying to create DataSource for non existing outlier evaluation dataset! {path}"
        self.name = dataset_name
        self.duplicates = duplicates
        self.normalized = normalized
        self.anomaly_fraction = anomaly_fraction
        self.version = version
        self.preprocessing = preprocessing
        self.path = path

    def get_record(self):
        return {
            "dataset_id": self.name,
            "dataset_class": "outlierevaluation",
            "duplicates": self.duplicates,
            "normalized": self.normalized,
            "anomaly_fraction": self.anomaly_fraction,
            "dataset_version": self.version,
            "preprocessing": self.preprocessing,
            "dataset_type": self.path.parent.parent.name,
        }

    def get_dataset(self):
        data = arff.loadarff(self.path)
        df = pd.DataFrame(data[0])
        df.drop("id", axis=1, inplace=True)
        df["outlier"] = df["outlier"].transform(
            lambda x: 1 if "yes" in x.decode("utf8") else 0
        )
        X = df.drop("outlier", axis=1)
        y = df["outlier"]
        return Dataset(X.to_numpy(), y.to_numpy(), name=self.name, shuffle=False)

    @staticmethod
    def from_record(record):
        # record = record.fillna(None)
        name = record['dataset_id']
        duplicates = record['duplicates']
        normalized = record['normalized']
        anomaly_percentage = record['anomaly_fraction']
        version = record['dataset_version']
        version = int(version) if not pd.isna(version) else None
        preprocessing = record['preprocessing']
        preprocessing = None if pd.isna(preprocessing) else preprocessing

        try:
            return OutlierEvaluationDataSource(name,duplicates, normalized, anomaly_percentage, version, preprocessing)
        except AssertionError as e:
            return OutlierEvaluationDataSource(name,duplicates, normalized, None, version, preprocessing)

    @staticmethod
    def from_file_name(filename, dataset_path=DATASET_PATH):
        return OutlierEvaluationDataSource(
            get_dataset_name_from_filename(filename),
            has_duplicates_from_filename(filename),
            is_normalized_from_filename(filename),
            get_anomaly_percentage_from_filename(filename),
            get_version_from_filename(filename),
            get_preprocessing_from_filename(filename),
            dataset_path,
        )

    @classmethod
    def from_config(cls, config):
        datasets = config.pop("datasets", "all")
        duplicates = config.pop("duplicates", False)
        normalized = config.pop("normalized", True)
        versions = config.pop("versions", "all")
        anomaly_fraction = config.pop("anomaly_fraction", 5)
        max_anomaly_fraction = config.pop("max_anomaly_fraction", None)
        yield from get_datasets(
            datasets,
            duplicates,
            normalized,
            versions,
            anomaly_fraction=anomaly_fraction,
            max_anomaly_fraction=max_anomaly_fraction
        )

    def __repr__(self):
        return f"OutlierEvaluationDataSource({self.name},{repr(self.duplicates)},{repr(self.normalized)},{repr(self.anomaly_fraction)},{repr(self.version)})"


def get_datasets(
    dataset_names,
    duplicates,
    normalized,
    versions,
    preprocessed=["idf"],
    anomaly_fraction=5,
    max_anomaly_fraction = None,
):
    if not isinstance(anomaly_fraction, list):
        anomaly_fraction = [anomaly_fraction]
    dataset_index = get_dataset_index()
    # filter dataset names
    if dataset_names is not None and "all" not in dataset_names:
        dataset_index = dataset_index[dataset_index.dataset_name.isin(dataset_names)]

    # filter on preprocessing
    dataset_index = dataset_index[
        (dataset_index.preprocessing.isnull())
        | (dataset_index.preprocessing.isin(preprocessed))
    ]

    # filter duplicates and normalized
    dataset_index = dataset_index[
        (dataset_index.duplicates == duplicates)
        & (dataset_index.normalized == normalized)
    ]

    # filter versions
    if versions != 'all':
        dataset_index = dataset_index[
            (dataset_index.version.isna())|(dataset_index.version.isin(versions))
        ]

    # original dataset (or close to original filtered on version)
    original_versions = dataset_index[dataset_index.version.isna() | (dataset_index.anomaly_fraction.isna())].copy()

    # for the ones where the contamination is NaN read the dataset to determine the contamination
    for index, row in original_versions[original_versions.anomaly_fraction.isna()].iterrows():
        contamination = round(
            OutlierEvaluationDataSource.from_file_name(Path(row.dataset_path).name).get_dataset().contamination * 100)
        original_versions.loc[index, 'anomaly_fraction'] = contamination



    # filter based on max anomaly fraction
    if max_anomaly_fraction is not None:
        dataset_index = dataset_index[dataset_index.anomaly_fraction <= max_anomaly_fraction]
        original_versions_to_use = original_versions[original_versions<max_anomaly_fraction]
    else:
        original_versions_to_use = original_versions

    #
    # def anomaly_fraction_version_filter(row, anomaly_fraction):
    #     if math.isnan(row.version) and math.isnan(row.anomaly_fraction):
    #         # print(row)
    #         # raise Exception('this should not happen!')
    #         return True
    #     original_contamination_of_dataset = original_contamination[row.dataset_name]
    #     if math.isnan(original_contamination_of_dataset):
    #         # if no anomaly fraction for the dataset
    #         # print(row)
    #         if versions is not None and "all" not in versions:
    #             return row.version in versions or math.isnan(row.version)
    #         else:
    #             return True
    #     elif original_contamination_of_dataset < anomaly_fraction:
    #         fraction_to_use = original_contamination_of_dataset
    #         # version does not matter
    #         return row.anomaly_fraction == fraction_to_use
    #     else:
    #         fraction_to_use = anomaly_fraction
    #         if versions is not None and "all" not in versions:
    #             return (
    #                 row.anomaly_fraction == fraction_to_use and (row.version in versions or math.isnan(row.version))
    #             )
    #         return row.anomaly_fraction == fraction_to_use

    full_filter = None
    for anomaly_fr in anomaly_fraction:
        if anomaly_fr == -1: # use all the originals
            filter = dataset_index.index.isin(original_versions_to_use.index)
        else:
            # include the original if necessary
            filter1 = dataset_index.index.isin(original_versions_to_use[original_versions_to_use.anomaly_fraction < anomaly_fr].index)
            # otherwise just include this contamination
            filter2 = dataset_index.anomaly_fraction == anomaly_fr
            filter = filter1 | filter2

        if full_filter is None:
            full_filter = filter
        else:
            full_filter = full_filter | filter
    dataset_index = dataset_index[full_filter]

    # we have filtered everything, so now simply read all the datasets
    for dataset_path in dataset_index.dataset_path:
        yield OutlierEvaluationDataSource.from_file_name(Path(dataset_path).name)


def get_dataset_index(dataset_path=DATASET_PATH):
    # Obtain a DataFrame with all datasets and their properties
    data = []
    for data_dir in DATASET_DIRS:
        full_dataset_path = dataset_path / data_dir
        data.extend(get_dataset_index_from_path(full_dataset_path))
    return pd.DataFrame(
        data=data,
        columns=[
            "dataset_name",
            "duplicates",
            "normalized",
            "version",
            "anomaly_fraction",
            "preprocessing",
            "dataset_path",
        ],
    )


def get_dataset_index_from_path(dataset_path: Path):
    dataset_data = []
    for dataset_dir in dataset_path.iterdir():
        for dataset_file in dataset_dir.iterdir():
            duplicates = has_duplicates_from_filename(dataset_file.name)
            version = get_version_from_filename(dataset_file.name)
            normalized = is_normalized_from_filename(dataset_file.name)
            anomaly_fraction = get_anomaly_percentage_from_filename(dataset_file.name)
            name = get_dataset_name_from_filename(dataset_file.name)
            preprocessing = get_preprocessing_from_filename(dataset_file.name)
            dataset_data.append(
                (
                    name,
                    duplicates,
                    normalized,
                    version,
                    anomaly_fraction,
                    preprocessing,
                    str(dataset_file.absolute()),
                )
            )
    return dataset_data


# HELPERS
def get_dataset_path(
    dataset_name,
    duplicates,
    normalized,
    anomaly_fraction,
    version,
    preprocessing,
    dataset_path=DATASET_PATH,
):
    filename = get_dataset_filename(
        dataset_name,
        duplicates,
        normalized,
        anomaly_fraction,
        version,
        preprocessing,
    )
    for datadir in DATASET_DIRS:
        datapath = Path(dataset_path) / datadir / dataset_name
        if datapath.exists():
            return datapath / filename
    raise Exception(f"unkown dataset name: {dataset_name}")


def get_dataset_filename(
    dataset_name, duplicates, normalized, anomaly_fraction, version, preprocessing=None
):
    duplicate_text = None if duplicates is True else "withoutdupl"
    normalized_text = None if normalized is False else "norm"
    anomaly_f_text = (
        f"{int(anomaly_fraction):02d}" if anomaly_fraction is not None else None
    )
    version_text = (
        f"v{int(version):02d}"
        if version is not None and not np.isnan(version)
        else None
    )
    preprocessing_text = preprocessing

    return (
        "_".join(
            text
            for text in [
                dataset_name,
                duplicate_text,
                normalized_text,
                preprocessing_text,
                anomaly_f_text,
                version_text,
            ]
            if text is not None
        )
        + ".arff"
    )


def get_preprocessing_from_filename(filename):
    potential_preprocessing_texts = ["original", "1ofn", "catremoved", "idf"]
    for t in potential_preprocessing_texts:
        if t in filename:
            return t
    return None


def get_version_from_filename(filename):
    filename_no_ex = filename[:-5]
    last_token = filename_no_ex.split("_")[-1]
    if not last_token.startswith("v"):
        return None
    else:
        return int(last_token[1:])


def has_duplicates_from_filename(filename):
    return not "withoutdupl" in filename


def is_normalized_from_filename(filename):
    return "norm" in filename


def get_dataset_name_from_filename(filename):
    return filename.split("_")[0]


def get_anomaly_percentage_from_filename(filename):
    filename_no_ex = filename[:-5]
    tokens = filename_no_ex.split("_")
    if tokens[-1].startswith("v"):
        perc_token = tokens[-2]
    else:
        perc_token = tokens[-1]
    try:
        return int(perc_token)
    except ValueError:
        # there is no anomaly percentage
        return None


if __name__ == "__main__":
    dataset_generator = list(get_datasets("all", False, True, [1], anomaly_fraction=[-1,2,5,10]))
    # print(len(dataset_generator))
    for dataset in dataset_generator:
        print(dataset.get_dataset().summary_str, dataset.version)
