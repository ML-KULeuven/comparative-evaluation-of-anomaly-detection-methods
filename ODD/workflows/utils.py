from sklearn.metrics import average_precision_score, roc_auc_score

from affe.io import (
    FN_TEMPLATE_CLASSIC_FLOW,
    abspath,
    check_existence_of_directory,
    dump_object,
    get_flow_directory,
    get_subdirectory_paths,
    get_template_filenames,
    insert_subdirectory,
    mimic_fs,
    get_default_model_filename,
    get_filepath,
)


# IO
def get_io(
    flow_id=0,
    flow_identifier="benchmarks",
    root_levels_up=1,
    fs_depth=2,
    out_directory="out",
    out_parent="root",
    benchmark_datasets_directory="data.raw.outlier_evaluation_semantic",
    exclude_in_scan=frozenset(["notebooks", "visualisation", "tests", "admercs"]),
    basename=None,
    save_model=False,
    load_model=False,
    data_identifier=None,
    model_identifier=None,
):
    """
    Based on some parameters figures out project structure automatically
    A dict with all relevant IO configuration is returned

    """
    # Perform duties
    fs = mimic_fs(
        root_levels_up=root_levels_up,
        depth=fs_depth,
        exclude=exclude_in_scan,
    )

    ## Build the filesystem we desire
    fs, out_key = insert_subdirectory(
        fs, parent=out_parent, child=out_directory, return_key=True
    )

    flow_directory = get_flow_directory(keyword=flow_identifier)
    fs, flow_key = insert_subdirectory(
        fs, parent=out_key, child=flow_directory, return_key=True
    )

    check_existence_of_directory(fs)

    ## Collect relevant paths for later actions in the workflow
    benchmark_datasets_path = abspath(fs, benchmark_datasets_directory)
    flow_directory_paths = get_subdirectory_paths(fs, flow_key)
    flow_filenames_paths = get_template_filenames(
        flow_directory_paths,
        basename=basename,
        idx=flow_id,
        template=FN_TEMPLATE_CLASSIC_FLOW,
    )

    ## Model IO
    model_filename_path = _get_model_filename_path(
        fs,
        load_model,
        save_model,
        data_identifier=data_identifier,
        model_identifier=model_identifier,
        basename=basename,
    )

    # collect outgoing information
    io = dict(
        flow_id=flow_id,
        flow_identifier=flow_identifier,
        fs=fs,
        benchmark_datasets_path=benchmark_datasets_path,
        flow_key=flow_key,
        flow_directory_paths=flow_directory_paths,
        flow_filenames_paths=flow_filenames_paths,
        model_filename_path=model_filename_path,
        load_model=load_model,
        save_model=save_model,
    )

    return io


def _get_model_filename_path(
    fs,
    load_model,
    save_model,
    data_identifier=None,
    model_identifier=None,
    basename=None,
):

    model_filename = _get_model_filename(
        data_identifier=data_identifier,
        model_identifier=model_identifier,
        basename=basename,
    )

    if load_model:
        return get_filepath(
            tree=fs, node="models", filename=model_filename, check_file=True
        )
    elif save_model:
        return get_filepath(
            tree=fs, node="models", filename=model_filename, check_file=False
        )
    else:
        return


def _get_model_filename(
    data_identifier=None,
    model_identifier=None,
    basename=None,
):
    if model_identifier is not None:
        model_filename = get_default_model_filename(
            data_identifier=data_identifier, model_identifier=model_identifier
        )
    else:
        model_filename = get_default_model_filename(
            data_identifier=data_identifier, model_identifier=basename
        )
    return model_filename


# Data
def get_metadata(dataset):

    # perform duties
    print(type(dataset))
    dataset_name = dataset.name
    n_features = dataset.X.shape[1]
    n_instances = dataset.X.shape[0]
    anomaly_fraction = dataset.contamination
    version = dataset.version
    # normalized = dataset.normalized

    # collect outgoing information
    metadata = dict(
        dataset=dataset_name,
        n_features=n_features,
        n_instances=n_instances,
        anomaly_fraction=anomaly_fraction,
        version=version,
    )

    # metadata = {**metadata, **dataset.config}
    return metadata


def get_actual_data(dataset):
    preprocessing = dict(
        X=dataset.X, y=dataset.anomaly_labels, anomaly_labels=dataset.anomaly_labels
    )

    return preprocessing


# Analysis
def get_new_analysis(results, dataset):
    """
    Analyses the results
    """
    # collect ingoing information
    y_true = dataset["anomaly_labels"]
    scores = results["scores"]
    #
    # y_true = answers.get("y_true")
    # scores = answers.get("scores")
    # fit_time_s = model.get("fit_time_s")
    # predict_time_s = answers.get("predict_time_s")

    # perform duties
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    # collect outgoing information
    analysis = dict(
        auc=auc,
        ap=ap,
        # scores=scores,
        # fit_time_s=fit_time_s,
        # predict_time_s=predict_time_s,
    )
    return analysis


# Analysis
def get_analysis(scores, fit_time_s, predict_time_s, dataset):
    # collect ingoing information
    y_true = dataset.anomaly_labels

    # perform duties
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    # collect outgoing information
    analysis_result = dict(
        auc=auc,
        ap=ap,
        fit_time_s=fit_time_s,
        predict_time_s=predict_time_s,
    )


    analysis_result["scores"] = scores

    return analysis_result


# Retention
def get_retention(io, dataset, analysis, config):
    """
    Stores the results and config of this flow
    """
    # collect ingoing information
    # metadata = get_metadata(dataset)
    self.out_dp
    fn_results = io.get("flow_filenames_paths").get("results")
    fn_config = io.get("flow_filenames_paths").get("config")
    fn_model = io.get("flow_filenames_paths").get("model")

    # results = dict(analysis=analysis, metadata=metadata)
    results = dict(analysis=analysis)
    # perform duties
    results_ok = dump_object(results, fn_results)

    config_ok = dump_object(config, fn_config)

    # collect outgoing information
    oks = [results_ok, config_ok]

    return all(oks)
