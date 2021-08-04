import pandas as pd


def get_datasets_to_use(processed_path, evaluation_settings, contamination_filter, init = None):
    common_datasets = init
    for evaluation_setting in evaluation_settings:
        if (processed_path / evaluation_setting / 'dataset_info.csv').exists():
            dataset_info = pd.read_csv(processed_path / evaluation_setting / 'dataset_info.csv')
            if common_datasets is None:
                common_datasets = dataset_info
            else:
                # calculate intersection of indices
                common_datasets = pd.merge(common_datasets, dataset_info.drop(columns = ['dataset_version']), how = 'inner', on = ['dataset_id', 'anomaly_fraction'], validate = '1:1')

    if common_datasets is None:
        # no evaluation settings specified datasets explicitly so they all have all results
        # so just read a result at random and use that
        if (processed_path/evaluation_settings[0]/'HBOS.csv').exists():
            df = pd.read_csv(processed_path/evaluation_settings[0]/'HBOS.csv')
        else:
            df = pd.read_csv(processed_path/evaluation_settings[0]/'average'/'HBOS.csv')
        common_datasets = df[['dataset_id', 'anomaly_fraction', 'dataset_version']].drop_duplicates()
    datasets_to_use = apply_contamination_filter(contamination_filter, common_datasets)
    return datasets_to_use.set_index(['dataset_id', 'anomaly_fraction'])


def select_contamination(contamination, result_df):
    return result_df[result_df.anomaly_fraction == contamination]


def original_or_random(result_df):
    # first criteria if evaluation_setting in the dataframe
    original_index = result_df.index.names
    result_df = result_df.reset_index()
    if 'evaluation_setting' in result_df.columns:
        results_per_dataset = result_df.groupby(['dataset_id', 'anomaly_fraction'], dropna=False)[
            'evaluation_setting'].nunique()
        datasets_with_all_results = results_per_dataset.index[results_per_dataset == results_per_dataset.max()]
        result_df = result_df.set_index(['dataset_id', 'anomaly_fraction']).loc[datasets_with_all_results,
                    :].reset_index()

    original_useable_versions = (
        result_df
            .pipe(lambda x: x[x.dataset_version.isna() & (x.anomaly_fraction <= 20)])
    )
    remaining = (
        result_df
            .pipe(lambda x: x[~x.dataset_id.isin(original_useable_versions.dataset_id)])
            .pipe(lambda x: x[x.anomaly_fraction <= 20])
            .groupby('dataset_id').sample(1, random_state=1234)
    )
    dfs = []
    for idx, (dataset_id, fraction) in remaining[['dataset_id', 'anomaly_fraction']].iterrows():
        dfs.append(result_df[(result_df.dataset_id == dataset_id) & (result_df.anomaly_fraction == fraction)])

    chosen_datasets = pd.concat([original_useable_versions] + dfs, axis=0)
    # if there is no index don't restore it
    if len(original_index) == 1 and original_index[0] is None:
        return chosen_datasets
    return chosen_datasets.set_index(original_index)
    # return chosen_datasets


def select_max_contamination(result_df, threshold=None):
    if threshold is not None:
        result_df = result_df[result_df.anomaly_fraction <= threshold]
    return result_df.loc[result_df.groupby(['algo_name', 'dataset_id']).anomaly_fraction.idxmax()]


def apply_contamination_filter(name, df):
    return dataset_filters[name](df)


dataset_filters = dict(
    contamination2=lambda x: select_contamination(2, x),
    contamination5=lambda x: select_contamination(5, x),
    contamination10=lambda x: select_contamination(10, x),
    select_original=lambda x: select_max_contamination(x),
    select_original_max20=lambda x: select_max_contamination(x, 20),
    original_or_random=lambda x: original_or_random(x),
    all=lambda x: x,
)