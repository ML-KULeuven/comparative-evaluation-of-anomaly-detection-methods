import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import dask.dataframe as dd

def select_peak_performance(result_df, metric = 'auc'):
    # for each method for each dataset select the row with the highest auc
    indices_of_best_auc = result_df.groupby(['algo_name', 'dataset_id', 'anomaly_fraction', 'dataset_version'], dropna= False)[metric].idxmax()
    return result_df.loc[indices_of_best_auc]

def calculate_best_average_performance(grid_dir, algorithms, grid_versions_to_use, datasets_to_use, result_path):
    for algo in algorithms:
        grid_version = grid_versions_to_use[algo]
        grid_df = pd.read_pickle(grid_dir / f"grid_{algo}_v{grid_version}.pkl")

        # only keep the datasets you need to use
        grid_df = grid_df.groupby(['dataset_id', 'anomaly_fraction']).filter(lambda x: x.name in datasets_to_use.index).reset_index()

        # filter the best average performance
        best_average_performance = select_best_average_performance(grid_df)

        # write
        best_average_performance.to_csv(result_path/f"{algo}.csv")
        best_average_performance.to_pickle(result_path / f"{algo}.pkl")


def select_best_average_performance(result_df):
    # parameters of the method
    parameters = result_df.columns.difference({'flow_id', 'flow_identifier', 'algo_name', 'algorithm_instance_name', 'dataset_id', 'dataset_class','duplicates', 'normalized', 'anomaly_fraction', 'dataset_version', 'preprocessing', 'dataset_type', 'auc', 'ap', 'fit_time_s', 'predict_time_s', 'scores'}).values

    # calculate average auc per parameter setting (ensure that this parameter set has all results)
    max_number_of_results_per_parameter_set = result_df.groupby(list(parameters)).size().max()
    performance_per_parameter = result_df.groupby(list(parameters)).filter(lambda x: x.shape[0] == max_number_of_results_per_parameter_set).groupby(list(parameters)).auc.mean()

    # the best parameters
    best_parameters = performance_per_parameter.idxmax()
    if not isinstance(best_parameters, tuple):
        best_parameters = (best_parameters,)

    # only keep the rows with the best parameters
    best_average_performance = result_df.copy()
    for parameter_name, parameter_value in zip(parameters, best_parameters):
        best_average_performance = best_average_performance[best_average_performance[parameter_name] == parameter_value]

    return best_average_performance


def calculate_validation_set_performances(result_df, validation_set_df):
    columns_in_common = result_df.columns.intersection(validation_set_df.columns).difference(['dataset_id', 'anomaly_fraction'])
    result_df = result_df.drop(columns = columns_in_common )
    temp_df = result_df.merge(validation_set_df, on = ['dataset_id', 'anomaly_fraction'])
    performances = (
        temp_df.apply(calculate_validation_and_test_performance, axis = 1, result_type = 'expand')
        .set_axis(['validation_auc', 'test_auc', 'test_ap'], axis = 1)
    )
    new_result_df = pd.concat([temp_df, performances], axis =1 )
    return new_result_df

def select_best_validation_performance(result_df, validation_set_df, dask_client = None, return_performance_df = False):
    # result_df = add_unique_id(result_df)
    # validation_df = add_unique_id(validation_set_df)
    # one side should drop the data attributes
    result_df = result_df.drop(columns = ['duplicates', 'normalized', 'dataset_version', 'preprocessing']).reset_index()
    # result_df.loc[:, ['validation_auc', 'test_auc', 'test_ap']] =
    temp_df = result_df.merge(validation_set_df, on= ['dataset_id', 'anomaly_fraction'])
    if dask_client is not None:
        ddf = dd.from_pandas(temp_df, npartitions = 80)
        ddf = dask_client.persist(ddf)
        performances = ddf.apply(calculate_validation_and_test_performance, axis = 1, result_type = 'expand', meta = pd.DataFrame(columns = [0,1,2], dtype = 'float'))
        performances = performances.compute()
    else:
        performances = temp_df.apply(calculate_validation_and_test_performance, axis = 1, result_type = 'expand')
    # return select_peak_performance(result_df, metric = 'validation_auc')
    performances.columns = ['validation_auc', 'test_auc', 'test_ap']
    new_result_df = pd.concat([temp_df, performances], axis = 1)
    peak_performance_df = select_peak_performance(new_result_df[~new_result_df.validation_auc.isna()], 'validation_auc')
    df = (
        peak_performance_df
        # rename columns
        .rename(
            columns = dict(auc = 'full_auc', ap = 'full_ap')
        )
    )
    if not return_performance_df:
        return df
    return df, new_result_df

def calculate_validation_and_test_performance(row):
    labels = row.labels
    scores = row.scores
    validation_indices = row.validation_indices
    selector = np.zeros(labels.shape[0])
    selector[validation_indices] = 1
    selector = selector.astype('bool')
    validation_auc = roc_auc_score(labels[selector], scores[selector])
    test_auc = roc_auc_score(labels[~selector], scores[~selector])
    test_ap = average_precision_score(labels[~selector], scores[~selector])
    return validation_auc, test_auc, test_ap