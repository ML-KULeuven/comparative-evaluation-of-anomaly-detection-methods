import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from ODD.data.outlierevaluationdata import OutlierEvaluationDataSource
from numpy.random import default_rng
from scipy.stats import norm


def generate_validation_sets_statistical(data_df, significance_levels, auc_threshold, reasonable_threshold=0.5,
                                         seed=None):
    random_generator = default_rng(seed)

    # add labels and instance/anom information
    data_df = prepare_data_df(data_df)

    # add threshold to the data_df
    # this could be a simple join but we aren't sure about NaN behavior so we use a workaround
    data_df = data_df.join(auc_threshold)

    all_validation_dfs = dict()
    reference_validation_df = None
    for significance_level in sorted(significance_levels):
        print(f"choosing validation set for p = {significance_level}")
        if reference_validation_df is None:
            validation_df = data_df.copy()
        else:
            validation_df = reference_validation_df.copy()
        # determine the validation set size based on the statistical criterion
        validation_df['validation_set_size'] = data_df.apply(
            lambda x: get_min_validation_set_size_binary_search(x.auc_threshold, np.sum(x.labels) / x.labels.shape[0],
                                                                len(x.labels), significance_level), axis=1)
        # reject datasets where the validation set size is bigger than reasonable threshold
        validation_df = validation_df[
            (validation_df.validation_set_size / validation_df['n_instances']) < reasonable_threshold]

        # determine the number of anomalies and normal instances
        validation_df['n_anomalies_validation_set'] = np.ceil(
            validation_df['contam'] * validation_df['validation_set_size']).astype('int')
        validation_df['n_normals_validation_set'] = (validation_df['validation_set_size'] - validation_df[
            'n_anomalies_validation_set']).astype('int')

        # sample the validation indices (subsamples existing validation_indices)
        validation_df['validation_indices'] = validation_df.apply(select_validation_set,
                                                                  random_generator=random_generator, axis=1)
        check_validation_set_df(validation_df)
        all_validation_dfs[significance_level] = validation_df

        # use the first one as a reference for all the rest
        if reference_validation_df is None:
            reference_validation_df = (
                validation_df
                    .rename(columns={'validation_indices': 'reference_validation_set'})
                    .drop(columns=['n_anomalies_validation_set', 'n_normals_validation_set'])
            )

    return reference_validation_df, all_validation_dfs

def generate_validation_sets_statistical_subset_multiple_runs(data_df, auc_threshold, significance_levels, n_runs = 5, upper_limit = None, reasonable_threshold=0.5,
                                      seed=None):
    """
        This function ensures that every smaller validation set is a subset from the bigger validation set
        Note this is only the case if you stay within one anomaly multiplier
        Something else might be possible with multiple anomaly multipliers but we'll ignore that for now!

    """
    random_generator = default_rng(seed)

    # add labels and instance information
    data_df = prepare_data_df(data_df)

    # add auc threshold information
    data_df = data_df.join(auc_threshold)


    all_validation_dfs = defaultdict(list)
    all_reference_validation_dfs = []
    print('choosing reference validation set')

    for run_idx in range(n_runs):
        current_reference = None
        for significance_level in sorted(significance_levels):
            print(f'choosing validation set for size {significance_level} and run {run_idx}')
            if current_reference is None:
                validation_df = data_df.copy()
            else:
                validation_df = current_reference.copy()

            validation_df['validation_set_size'] = data_df.apply(
                lambda x: get_min_validation_set_size_binary_search(x.auc_threshold,
                                                                    np.sum(x.labels) / x.labels.shape[0],
                                                                    len(x.labels), significance_level), axis=1)
            if upper_limit is not None:
                upper_limit_validation_size = np.ceil(validation_df.n_instances * upper_limit)
                is_over_limit = validation_df.validation_set_size > upper_limit_validation_size
                validation_df.loc[is_over_limit, 'validation_set_size'] = upper_limit_validation_size[is_over_limit]

            # reject datasets where the validation set size is bigger than reasonable threshold
            validation_df = validation_df[(validation_df['validation_set_size'] / validation_df['n_instances']) <= reasonable_threshold]

            # determine the number of anomalies
            validation_df['n_anomalies_validation_set'] = np.ceil(validation_df['contam']* validation_df['validation_set_size']).astype('int')

            # Ensure at least one anomaly in the test set
            has_too_much_anoms = validation_df['n_anomalies_validation_set'] >= validation_df['n_anomalies']
            validation_df.loc[has_too_much_anoms, 'n_anomalies_validation_set' ] = validation_df.loc[has_too_much_anoms, 'n_anomalies'] -1

            validation_df['n_normals_validation_set'] = (validation_df['validation_set_size'] - validation_df['n_anomalies_validation_set']).astype('int')
            validation_df['validation_indices'] = validation_df.apply(select_validation_set, axis=1,
                                                                random_generator=random_generator)
            all_validation_dfs[significance_level].append(validation_df)

            reference = (
                validation_df
                .drop(columns = ['n_normals_validation_set', 'n_anomalies_validation_set', 'reference_validation_set'], errors= 'ignore')
                .rename(columns = {'validation_indices': 'reference_validation_set'})
            )
            if current_reference is None:
                all_reference_validation_dfs.append(reference)
            current_reference = reference

    return all_reference_validation_dfs, all_validation_dfs


def generate_validation_sets_absolute_subset(data_df, absolute_sizes, anomaly_multipliers, reasonable_threshold=0.5,
                                      seed=None):
    """
        This function ensures that every smaller validation set is a subset from the bigger validation set
        Note this is only the case if you stay within one anomaly multiplier
        Something else might be possible with multiple anomaly multipliers but we'll ignore that for now!

    """
    random_generator = default_rng(seed)

    # add labels and instance information
    data_df = prepare_data_df(data_df)

    all_validation_dfs = dict()
    print('choosing reference validation set')
    reference_validation_df = generate_reference_validation_set_for_absolute_sizes(data_df, absolute_sizes,
                                                                                   anomaly_multipliers, reasonable_threshold, seed)

    for anomaly_multiplier in anomaly_multipliers:
        current_reference = reference_validation_df
        for size in sorted(absolute_sizes, reverse = True):
            print(f'choosing validation set for size {size} and anom {anomaly_multiplier}')
            validation_df = current_reference.copy()

            # reject datasets where the validation set size is bigger than reasonable threshold
            validation_df = validation_df[(size / validation_df['n_instances']) <= reasonable_threshold]

            # determine the number of anomalies
            validation_df['n_anomalies_validation_set'] = np.ceil(data_df['contam'] * anomaly_multiplier * size).astype('int')

            # Ensure at least one anomaly in the test set
            has_too_much_anoms = validation_df['n_anomalies_validation_set'] >= validation_df['n_anomalies']
            validation_df.loc[has_too_much_anoms, 'n_anomalies_validation_set' ] = validation_df.loc[has_too_much_anoms, 'n_anomalies'] -1

            validation_df['n_normals_validation_set'] = (size - validation_df['n_anomalies_validation_set']).astype('int')
            validation_df['validation_indices'] = validation_df.apply(select_validation_set, axis=1,
                                                                random_generator=random_generator)
            all_validation_dfs[(size, anomaly_multiplier)] = validation_df
            current_reference = (
                validation_df
                .drop(columns = ['n_normals_validation_set', 'n_anomalies_validation_set', 'reference_validation_set'])
                .rename(columns = {'validation_indices': 'reference_validation_set'})
            )
    return reference_validation_df, all_validation_dfs

def generate_validation_sets_absolute_subset_multiple_runs(data_df, absolute_sizes, n_runs = 5, upper_limit = None, reasonable_threshold=0.5,
                                      seed=None):
    """
        This function ensures that every smaller validation set is a subset from the bigger validation set
        Note this is only the case if you stay within one anomaly multiplier
        Something else might be possible with multiple anomaly multipliers but we'll ignore that for now!

    """
    random_generator = default_rng(seed)

    # add labels and instance information
    data_df = prepare_data_df(data_df)

    all_validation_dfs = defaultdict(list)
    all_reference_validation_dfs = []
    print('choosing reference validation set')

    for run_idx in range(n_runs):
        current_reference = generate_reference_validation_set_for_absolute_sizes(data_df, absolute_sizes,
                                                                                    [1], reasonable_threshold, upper_limit, random_generator.integers(1, 10000000))
        all_reference_validation_dfs.append(current_reference)
        for size in sorted(absolute_sizes, reverse = True):
            print(f'choosing validation set for size {size} and run {run_idx}')
            validation_df = current_reference.copy()

            validation_df['validation_set_size'] = size
            if upper_limit is not None:
                upper_limit_validation_size = np.ceil(validation_df.n_instances * upper_limit)
                is_over_limit = validation_df.validation_set_size > upper_limit_validation_size
                validation_df.loc[is_over_limit, 'validation_set_size'] = upper_limit_validation_size[is_over_limit]

            # reject datasets where the validation set size is bigger than reasonable threshold
            validation_df = validation_df[(validation_df['validation_set_size'] / validation_df['n_instances']) <= reasonable_threshold]

            # determine the number of anomalies
            validation_df['n_anomalies_validation_set'] = np.ceil(validation_df['contam']* validation_df['validation_set_size']).astype('int')

            # Ensure at least one anomaly in the test set
            has_too_much_anoms = validation_df['n_anomalies_validation_set'] >= validation_df['n_anomalies']
            validation_df.loc[has_too_much_anoms, 'n_anomalies_validation_set' ] = validation_df.loc[has_too_much_anoms, 'n_anomalies'] -1

            validation_df['n_normals_validation_set'] = (validation_df['validation_set_size'] - validation_df['n_anomalies_validation_set']).astype('int')
            validation_df['validation_indices'] = validation_df.apply(select_validation_set, axis=1,
                                                                random_generator=random_generator)
            all_validation_dfs[size].append(validation_df)
            current_reference = (
                validation_df
                .drop(columns = ['n_normals_validation_set', 'n_anomalies_validation_set', 'reference_validation_set'])
                .rename(columns = {'validation_indices': 'reference_validation_set'})
            )
    return all_reference_validation_dfs, all_validation_dfs


def generate_validation_sets_absolute(data_df, absolute_sizes, anomaly_multipliers, reasonable_threshold=0.5,
                                      seed=None):
    random_generator = default_rng(seed)

    # add labels and instance information
    data_df = prepare_data_df(data_df)

    all_validation_dfs = dict()
    print('choosing reference validation set')
    reference_validation_df = generate_reference_validation_set_for_absolute_sizes(data_df, absolute_sizes,
                                                                                   anomaly_multipliers, reasonable_threshold, seed)

    for size, anomaly_multiplier in itertools.product(absolute_sizes, anomaly_multipliers):
        print(f'choosing validation set for size {size} and anom {anomaly_multiplier}')
        validation_df = reference_validation_df.copy()

        # reject datasets where the validation set size is bigger than reasonable threshold
        validation_df = validation_df[(size / validation_df['n_instances']) <= reasonable_threshold]

        # determine the number of anomalies
        validation_df['n_anomalies_validation_set'] = np.ceil(data_df['contam'] * anomaly_multiplier * size).astype('int')

        # Ensure at least one anomaly in the test set
        has_too_much_anoms = validation_df['n_anomalies_validation_set'] >= validation_df['n_anomalies']
        validation_df.loc[has_too_much_anoms, 'n_anomalies_validation_set' ] = validation_df.loc[has_too_much_anoms, 'n_anomalies'] -1

        validation_df['n_normals_validation_set'] = (size - validation_df['n_anomalies_validation_set']).astype('int')
        validation_df['validation_indices'] = validation_df.apply(select_validation_set, axis=1,
                                                            random_generator=random_generator)
        all_validation_dfs[(size, anomaly_multiplier)] = validation_df
    return reference_validation_df, all_validation_dfs



def generate_reference_validation_set_for_absolute_sizes(data_df, absolute_sizes, anomaly_multipliers, reasonable_threshold = 0.5, upper_limit = None, seed=None):
    random_generator = default_rng(seed)

    # add labels and instance information
    data_df = prepare_data_df(data_df)

    max_size = max(absolute_sizes)
    min_multiplier = min(anomaly_multipliers)
    max_multiplier = max(anomaly_multipliers)

    data_df['validation_set_size'] = max_size
    if upper_limit is not None:
        print('limiting validation set size')
        upper_limit_validation_size = np.ceil(data_df.n_instances * upper_limit)
        is_over_limit = data_df.validation_set_size > upper_limit_validation_size
        data_df.loc[is_over_limit, 'validation_set_size'] = upper_limit_validation_size[is_over_limit]


    # reject datasets where the validation set size is bigger than reasonable threshold
    data_df = data_df[(data_df['validation_set_size'] / data_df['n_instances']) <= reasonable_threshold]

    min_nb_of_anomalies = np.ceil(data_df['contam'] * data_df['validation_set_size']* min_multiplier)
    # ensure that there is always one anomaly for the test set
    min_nb_of_anomalies[min_nb_of_anomalies>=data_df['n_anomalies']] = data_df.loc[min_nb_of_anomalies>=data_df['n_anomalies'], 'n_anomalies']-1

    max_nb_of_anomalies = np.ceil(data_df['contam'] * data_df['validation_set_size']*  max_multiplier)
    # ensure that there is always one anomaly for the test set
    max_nb_of_anomalies[max_nb_of_anomalies >= data_df['n_anomalies']] = data_df.loc[max_nb_of_anomalies >= data_df[
        'n_anomalies'], 'n_anomalies'] - 1

    nb_of_normals_in_reference = data_df['validation_set_size'] - min_nb_of_anomalies
    nb_of_anomalies_in_reference = max_nb_of_anomalies
    data_df['n_normals_validation_set'] = nb_of_normals_in_reference
    data_df['n_anomalies_validation_set'] = nb_of_anomalies_in_reference
    data_df[['n_normals_validation_set', 'n_anomalies_validation_set']] = data_df[['n_normals_validation_set', 'n_anomalies_validation_set']].astype('int')
    data_df['reference_validation_set'] = data_df.apply(select_validation_set, axis=1,
                                                        random_generator=random_generator)
    return data_df.drop(['n_normals_validation_set', 'n_anomalies_validation_set', 'validation_set_size'], axis=1)


def select_validation_set(row, random_generator):
    n_normals_validation_set = row.n_normals_validation_set
    n_anomalies_validation_set = row.n_anomalies_validation_set
    n_instances = row.n_instances
    labels = row.labels

    if 'reference_validation_set' in row.index:
        indices_to_choose_from = row.reference_validation_set
    else:
        indices_to_choose_from = np.arange(0, n_instances)

    anomaly_indices = indices_to_choose_from[np.where(labels[indices_to_choose_from] == 1)]
    normal_indices = indices_to_choose_from[np.where(labels[indices_to_choose_from] == 0)]

    # print(f'need to sample {n_anomalies_validation_set} anomalies from {len(anomaly_indices)}')
    # print(f'need to sample {n_normals_validation_set} normals from {len(normal_indices)}')

    chosen_anomaly_indices = random_generator.choice(anomaly_indices, n_anomalies_validation_set, replace=False)
    chosen_normal_indices = random_generator.choice(normal_indices, n_normals_validation_set, replace=False)
    chosen_indices = np.concatenate([chosen_normal_indices, chosen_anomaly_indices], axis=0)
    return chosen_indices


def check_validation_set_df(validation_df):
    def check_validation(row):
        labels = row.labels
        validation_indices = row.validation_indices
        selector = np.zeros(labels.shape[0])
        selector[validation_indices] = 1
        selector = selector.astype('bool')
        return np.sum(labels[selector]) > 0

    def check_test(row):
        labels = row.labels
        validation_indices = row.validation_indices
        selector = np.zeros(labels.shape[0])
        selector[validation_indices] = 1
        selector = selector.astype('bool')
        return np.sum(labels[~selector]) > 0

    assert validation_df.apply(check_validation, axis=1).all(axis=None)
    assert validation_df.apply(check_test, axis=1).all(axis=None)


def prepare_data_df(data_df):
    data_df = data_df.copy()
    if 'labels' not in data_df.columns:
        data_df = add_labels(data_df)

    if 'n_instances' not in data_df.columns:
        data_df['n_instances'] = data_df.labels.apply(lambda x: len(x))
        data_df['n_anomalies'] = data_df.labels.apply(lambda x: np.sum(x))
        data_df['contam'] = data_df['n_anomalies'] / data_df['n_instances']
    return data_df


def add_labels(data_df):
    def get_labels(row):
        try:
            dataset = OutlierEvaluationDataSource.from_record(row).get_dataset()
        except Exception as e:
            print(row)
            raise e
        return dataset.y

    data_df = data_df.assign(
        labels=lambda x: x.apply(get_labels, axis=1)
    )
    return data_df

def add_nb_attributes(data_df):
    def get_labels(row):
        try:
            dataset = OutlierEvaluationDataSource.from_record(row).get_dataset()
        except Exception as e:
            print(row)
            raise e
        return dataset.nb_attributes

    data_df = data_df.assign(
        nb_attributes=lambda x: x.apply(get_labels, axis=1)
    )
    return data_df

def probability_random_auc_better_than(auc_threshold, validation_size, contamination):
    # simply lorenzo's formula
    return 1 - norm.cdf(((auc_threshold - 0.5) * validation_size) / np.sqrt(
        (validation_size + 1) / (12 * contamination * (1 - contamination))))


def get_min_validation_set_size_binary_search(auc_threshold, contamination, number_of_instances,
                                              probability_threshold=0.05):
    lower_bound = 1
    upper_bound = number_of_instances
    if probability_random_auc_better_than(auc_threshold, upper_bound, contamination) > probability_threshold:
        return None
    while lower_bound != upper_bound:
        middle = (upper_bound + lower_bound) // 2
        probability = probability_random_auc_better_than(auc_threshold, middle, contamination)
        # probability is to high
        if probability > probability_threshold:
            # make the validation set bigger
            lower_bound = middle + 1
        if probability <= probability_threshold:
            # try to make the validation set smaller
            upper_bound = middle
    return lower_bound


def get_min_validation_set_size(auc_threshold, contamination, probability_threshold=0.05):
    current_size = 1
    while probability_random_auc_better_than(auc_threshold, current_size, contamination) > probability_threshold:
        current_size += 1
    return current_size


def get_average_performance_per_dataset(result_path):
    # assume that in given result_path you have <algo>.csv files
    dfs = [pd.read_csv(path) for path in result_path.glob('*.csv')]
    result_df = pd.concat(dfs, axis=0)
    return result_df.groupby(["dataset_id", "anomaly_fraction", "dataset_version"], dropna=False)[
        ['auc', 'ap']].mean().reset_index(level=2)


def get_min_performance_per_dataset(result_path):
    # assume that in given result_path you have <algo>.csv files
    dfs = [pd.read_csv(path) for path in result_path.glob('*.csv')]
    result_df = pd.concat(dfs, axis=0)
    return result_df.groupby(["dataset_id", "anomaly_fraction", "dataset_version"], dropna=False)[
        ['auc', 'ap']].min().reset_index(level=2)


def get_data_df(grid_path):
    data_df = (
        # just read a gridsearch
        pd.read_pickle(grid_path)
            # select relevant dataset information
            .loc[:, ['dataset_id', 'anomaly_fraction', 'duplicates', 'normalized', 'dataset_version', 'preprocessing']]
            # drop duplicate entries
            .drop_duplicates()
            # add the labels once
            .pipe(add_labels)
            .pipe(add_nb_attributes)
            .drop(columns=['duplicates', 'normalized', 'preprocessing'])
            .set_index(['dataset_id', 'anomaly_fraction'])
    )
    return data_df
