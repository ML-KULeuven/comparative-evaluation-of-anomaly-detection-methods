import pandas as pd
import numpy as np





def average_ranks_with_versions_and_nemenyi(df, alpha = 0.05):
    rank_df = average_ranks_with_versions(df)
    critical_value = nemenyi_significance_test(
        len(df.dataset_id.drop_duplicates()), rank_df, alpha = alpha
    )
    return rank_df, critical_value

def average_ranks_with_versions(df):
    df = df.fillna({"dataset_version": -1})
    final_ranks = (
        df.set_index('algo_name')
        .groupby(['algo_name', 'dataset_id'])[['auc', 'ap']].mean()

        # rank results per dataset
        .groupby("dataset_id")
        .rank(ascending=False)

        # average ranks per algorithm
        .groupby(["algo_name"])
        .mean()
    )
    final_ranks.columns = ["auc_ranks", 'ap_ranks']
    return final_ranks


def average_performance_per_method(df): #USED
    return (
        # average over datasets
        df.groupby(["algo_name", "dataset_id"])[["auc", "ap"]]
        .mean()
        # average over algorithms
        .groupby(['algo_name'])
        .mean()
        .rename(columns=dict(auc="avg_auc", ap="avg_ap"))
    )


def get_avg_performance_dataframe(df): # USED
    return (
        df.groupby(by=["dataset_id"])[['auc', 'ap']]
        .mean()
        .reset_index()
        .rename(columns=dict(auc="avg_auc", ap="avg_ap"))
    )


def get_align_dataframe(performance_df, avg_performance_df):
    align_df = pd.merge(performance_df, avg_performance_df, on="dataset_id")
    align_df["align_auc"] = align_df.apply(lambda r: r.auc - r.avg_auc, axis=1)
    align_df["align_ap"] = align_df.apply(lambda r: r.ap - r.avg_ap, axis=1)
    return align_df


def average_aligned_ranks_with_versions(df): # USED
    df = (
        # average over versions
        df.groupby(by=["algo_name", "dataset_id"])[['auc', 'ap']]
        .mean()
        # missing results
        .fillna(0)
        .reset_index()
        .sort_values("dataset_id")
    )
    # Remove ignore ids
    # df = df[~df["dataset_id"].isin(ignore_ids)]

    # Isolate relevant ids
    # if relevant_ids is not None:
    #     df = df[df["dataset_id"].isin(relevant_ids)]

    avg_performance_df = get_avg_performance_dataframe(df)
    align_df = get_align_dataframe(df, avg_performance_df)

    # Get aligned ranks
    align_df["rank_auc"] = align_df["align_auc"].rank(ascending=False)
    align_df["rank_ap"] = align_df["align_ap"].rank(ascending=False)

    align_rank_df = align_df.groupby(by=["algo_name"]).mean()[
        ["rank_auc", "rank_ap"]
    ]

    return align_rank_df


def win_loss_counts_with_versions(
    df, reference, metric_to_use, draw_threshold, win_over_reference
):
    df = df.fillna({"dataset_version": -1})
    without_versions = df.groupby(["algorithm_instance_name", "dataset_id"])[
        [metric_to_use]
    ].mean()
    return win_loss_counts(
        without_versions, reference, metric_to_use, draw_threshold, win_over_reference
    )


def win_loss_counts(
    df,
    reference="AD-Mercs",
    metric_to_use="auc",
    draw_threshold=0.01,
    win_over_reference=False,
):
    df = pd.pivot_table(
        df, index="dataset_id", columns="algorithm_instance_name", values=metric_to_use
    )
    reference = df[reference]
    difference_df = df.apply(lambda x: x - reference, axis=0)

    def fill_win_loss(value):
        if abs(value) <= draw_threshold:
            return "Draw"
        if not win_over_reference:
            if value < 0:
                return "Win"
            return "Loss"
        else:
            if value < 0:
                return "Loss"
            return "Win"

    win_loss_df = difference_df.applymap(fill_win_loss)
    # get back to column format
    win_loss_df = win_loss_df.unstack().to_frame("win/loss").reset_index()
    win_loss_df = (
        win_loss_df.groupby("algorithm_instance_name")["win/loss"]
        .value_counts()
        .to_frame("counts")
        .reset_index()
    )
    win_loss_df = pd.pivot_table(
        win_loss_df,
        index="algorithm_instance_name",
        columns="win/loss",
        values="counts",
        fill_value=0,
    )
    # make sure all columns are present
    for col in ["Win", "Draw", "Loss"]:
        if col not in win_loss_df.columns:
            win_loss_df[col] = 0
    return win_loss_df

def nemenyi_significance_test(N_datasets, ranking_df: pd.DataFrame, alpha = 0.05):
    ranking_df = ranking_df.copy()
    algorithm_columns = ranking_df.index.names
    ranking_df = ranking_df.reset_index()
    ranking_df["name"] = ranking_df[algorithm_columns].apply(
        lambda row: ",".join(row), axis=1
    )
    lookup_df = ranking_df.set_index("name")
    methods = ranking_df["name"].tolist()
    # alpha = 0.05
    k = ranking_df.shape[0]
    N = N_datasets
    q_a = _get_Nemenyi_values(alpha, k)
    critical_value = q_a * np.sqrt((k * (k + 1)) / (6 * N))
    # differences = 0
    # significant_diffs = pd.DataFrame(0, columns=methods, index=methods)
    # for i, m1 in enumerate(methods):
    #     for j, m2 in enumerate(methods):
    #         if m1 == m2:
    #             continue
    #         else:
    #             m1_rank = lookup_df.loc[m1, "ranks"]
    #             m2_rank = lookup_df.loc[m2, "ranks"]
    #             if abs(m1_rank - m2_rank) > critical_value:
    #                 significant_diffs.iloc[i, j] = 1
    #                 significant_diffs.iloc[j, i] = 1
    #                 differences += 1
    return critical_value#, significant_diffs


def _get_Nemenyi_values(pvalue, models):
    pvalue = float(pvalue)
    models = int(models)

    CRITICAL_VALUES = [
        # p   0.01   0.05   0.10  Models
        [2.576, 1.960, 1.645],  # 2
        [2.913, 2.344, 2.052],  # 3
        [3.113, 2.569, 2.291],  # 4
        [3.255, 2.728, 2.460],  # 5
        [3.364, 2.850, 2.589],  # 6
        [3.452, 2.948, 2.693],  # 7
        [3.526, 3.031, 2.780],  # 8
        [3.590, 3.102, 2.855],  # 9
        [3.646, 3.164, 2.920],  # 10
        [3.696, 3.219, 2.978],  # 11
        [3.741, 3.268, 3.030],  # 12
        [3.781, 3.313, 3.077],  # 13
        [3.818, 3.354, 3.120],  # 14
        [3.853, 3.391, 3.159],  # 15
        [3.884, 3.426, 3.196],  # 16
        [3.914, 3.458, 3.230],  # 17
        [3.941, 3.489, 3.261],  # 18
        [3.967, 3.517, 3.291],  # 19
        [3.992, 3.544, 3.319],  # 20
        [4.015, 3.569, 3.346],  # 21
        [4.037, 3.593, 3.371],  # 22
        [4.057, 3.616, 3.394],  # 23
        [4.077, 3.637, 3.417],  # 24
        [4.096, 3.658, 3.439],  # 25
        [4.114, 3.678, 3.459],  # 26
        [4.132, 3.696, 3.479],  # 27
        [4.148, 3.714, 3.498],  # 28
        [4.164, 3.732, 3.516],  # 29
        [4.179, 3.749, 3.533],  # 30
        [4.194, 3.765, 3.550],  # 31
        [4.208, 3.780, 3.567],  # 32
        [4.222, 3.795, 3.582],  # 33
        [4.236, 3.810, 3.597],  # 34
        [4.249, 3.824, 3.612],  # 35
        [4.261, 3.837, 3.626],  # 36
        [4.273, 3.850, 3.640],  # 37
        [4.285, 3.863, 3.653],  # 38
        [4.296, 3.876, 3.666],  # 39
        [4.307, 3.888, 3.679],  # 40
        [4.318, 3.899, 3.691],  # 41
        [4.329, 3.911, 3.703],  # 42
        [4.339, 3.922, 3.714],  # 43
        [4.349, 3.933, 3.726],  # 44
        [4.359, 3.943, 3.737],  # 45
        [4.368, 3.954, 3.747],  # 46
        [4.378, 3.964, 3.758],  # 47
        [4.387, 3.973, 3.768],  # 48
        [4.395, 3.983, 3.778],  # 49
        [4.404, 3.992, 3.788],  # 50
    ]

    # get the critical value
    if pvalue == 0.01:
        col_idx = 0
    elif pvalue == 0.05:
        col_idx = 1
    elif pvalue == 0.10:
        col_idx = 2
    else:
        raise ValueError("p-value must be one of 0.01, 0.05, or 0.10")

    # models
    if not (2 <= models and models <= 50):
        raise ValueError("number of models must be in range [2, 50]")
    else:
        row_idx = models - 2

    q_a = CRITICAL_VALUES[row_idx][col_idx]

    # ciritical difference
    # NOT IMPLEMENTED
    return q_a