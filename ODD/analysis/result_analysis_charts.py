import pandas as pd
from ODD.analysis.dataset_selection import apply_contamination_filter
import altair as alt

def compare_evaluation_settings_chart(processed_path, evaluation_settings, contamination_filter, algorithm, metric_to_use='full_auc'):
    result_df = pd.concat(
        [pd.read_csv(processed_path / evaluation_setting / f"{algorithm}.csv") for evaluation_setting in
         evaluation_settings], keys=evaluation_settings, names=['evaluation_setting'])
    result_df = apply_contamination_filter(contamination_filter, result_df)
    if metric_to_use is not None:
        result_df.loc[result_df.auc.isna(), 'auc'] = result_df.loc[result_df.auc.isna(), metric_to_use]
    return alt.Chart(result_df.reset_index()).mark_point().encode(
        x='dataset_id:N',
        y='auc:Q',
        color='evaluation_setting:N'
    ).properties(title=algorithm)




def compare_evaluation_settings_chart_default_as_reference_relative(processed_path, evaluation_settings, contamination_filter, algorithm, default_reference,
                                          tuned_reference, metric_to_use='full_auc'):
    # read the data
    result_df = pd.concat(
        [pd.read_csv(processed_path / evaluation_setting / f"{algorithm}.csv") for evaluation_setting in
         evaluation_settings], keys=evaluation_settings, names=['evaluation_setting'], ignore_index = True)
    result_df = apply_contamination_filter(contamination_filter, result_df)
    result_df = result_df.set_index(['dataset_id', 'anomaly_fraction'], append=True).droplevel(1)
    if metric_to_use is not None:
        result_df.loc[result_df.auc.isna(), 'auc'] = result_df.loc[result_df.auc.isna(), metric_to_use]

    # data transformation for relative plotting and sorting
    default_performance = result_df.loc[default_reference].auc.to_frame('default_performance')
    tuned_performance = result_df.loc[tuned_reference].auc.to_frame('tuned_performance')
    result_df = default_performance.join(tuned_performance).join(result_df).reset_index()
    result_df['relative_performance'] = result_df.auc / result_df.default_performance
    result_df['relative_tuned_performance'] = result_df.tuned_performance / result_df.default_performance

    result_df = apply_contamination_filter(contamination_filter, result_df)

    # plot the results
    return alt.Chart(result_df.reset_index()).mark_point().encode(
        x=alt.X('dataset_id:N', sort=alt.EncodingSortField(field='relative_tuned_performance')),
        y='relative_performance:Q',
        color='evaluation_setting:N'
    ).properties(title=algorithm)

def compare_evaluation_settings_chart_default_as_reference_absolute(processed_path, evaluation_settings, contamination_filter, algorithm, default_reference,
                                          tuned_reference, metric_to_use='full_auc', show_only_tuned = True):
    # read the data
    result_df = pd.concat(
        [pd.read_csv(processed_path / evaluation_setting / f"{algorithm}.csv") for evaluation_setting in
         evaluation_settings], keys=evaluation_settings, names=['evaluation_setting']).droplevel(1)
    # apply the contamination filter
    result_df = apply_contamination_filter(contamination_filter, result_df)

    # only display datasets where tuned has a result
    if show_only_tuned:
        datasets_tuned_result = result_df[result_df.index.get_level_values(0).str.startswith('tuned_performance')].dataset_id.unique()
        result_df = result_df[result_df.dataset_id.isin(datasets_tuned_result)]

    result_df = result_df.set_index(['dataset_id', 'anomaly_fraction'], append=True)
    if metric_to_use is not None:
        result_df.loc[result_df.auc.isna(), 'auc'] = result_df.loc[result_df.auc.isna(), metric_to_use]

    # data transformation for relative plotting and sorting
    default_performance = result_df.loc[default_reference].auc.to_frame('default_performance')
    tuned_performance = result_df.loc[tuned_reference].auc.to_frame('tuned_performance')
    plot_df = default_performance.join(tuned_performance).join(result_df).reset_index()
    plot_df['relative_performance'] = plot_df.auc - plot_df.default_performance
    plot_df['relative_tuned_performance'] = plot_df.tuned_performance - plot_df.default_performance



    # plot the results
    return alt.Chart(plot_df.reset_index()).mark_point().encode(
        x=alt.X('dataset_id:N', sort=alt.EncodingSortField(field='relative_tuned_performance')),
        y='relative_performance:Q',
        color='evaluation_setting:N'
    ).properties(title=algorithm), result_df