import itertools
import warnings
from collections import defaultdict
from pathlib import Path
import random
import pandas as pd
import toml
from affe import DTAIExperimenterProcessExecutor
from affe.execs import DTAIExperimenterShellExecutor
from affe.execs.CompositeExecutor import DaskExecutor, JoblibExecutor
from affe.io import load_object
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


from ODD.data.dataset import DataSource
from .BenchmarkInterface import BenchmarkInterface

def get_results_from_config(config_path):
    # parse the config
    all_flows, flows = get_flows_from_config(config_path)
    # get the results
    result_df = get_results_from_flows(all_flows)
    return result_df

def get_results_from_result_path(result_path):
    flow_dir = result_path/ 'flow'
    # read all flows
    paths = list(flow_dir.iterdir())
    all_flows = [load_object(path) for path in tqdm(paths, desc = 'reading flows', leave = False)]
    result_df = get_results_from_flows(all_flows)
    return result_df

def execute_experiment(config_path, result_path, dask_scheduler=None, dask_batch_size = None, local_jobs=4, progress=True, shuffle = False):
    if result_path.exists():
        print('results already exist! returning stored results!')
        return pd.read_pickle(result_path)

    # parse the config
    all_flows, flows = get_flows_from_config(config_path)
    print(f"remaining_flows = {len(flows)}")

    if shuffle:
        random.shuffle(flows)
    # execute the flows
    if len(flows) == 0:
        pass
    elif dask_scheduler is not None:
        DaskExecutor(
            flows, DTAIExperimenterShellExecutor, dask_scheduler, show_progress=progress, notebook = True, batch_size=dask_batch_size
        ).execute()
    elif local_jobs > 1:
        JoblibExecutor(flows, executor=DTAIExperimenterProcessExecutor, n_jobs=local_jobs).execute()
    else:
        # run locally on a single thread (often used for debugging!)
        for flow in tqdm(flows):
            try:
                flow.execute()
            except Exception as e:
                print(f"flow {flow.STR} on {flow.data_config} FAILED, skipping...")
                raise e

    # get the results
    result_df = get_results_from_flows(all_flows)
    if result_path is not None:
        result_df.to_pickle(result_path)
    return result_df


def get_results_from_flows(flows):
    records = []
    missing = 0
    for flow in tqdm(flows, desc= 'getting results'):
        record = flow.get_results_as_record()
        if record is not None:
            records.append(record)
        else:
            missing += 1
    assert len(records) > 0, "you are trying to create a dataframe from nothing!"
    if missing > 0:
        warnings.warn(f"there are {missing} missing results!")
    df = pd.DataFrame.from_records(records, index="flow_id")
    return df


def get_flows_from_config(config_path):
    with open(config_path, "r") as f:
        config = toml.load(f)

    # parse the main config
    main_config = config.pop('main')
    experiment_title = main_config.pop('title')
    root_levels_up = main_config.pop('root_levels_up', None)
    out_path = main_config.pop('out_path', None)
    assert root_levels_up or out_path, 'you should specify root_levels_up or the out_path'
    timeout_s = main_config.pop('timeout_s', 60)
    if len(main_config) > 0:
        warnings.warn(f"Unknown parameters in main section: {main_config}")

    # parse the data config
    data_config = config.pop('data')
    data_sources = list(DataSource.parse_source_configuration(data_config))

    all_grid_points = []
    # parse the algorithm configs (all remaining)
    for algo, algo_dict in config.items():
        for algo_exp_name, parameter_grid in algo_dict.items():
            # the ParameterGrid class cannot handle parameters with a single value
            # it has to be a list with a single item
            new_dict = {key: value if isinstance(value, list) else [value] for key, value in parameter_grid.items()}
            all_grid_points.extend((algo, algo_exp_name, config) for config in ParameterGrid(new_dict))

    flows = []
    all_flows = []
    existing = None
    for flow_id, ((algo, algo_exp_name, config), data_source) in tqdm(
            enumerate(itertools.product(all_grid_points, data_sources)), total=len(all_grid_points) * len(data_sources),
            desc='Making flows'):
        try:
            flow = get_flow(algo, algo_exp_name, config, data_source, experiment_title, timeout_s, flow_id, out_path,
                     root_levels_up)
            if existing is None:
                result_path = flow.result_path.parent
                existing = set(path.name for path in result_path.iterdir())
                print(f'found {len(existing)} results')
            all_flows.append(flow)
            if flow.result_path.name not in existing:
                flows.append(flow)
        except Exception as e:
            warnings.warn(f"Failed to make a flow: {e}")
            raise e
    return all_flows, flows


def get_flow(algo, algo_exp_name, algo_config, data_source, experiment_title, timeout_s, flow_id, out_path, root_levels_up):
    flow_identifier = f"{algo}.{algo_exp_name}-{data_source.name}-{flow_id}"
    return BenchmarkInterface(algo, experiment_title, flow_identifier, out_path, root_levels_up, data_source, algo_config, timeout_s, algorithm_instance_name = algo_exp_name).unwrap()

class ConfigBuilder:
    def __init__(self, title, root_levels_up=None, timeout_s = 600, out_dp = None):
        self.dict = defaultdict(dict)
        self.dict['main'] = dict(
            title = title,
            root_levels_up = root_levels_up,
            timeout_s = timeout_s,
            out_dp = out_dp
        )

    def add_dataset(self, dataset_name, **options):
        self.dict['data'][dataset_name] = options
        return self

    def add_algorithm(self, algo_name, algorithm_instance_name, **algo_config):
        # convert value to list if necessary
        algo_config = {key:value if isinstance(value, list) else list(value) for key,value in algo_config.items()}
        self.dict[algo_name][algorithm_instance_name] = algo_config
        return self

    def to_file(self, path):
        with open(path, 'w') as f:
            toml.dump(self.dict, f, encoder=toml.TomlNumpyEncoder())
        return self