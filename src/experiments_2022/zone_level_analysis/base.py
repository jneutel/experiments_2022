import copy
import pandas as pd
import functools


MW_PER_TON = 3.5168528421 / 1000
WH_PER_BTU = 0.293071
M2_PER_SF = 0.092903

ALL_BUILDINGS = [
    "OFF-1",
    "OFF-2",
    "OFF-3",
    "OFF-4",
    "OFF-5",
    "OFF-6",
    "OFF-7",
    "LAB-1",
    "LAB-2",
    "LAB-3",
]

VAV_BUILDINGS = [
    "OFF-1",
    "OFF-2",
    "OFF-3",
    "OFF-4",
    "OFF-5",
    "OFF-6",
    "OFF-7",
]

FCU_BUILDINGS = [
    "LAB-1",
    "LAB-2",
    "LAB-3",
]

EXPERIMENTS_BUILDINGS = {
    "2021": [
        "OFF-1",
        "OFF-3",
        "OFF-4",
        "OFF-6",
        "LAB-1",
        "LAB-3",
    ],
    "2022": [
        "OFF-1",
        "OFF-2",
        "OFF-3",
        "OFF-4",
        "OFF-5",
        "OFF-6",
        "OFF-7",
        "LAB-1",
        "LAB-2",
        "LAB-3",
    ],
    "2024": [
        "OFF-2",
        "OFF-3",
        "OFF-4",
        "OFF-5",
        "OFF-6",
        "OFF-7",
    ],
}


def input_to_dict(input, projects):
    """
    Helper function to check whether input is of dict type, and if not converts it

    Parameters
    ----------
    input : many
        an input to another function, can be int, str etc
    projects : list
        list of buildings to convert input_dict to, with buildings as key

    Returns
    -------
    dict ready to be input to function
    """
    if isinstance(input, dict):
        return input
    return_dict = {}
    for project in projects:
        return_dict[project] = input
    return return_dict


def trim_to_common_elements(dfs, clean_cols=True, clean_idx=True):
    """
    Trim a list of dataframe to only include common elements and indexes

    Parameters
    ----------
    dfs : list of pd.DataFrames
    clean_cols : bool
    clean_idx :  bool

    Returns
    -------
    list of pandas.DataFrames
    """
    for i in range(len(dfs)):
        dfs[i] = copy.deepcopy(dfs[i])
    if len(dfs) == 1:
        return dfs
    if clean_cols:
        cols = functools.reduce(
            lambda left, right: left.intersection(right),
            [dfs[i].columns for i in range(len(dfs))],
        )
        dfs = [df.loc[:, cols] for df in dfs]
    if clean_idx:
        idx = functools.reduce(
            lambda left, right: left.intersection(right),
            [dfs[i].index for i in range(len(dfs))],
        )
        dfs = [df.loc[idx, :] for df in dfs]
    return dfs


def run_passive_test(df, this_test, axis=0):
    if this_test == "Sum":
        return df.sum(axis=axis)
    if this_test == "Mean":
        return df.mean(axis=axis)
    if this_test == "Median":
        return df.median(axis=axis)
    if this_test == "Min":
        return df.min(axis=axis)
    if this_test == "Max":
        return df.max(axis=axis)
    if this_test == "Std":
        return df.std(axis=axis)


def run_passive_test_on_dfs(
    dfs,
    this_test="Mean",
    col_name=None,
    results=None,
):
    """
    Helper function to run passive test and return dict object ready for plotting

    Parameters
    ----------
    dfs : dict
        building as key and df as value
    this_test : str
        statistic to summarize over time
        Options include "Sum", "Mean", "Median", "Min", "Max", "Std"
        default Mean
    col_name : str
        default None, name of test is used
    results : dict
        function accepts its own output so that results dict's can be built up

    Returns
    -------
    dict of pd.DataFrame()'s
    """
    projects = list(dfs.keys())
    # init
    if results is None:
        results = {}
        for project in projects:
            results[project] = pd.DataFrame()
    # col name
    if col_name is None:
        col_name = this_test
    for project in projects:
        df = dfs[project]
        # run test
        this_result = run_passive_test(df, this_test, axis=0).to_frame()
        this_result.sort_index(inplace=True)
        this_result.columns = [col_name]
        results[project] = pd.concat([results[project], this_result], axis=1)
    return results


def make_common_index(dicts, print_message=False):
    """
    Helper function to make sure dicts have common indexes

    Parameters
    ----------
    dicts : list
        list of dicts, which are dict of pd.DataFrame()'s
    print_message : bool
        whether or not to print out which zones are lost by the operation

    Returns
    -------
    list
    """
    for project in dicts[0]:
        dfs = []
        for this_dict in dicts:
            dfs.append(this_dict[project])
        newdfs = trim_to_common_elements(dfs, clean_cols=False, clean_idx=True)
        if print_message:
            for i in range(len(dfs)):
                lost_boxes = list(set(list(dfs[i].index)) - set(list(newdfs[i].index)))
                if len(lost_boxes) > 0:
                    print(
                        f"By making common we have lost these boxes in {project}: {lost_boxes}"
                    )
        i = 0
        for this_dict in dicts:
            this_dict[project] = newdfs[i]
            i += 1
    return dicts


def combine_dicts(dicts):
    """
    Helper function to combine several dicts into one dict with many columns

    Parameters
    ----------
    dicts : list
        list of results dicts

    Returns
    -------
    Combined dict
    """
    dicts = make_common_index(dicts, print_message=False)
    new_dict = {}
    for this_dict in dicts:
        for this_project in this_dict:
            this_df = this_dict[this_project]
            if this_project not in new_dict:
                new_dict[this_project] = pd.DataFrame(index=this_df.index)
            new_dict[this_project][list(this_df.columns)] = this_df
    return new_dict


def calculate_airflow_weighted_average(this_dict, airflow_dict, project="dummy"):
    """
    Helper function to calculate airflow weighted average

    Parameters
    ----------
    this_dict : dict or pd.DataFrame
        dict with building as key and pd.DataFrame as value
        compatible with single df with "dummy" key
    airflow_dict : dict
        dict with building as key and pd.DataFrame as value
        should have some keys, columns, index as this_dict
        compatible with single df with "dummy" key
    project : str
        optional, to help make compatible with single df

    Returns
    -------
    weighted_average_dict
    """
    this_dict = input_to_dict(this_dict, [project])
    airflow_dict = input_to_dict(airflow_dict, [project])

    projects = list(this_dict.keys())
    weighted_average_dict = {}

    for project in projects:
        cols = list(
            set(this_dict[project].columns).intersection(
                set(airflow_dict[project].columns)
            )
        )
        idx = list(
            set(this_dict[project].index).intersection(set(airflow_dict[project].index))
        )
        idx.sort()
        weighted_average_dict[project] = (
            (
                (
                    this_dict[project].loc[idx, cols]
                    * airflow_dict[project].loc[idx, cols]
                ).sum(axis=1)
            )
            / (airflow_dict[project].loc[idx, cols].sum(axis=1))
        ).to_frame()
        weighted_average_dict[project].columns = [project]
    return weighted_average_dict
