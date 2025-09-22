import inspect
import pandas as pd

from experiments_2022 import DATASETS_PATH
from experiments_2022.datasets import utils
from experiments_2022.zone_level_analysis.cleaning import clean_df


def load_zones(
    dataset,
    project,
    name,
    clean_data=False,
    resample_rule="1h",
    resample_statistic="Mean",
    resample_data=False,
):
    """
    Load zonal data for a particular dataset, building, filter

    Raises an error if the dataset is unavailable.

    Parameters
    ----------
    dataset : str
    project : str
    name : str
    clean_data : bool
    resample_rule : str
    resample_statistic : str
    resample_data : bool ### to delete, will be breaking change

    Returns
    -------
    pd.DataFrame
        df with index as timestamps and columns as zones.

    Notes
    -----
    Dependent variables are calculated from core variables

    Examples
    --------
    `datasets.load_zones("2022", "ALUMNI", "zone-temps")
    """
    if name in utils.VARIABLE_DEPENDENCIES:
        try:
            data_dict = {}
            variables = utils.VARIABLE_DEPENDENCIES[name]
            if ("weather-oat" in variables) or ("weather-rh" in variables):
                resample_data = True
            for this_var in variables:
                if this_var == "weather-oat":
                    this_df = load_weather(dataset)["temperature"].to_frame()
                    this_df.columns = ["temperature"]
                elif this_var == "weather-rh":
                    this_df = load_weather(dataset)["RH"].to_frame()
                    this_df.columns = ["RH"]
                else:
                    this_df = load_zones(
                        dataset,
                        project,
                        this_var,
                        clean_data=clean_data,
                        resample_rule=resample_rule,
                        resample_statistic=resample_statistic,
                    )
                data_dict[this_var] = this_df
            fn = utils.FUNCTIONS[name]
            if "project" in inspect.signature(fn).parameters:
                df = fn(project, data_dict)
            else:
                df = fn(data_dict)
            return df
        except Exception:
            print(f"Could not load {name}")
    else:
        filename = DATASETS_PATH / dataset / f"{project}_{name}.csv"
        try:
            this_df = pd.read_csv(filename, parse_dates=True, index_col=0)
            if clean_data:
                this_df = clean_df(
                    this_df,
                    name,
                    only_business_hours=False,
                    no_weekends=False,
                    SI_units=False,
                )
            if (resample_rule is not None) and isinstance(
                this_df.index, pd.DatetimeIndex
            ):
                if resample_statistic == "Sum":
                    this_df = this_df.resample(resample_rule).sum()
                else:
                    this_df = this_df.resample(resample_rule).mean()
            return this_df
        except FileNotFoundError:
            print(f"Could not find {filename}")


def pull_from_dataset(
    dataset,
    projects,
    this_var,
    clean_data=False,
    resample_rule="1h",
    resample_statistic="Mean",
    resample_data=False,
):
    """
    Helper function to pull from datasets for several buildings

    Parameters
    ----------
    dataset : str
    projects : list
    this_var : str
    clean_data : bool
    resample_rule : str
    resample_statistic : str
    resample_data : bool ### to delete, will be breaking change

    Returns
    -------
    dfs in dict form
    """
    dfs = {}
    for project in projects:
        dfs[project] = load_zones(
            dataset=dataset,
            project=project,
            name=this_var,
            clean_data=clean_data,
            resample_rule=resample_rule,
            resample_statistic=resample_statistic,
            resample_data=resample_data,
        )
    return dfs


def load_building(dataset, utility):
    """
    Loads building-level data for a particular dataset, utility

    Raises an error if the dataset is unavailable.

    Parameters
    ----------
    dataset : str
    utility : str, "E" (electricity) or "C" (cooling)

    Returns
    -------
    pd.DataFrame
        df with index as timestamps and columns as buildings.

    Examples
    --------
    `datasets.load_building("2022", "C")`
    """
    if utility not in ["E", "C", "H"]:
        raise ValueError(f"utility: expected one of E,C,H but got {utility}")
    filename = DATASETS_PATH / dataset / f"{utility}.csv"
    try:
        return pd.read_csv(filename, parse_dates=True, index_col=0)
    except FileNotFoundError:
        print(f"Could not find {filename}")


def load_weather(dataset):
    """
    Loads weather data for a particular dataset

    Raises an error if the dataset is unavailable.

    Parameters
    ----------
    dataset : str

    Returns
    -------
    pd.DataFrame
        df with index as timestamps and columns as weather variables.

    Examples
    --------
    `datasets.load_weather("2022")`
    """
    filename = DATASETS_PATH / dataset / "weather.csv"
    try:
        df = pd.read_csv(filename, parse_dates=True, index_col=0)
        df = df[~df.index.duplicated(keep="first")]  # ensure unique index
        return df
    except FileNotFoundError:
        print(f"Could not find {filename}")
