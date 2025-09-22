import datetime as dt
import numpy as np
import pandas as pd
from experiments_2022.zone_level_analysis.regression_functions import (
    get_2021_2022_binary_df,
)
from experiments_2022.zone_level_analysis.base import input_to_dict


# in original units
LOWER_LIMIT = {
    "weather-oat": -10,
    "building-cooling": 0.001,  # exclude 0
    "building-heating": 0.001,  # exclude 0
    "building-electricity": 0.001,  # exclude 0
    "ahu-airflow": 0.001,  # exclude 0
    "ahu-dat": 40,
    "ahu-mat": 40,
    "zone-tloads": -101,
    "zone-airflow": 0.001,  # exclude 0
    "zone-airflowsp": 0.001,  # exclude 0
    "zone-temps": 60,
    "zone-coolsp": 60,
    "zone-heatsp": 50,
    "zone-dat": 40,
    "zone-datsp": 40,
    "zone-zonesp": 50,
    "zone-cool_offset": -8,
    "zone-heat_offset": -8,
    "zone-deviation_coolsp": -8,
    "zone-deviation_heatsp": -8,
    "zone-deadband_top": 60,
    "zone-deadband_bottom": 50,
    "zone-local_offset": -15,
    "zone-simple_cooling_requests": -0.1,
    "zone-total_cooling": 0,
}

# in original units
# included if <= upper_limit
UPPER_LIMIT = {
    "weather-oat": 130,
    "building-cooling": 10000,  # tons/h
    "building-heating": 10000,  # kbtu/h
    "building-electricity": 10000,  # kw
    "ahu-airflow": 10000000,  # inf
    "ahu-dat": 85,
    "ahu-mat": 85,
    "zone-tloads": 101,
    "zone-airflow": 100000,
    "zone-airflowsp": 100000,
    "zone-temps": 85,
    "zone-coolsp": 85,
    "zone-heatsp": 80,
    "zone-dat": 120,
    "zone-datsp": 120,
    "zone-zonesp": 85,
    "zone-cool_offset": 8,
    "zone-heat_offset": 8,
    "zone-deviation_coolsp": 8,
    "zone-deviation_heatsp": 8,
    "zone-deadband_top": 85,
    "zone-deadband_bottom": 80,
    "zone-local_offset": 15,
    "zone-simple_cooling_requests": 1.1,
    "zone-total_cooling": 1000,
}

OCCUPANCY_TIME_RANGE = (dt.time(9, 0), dt.time(18, 0))

NOT_ZONES = [
    "CHW Plant",
    "LEV_",
    "HEV_",
    "LSV_",
    "VVE_",
    "SVLV",
    "Room Points",
    "TF_",
    "EF",
    "SF",
    "AH",
    "AC",
]

NOT_VAVS = [
    "CHW Plant",
    "LEV_",
    "HEV_",
    "LSV_",
    "VVE_",
    "SVLV",
    "Room Points",
    "TF_",
    "EF",
    "SF",
    "AH",
    "AC",
    "Fcu",
    "FC",
]

NOT_AHUS = [
    "CHW Plant",
    "LEV_",
    "HEV_",
    "LSV_",
    "VVE_",
    "SVLV",
    "Room Points",
    "TF_",
    "EF",
    "VAV",
    "CAV",
    "RH",
    "VVS_",
    "Fcu",
    "FC",
]

ECON_REMOVE_STR = [
    " Economizer Damper Command",
    " OA Damper Cmd",
    " Econ Damper Cmd",
    " Econ Damper Pos",
    " OA Damper Mode",
]

NO_WEEKENDS = {
    "OFF-1": False,
    "OFF-2": False,
    "OFF-3": False,
    "OFF-4": False,
    "OFF-5": False,
    "OFF-6": False,
    "OFF-7": False,
    "OFF-8": False,
    "LAB-1": False,
    "LAB-2": False,
    "LAB-3": False,
}


TEMP_VARIABLES = [
    "weather-oat",
    "ahu-dat",
    "ahu-datsp",
    "ahu-rat",
    "ahu-mat",
    "ahu-oat",
    "zone-dat",
    "zone-datsp",
    "zone-coolsp",
    "zone-heatsp",
    "zone-temps",
    "zone-zonesp",
    "zone-deadband_top",
    "zone-deadband_bottom",
]


DELTA_TEMP_VARIABLES = [
    "zone-cool_offset",
    "zone-heat_offset",
    "zone-local_offset",
    "zone-deviation_coolsp",
    "zone-deviation_heatsp",
]


AIRFLOW_VARIABLES = [
    "zone-airflow",
    "zone-airflowsp",
    "ahu-airflow",
]


PRESSURE_VARIABLES = [
    "ahu-dap",
    "ahu-dapsp",
]


def create_sp_filter(sp_schedules, sps=[74], reverse_filter=False):
    if isinstance(sp_schedules, dict):
        return_df = False
        projects = list(sp_schedules.keys())
    else:
        return_df = True
        projects = ["project"]
        sp_schedules = input_to_dict(sp_schedules, projects)

    masks = {}
    for project in projects:
        if reverse_filter:
            mask = ~sp_schedules[project].isin(sps)
        else:
            mask = sp_schedules[project].isin(sps)

        # Convert boolean mask -> 1 / NaN
        masks[project] = mask.astype(int).where(mask)

    if return_df:
        return masks["project"]
    return masks


def clean_projects(
    df,
    start_date=None,
    end_date=None,
    only_business_hours=False,
    no_weekends=False,
    hourly_filter=None,
    lb_all=0.0,
    ub_all=None,
    lb_dict=None,
    up_dict=None,
    scrub_dates=None,
):
    """
    Input is now a df, and we can clean each column separatley

    Parameters
    ----------
    df : pd.DataFrame
        index is time, columns are buildings
    start_date : pd.Timestamp() or dict
        month-day-year
        if pd.Timestamp(), same date applies to all projects
        if dict, then can be building specific
        default = None, look at earliest date in dataset
    end_date : ppd.Timestamp() or dict
        month-day-year
        if pd.Timestamp(), same date applies to all projects
        if dict, then can be building specific
        default = None, look at latest date in dataset
    only_business_hours : bool or dict
        if bool, then same logic applies for all builings
        if dict, then can be building specific
        True means only look at 9 AM to 6 PM (OCCUPANCY_TIME_RANGE)
        default = True
    no_weekends : bool or dict
        if bool, then same logic applies for all builings
        if dict, then can be building specific
        default = True, remove weekends for all buildings
    hourly_filter : pd.Series() or dict of pd.Series()
        allows user to filter by specific hours
        value in series is unimportant, only index used to filter
        if pd.Series(), same filter applies to all projects
        if dict of pd.Series(), building specific filters
        default = None, no filter
    lb_all : float
        lower bound to apply to all buildings
        default 0
    ub_all : float
        upper bound to apply to all buildings
        default None
    lb_dict : dict
        lower bound building specific
        {project : int}
    up_dict : dict
        upper bound building specific
        {project : int}
    scrub_dates : dict
        {project : list(list)} or
        {project : list(tuple)}
        for each project, scrub out specific date ranges

    Notes
    -----
    All bounds not inclusive
    """
    df = clean_by_column(
        df=df,
        start_date=start_date,
        end_date=end_date,
        only_business_hours=only_business_hours,
        no_weekends=no_weekends,
        hourly_filter=hourly_filter,
    )
    if lb_all is not None:
        df[df < lb_all] = np.nan
    if ub_all is not None:
        df[df > ub_all] = np.nan
    if lb_dict is not None:
        for project in lb_dict:
            df[project][df[project] < lb_dict[project]] = np.nan
    if up_dict is not None:
        for project in up_dict:
            df[project][df[project] > up_dict[project]] = np.nan
    if scrub_dates is not None:
        for project in scrub_dates:
            for date_tuples in scrub_dates[project]:
                df.loc[date_tuples[0] : date_tuples[1], project] = np.nan
    return df


def clean_columns(df, this_var, remove_FCUs=False):
    """
    Cleans the columns of a df

    Parameters
    ----------
    df : pd.DataFrame
        zonal df to clean
    this_var : str
        variable associated with df
        if None, we replace with "zone-dummy"
    remove_FCUs : bool
        if True, FCUs are removed from zonal data
        default False

    Returns
    -------
    Cleaned df
    """
    # dont clean these
    if this_var in [
        "weather-oat",
        "weather-rh",
        "zone-map",
        "building-cooling",
        "building-electricity",
        "building-heating",
    ]:
        return df
    # edge case - econ flag can have strange names
    if this_var == "ahu-econ_cmd":
        cols = list(df.columns)
        for this_str in ECON_REMOVE_STR:
            cols = [s.replace(this_str, "") for s in cols]
        df.columns = cols
    # delete string columns
    del_cols = []
    for col in df:
        try:
            df[col] + 1
        except Exception:
            del_cols.append(col)
    df.drop(del_cols, axis=1, inplace=True)
    # delete AHU cols or zones
    if "zone" in this_var:
        if remove_FCUs:
            prefixes = NOT_VAVS  # delete non-VAVs
        else:
            prefixes = NOT_ZONES  # delete non-zones
    else:
        prefixes = NOT_AHUS  # delete non-AHUS
    del_cols = []
    for prefix in prefixes:
        ext = list(df.columns[df.columns.str.contains(prefix)])
        del_cols.extend(ext)
    df.drop(del_cols, axis=1, inplace=True)
    return df


def clean_df(
    df,
    this_var=None,
    start_date=None,
    end_date=None,
    only_business_hours=True,
    no_weekends=True,
    hourly_filter=None,
    hourly_filter_reverse=False,
    df_filter=None,
    resample_rule=None,
    resample_statistic="Mean",
    remove_FCUs=False,
    SI_units=False,
):
    """
    Cleans data

    Parameters
    ----------
    df : pd.DataFrame
        df to clean
    this_var : str
        variable associated with df
        None means no variable specific cleaning done
        use "zone-dummy" or "ahu-dummy" if you want to only clean cols
    start_date : pd.Timestamp()
        e.g. %m-%d-%y
    end_date : pd.Timestamp()
        e.g. %m-%d-%y
    only_business_hours : bool
        True means only look at 9 AM to 6 PM (OCCUPANCY_TIME_RANGE)
        default = True
    no_weekends : bool
        True means only look at M-F
        default = True
    hourly_filter : pd.Series
        hours to filter data by
        note: value in series is unimportant, only index used
        default = None
    hourly_filter_reverse : bool
        True, then scrub out the idxs out hourly filter
        False, then only have the hourly filter
        default False
    df_filter : pd.DatFrame
        way to filter by another df
        cols and index should match df
        values should either be NAN or 1
    resample_rule : str
        allows user to re-index a df into different time steps
        default None, another option is "1h"
    resample_statistic : str
        statistic used for resampling
        default "Mean", can also do "Sum"
    remove_FCUs : bool
        if True, FCUs are removed from zonal data
    SI_units : bool
        whether to convert to SI units
        currently implemented for airflow and temp variables
        converts F to C
        converts cfm to m3/h

    Returns
    -------
    None if df is empty
    Else, cleaned df
    """
    if df is None:
        return pd.DataFrame()
    # handle start and end date
    if start_date is None:
        start_date = list(df.index)[0]
    if end_date is None:
        end_date = list(df.index)[-1]
    df = df.loc[start_date:end_date, :]
    # resample
    if resample_rule is not None:
        if resample_statistic == "Sum":
            df = df.resample(resample_rule).sum()
        else:
            # default to Mean
            df = df.resample(resample_rule).mean()
    # clean columns
    if this_var is not None:
        df = clean_columns(df, this_var, remove_FCUs)
    if len(df.index) == 0 or len(df.columns) == 0:
        return pd.DataFrame()
    # handle business hours & weekends
    if only_business_hours:
        df = df.between_time(OCCUPANCY_TIME_RANGE[0], OCCUPANCY_TIME_RANGE[1])
    if no_weekends:
        df[df.index.dayofweek >= 5] = np.nan  # saturday (5), sunday (6)
    if df_filter is not None:
        df_filter = df_filter.loc[df.index, df.columns]
    # apply hourly filter
    if hourly_filter is not None:
        common = df.index.intersection(hourly_filter.index, sort=True)
        if hourly_filter_reverse:
            df.loc[common, :] = np.nan
        else:
            df = df.loc[common, :]
    elif df_filter is not None:
        df = df * df_filter
    # look at only time steps of interest
    df = df.dropna(axis=0, how="all")
    # handle nonsensicle data
    if (this_var is not None) and (this_var in LOWER_LIMIT):
        df = df.where(df >= LOWER_LIMIT[this_var], np.nan)
    if (this_var is not None) and (this_var in UPPER_LIMIT):
        df = df.where(df <= UPPER_LIMIT[this_var], np.nan)
    # handle SI units
    if (this_var is not None) and (SI_units):
        if this_var in TEMP_VARIABLES:
            df = (df - 32) * (5 / 9)  # F to C
        elif this_var in DELTA_TEMP_VARIABLES:
            df = df * (5 / 9)  # delta F to delta C
        elif this_var in AIRFLOW_VARIABLES:
            df = df * 1.699  # cfm to m3/h
        elif this_var in PRESSURE_VARIABLES:
            df = df * 249.0889  # pascal
    return df


def clean_dfs(
    dfs,
    this_var=None,
    start_date=None,
    end_date=None,
    only_business_hours=True,
    no_weekends=True,
    hourly_filter=None,
    hourly_filter_reverse=False,
    df_filter=None,
    resample_rule=None,
    resample_statistic="Mean",
    remove_FCUs=False,
    SI_units=False,
):
    """
    Cleans data frames

    Parameters
    ----------
    dfs : dict
        building as key and df as value
    this_var : str
        name of variable
        if None, variable specific cleaning not performed
    start_date : pd.Timestamp() or dict
        month-day-year
        if pd.Timestamp(), same date applies to all projects
        if dict, then can be building specific
        default = None, look at earliest date in dataset
    end_date : pd.Timestamp() or dict
        month-day-year
        if pd.Timestamp(), same date applies to all projects
        if dict, then can be building specific
        default = None, look at latest date in dataset
    only_business_hours : bool or dict
        if bool, then same logic applies for all builings
        if dict, then can be building specific
        True means only look at 9 AM to 6 PM (OCCUPANCY_TIME_RANGE)
        default = True
    no_weekends : bool or dict
        if bool, then same logic applies for all builings
        if dict, then can be building specific
        default = True, remove weekends for all buildings
    hourly_filter : pd.Series() or dict of pd.Series()
        allows user to filter by specific hours
        value in series is unimportant, only index used to filter
        if pd.Series(), same filter applies to all projects
        if dict of pd.Series(), building specific filters
        default = None, no filter
    hourly_filter_reverse : bool or dict of bool
        True, then scrub out the idxs out hourly filter
        False, then only have the hourly filter
        default False
    df_filter : pd.DataFrame or dict of of pd.DataFrame
        way to filter by another df
        cols and index should match df
        values should either be NAN or 1
    resample_rule : str
        allows user to re-index a df into different time steps
        default "1h", another option is None
    resample_statistic : str
        statistic used for resampling
        default "Mean", can also do "Sum"
    remove_FCUs : bool
        if True, FCUs are removed from zonal data
    SI_units : bool or dict o
        whether to convert to SI units
        currently implemented for airflow and temp variables
        converts F to C
        converts cfm to m3/h

    Returns
    -------
    Cleaned dfs as dict
    """
    projects = list(dfs.keys())
    # convert types
    start_date_dict = input_to_dict(start_date, projects)
    end_date_dict = input_to_dict(end_date, projects)
    only_business_hours_dict = input_to_dict(only_business_hours, projects)
    no_weekends_dict = input_to_dict(no_weekends, projects)
    hourly_filter_dict = input_to_dict(hourly_filter, projects)
    hourly_filter_reverse_dict = input_to_dict(hourly_filter_reverse, projects)
    df_filter_dict = input_to_dict(df_filter, projects)
    SI_units_dict = input_to_dict(SI_units, projects)
    cleaned_dfs = {}
    for project in projects:
        df = clean_df(
            df=dfs[project],
            this_var=this_var,
            start_date=start_date_dict[project],
            end_date=end_date_dict[project],
            only_business_hours=only_business_hours_dict[project],
            no_weekends=no_weekends_dict[project],
            hourly_filter=hourly_filter_dict[project],
            hourly_filter_reverse=hourly_filter_reverse_dict[project],
            df_filter=df_filter_dict[project],
            SI_units=SI_units_dict[project],
            resample_rule=resample_rule,
            resample_statistic=resample_statistic,
            remove_FCUs=remove_FCUs,
        )
        if df is None:
            df = pd.DataFrame()
        cleaned_dfs[project] = df
    return cleaned_dfs


def clean_by_column(
    df,
    start_date=None,
    end_date=None,
    only_business_hours=True,
    no_weekends=True,
    hourly_filter=None,
    hourly_filter_reverse=False,
    resample_rule="1h",
    SI_units=False,
):
    """
    Input is now a df, and we can clean each column separatley

    Parameters
    ----------
    df : pd.DataFrame
        index is time, columns are buildings or equips
    start_date : pd.Timestamp() or dict
        month-day-year
        if pd.Timestamp(), same date applies to all cols
        if dict, then can be col specific
        default = None, look at earliest date in dataset
    end_date : pd.Timestamp() or dict
        month-day-year
        if pd.Timestamp(), same date applies to all cols
        if dict, then can be col specific
        default = None, look at latest date in dataset
    only_business_hours : bool or dict
        if bool, then same logic applies for all builings
        if dict, then can be col specific
        True means only look at 9 AM to 6 PM (OCCUPANCY_TIME_RANGE)
        default = True
    no_weekends : bool or dict
        if bool, then same logic applies for all builings
        if dict, then can be col specific
        default = True, remove weekends for all cols
    hourly_filter : pd.Series() or dict of pd.Series()
        allows user to filter by specific hours
        value in series is unimportant, only index used to filter
        if pd.Series(), same filter applies to all cols
        if dict of pd.Series(), col specific filters
        default = None, no filter
    hourly_filter_reverse : bool or dict of bool
        True, then scrub out the idxs out hourly filter
        False, then only have the hourly filter
        default False
    resample_rule : str
        allows user to re-index a df into different time steps
        default "1h", another option is None
    SI_units : bool or dict of bools
        whether to convert to SI units
        currently implemented for airflow and temp variables
        converts F to C
        converts cfm to m3/h

    Returns
    -------
    Cleaned df
    """
    # convert types
    cols = list(df.columns)
    start_date_dict = input_to_dict(start_date, cols)
    end_date_dict = input_to_dict(end_date, cols)
    only_business_hours_dict = input_to_dict(only_business_hours, cols)
    no_weekends_dict = input_to_dict(no_weekends, cols)
    hourly_filter_dict = input_to_dict(hourly_filter, cols)
    hourly_filter_reverse_dict = input_to_dict(hourly_filter_reverse, cols)
    SI_units_dict = input_to_dict(SI_units, cols)

    cleaned_df = pd.DataFrame()
    for col in cols:
        try:
            ser = clean_df(
                df=df[col].to_frame(),
                this_var=None,
                start_date=start_date_dict[col],
                end_date=end_date_dict[col],
                only_business_hours=only_business_hours_dict[col],
                no_weekends=no_weekends_dict[col],
                hourly_filter=hourly_filter_dict[col],
                hourly_filter_reverse=hourly_filter_reverse_dict[col],
                resample_rule=resample_rule,
                SI_units=SI_units_dict[col],
            ).iloc[
                :, 0
            ]  # to ser
        except Exception as e:
            print(f"Error cleaning column {col}: {e}")
            continue
        # add rows if needed
        new_idx = list(set(ser.index) - set(cleaned_df.index))
        if len(new_idx) > 0:
            dummy_df = pd.DataFrame(np.nan, index=new_idx, columns=cleaned_df.columns)
            cleaned_df = pd.concat([cleaned_df, dummy_df], axis=0)
        cleaned_df[col] = ser
    cleaned_df.sort_index(inplace=True)
    return cleaned_df


def get_experiment_hourly_filter(
    projects, experiment_year, filter_columns, no_weekends=True, use_raw=False
):
    """
    Helper function to get dictionary of hourly filters based on experiments

    Parameters
    ----------
    projects : list
    experiment_year : str
    filter_columns : list or dict(list)
        list of binary df columns to include in filter
        with dict can be building specific
        "All" key word to get all columns
    no_weekends : bool
    use_raw : bool
        if True, use raw (un-corrected) daily sp schedule
        only applies to 2021/2022

    Returns
    -------
    hourly_filter dict
    """
    hourly_filter_dict = {}
    filter_columns_dict = input_to_dict(filter_columns, projects)
    for project in projects:
        binary_df = get_2021_2022_binary_df(
            project,
            experiment_year=experiment_year,
            freq="hourly",
            drop_baseline_column=False,
            no_weekends=no_weekends,
            control_for_weekends=(not no_weekends),
            use_raw=use_raw,
        )
        binary_df = binary_df[filter_columns_dict[project]]
        hourly_filter = binary_df.sum(axis=1)
        hourly_filter = hourly_filter[hourly_filter >= 1]
        hourly_filter_dict[project] = hourly_filter
    return hourly_filter_dict
