import numpy as np
import pandas as pd
import statsmodels.api as sm
from experiments_2022 import DATASETS_PATH

FORMAL_TRIALS_2022_START = {
    "OFF-1": pd.Timestamp("05-23-2022"),
    "OFF-2": pd.Timestamp("06-27-2022"),
    "OFF-3": pd.Timestamp("05-16-2022"),
    "OFF-4": pd.Timestamp("05-16-2022"),
    "OFF-5": pd.Timestamp("05-23-2022"),
    "OFF-6": pd.Timestamp("05-16-2022"),
    "OFF-7": pd.Timestamp("05-16-2022"),
    "OFF-8": pd.Timestamp("05-16-2022"),
    "LAB-1": pd.Timestamp("05-16-2022"),
    "LAB-2": pd.Timestamp("07-05-2022"),
    "LAB-3": pd.Timestamp("05-23-2022"),
}

FORMAL_TRIALS_2022_END = {
    "OFF-1": pd.Timestamp("09-11-2022"),
    "OFF-2": pd.Timestamp("09-11-2022"),
    "OFF-3": pd.Timestamp("08-21-2022"),
    "OFF-4": pd.Timestamp("09-11-2022"),
    "OFF-5": pd.Timestamp("09-11-2022"),
    "OFF-6": pd.Timestamp("08-21-2022"),
    "OFF-7": pd.Timestamp("08-28-2022"),
    "OFF-8": pd.Timestamp("09-11-2022"),
    "LAB-1": pd.Timestamp("08-21-2022"),
    "LAB-2": pd.Timestamp("09-11-2022"),
    "LAB-3": pd.Timestamp("08-03-2022"),
}


def get_2021_2022_binary_df(
    project,
    experiment_year="2022",
    freq="daily",
    baseline_column="CSP = 74F",
    drop_baseline_column=True,
    no_weekends=False,
    control_for_weekends=True,
    zone=None,
    use_raw=False,
):
    """
    Creates binary df to be used in 2022 regression analyses

    Parameters
    ----------
    project : str
        building to create binary df for
    experiment_year : str
        "2022" or "2021"
    freq : str
        daily or hourly
    baseline_column : str
        which column to use as baseline
        default is "CSP = 74F"
    drop_baseline_column : bool
        whether or not to drop baseline column
        default is True
    no_weekends : bool
        whether to exclude weekend data or not
        default is False
    control_for_weekends
        whether to control for weekends or not
        default is True, but ignored if no_weekends is True
    zone : str
        zone specific schedule
    use_raw : bool
        if True, use raw (un-corrected) daily sp schedule
        default False

    Returns
    -------
    df
    """
    # get encoded schedule
    if zone is not None:
        ez_csv = pd.read_csv(
            DATASETS_PATH / f"csvs/{experiment_year}_experiment_csvs/excluded_zones.csv"
        )
        ezs = list(ez_csv[project].dropna())
        # zonal schedule
        if freq == "hourly":
            path = f"{DATASETS_PATH}/csvs/{experiment_year}_experiment_csvs/sp_schedule_hourly_{project}.csv"
            encoded_schedule = pd.read_csv(
                path, index_col=0, parse_dates=True, date_format="%m/%d/%y %H:%M"
            )
        else:
            path = f"{DATASETS_PATH}/csvs/{experiment_year}_experiment_csvs/sp_schedule_daily_{project}.csv"
            encoded_schedule = pd.read_csv(
                path, index_col=0, parse_dates=True, date_format="%m/%d/%y"
            )
        if (zone not in list(encoded_schedule.columns)) or (zone in ezs):
            return get_2021_2022_binary_df(
                project=project,
                experiment_year=experiment_year,
                freq=freq,
                baseline_column=baseline_column,
                drop_baseline_column=drop_baseline_column,
                no_weekends=no_weekends,
                control_for_weekends=control_for_weekends,
                zone=None,
            )
        else:
            encoded_schedule = encoded_schedule[zone]

    else:
        # building wide
        if use_raw:
            if freq == "hourly":
                path = f"{DATASETS_PATH}/csvs/{experiment_year}_experiment_csvs/sp_schedule_hourly_raw.csv"
                encoded_schedule = pd.read_csv(
                    path, index_col=0, parse_dates=True, date_format="%m/%d/%y %H:%M"
                )[project]
            else:
                path = f"{DATASETS_PATH}/csvs/{experiment_year}_experiment_csvs/sp_schedule_daily_raw.csv"
                encoded_schedule = pd.read_csv(
                    path, index_col=0, parse_dates=True, date_format="%m/%d/%y"
                )[project]
        else:
            if freq == "hourly":
                path = f"{DATASETS_PATH}/csvs/{experiment_year}_experiment_csvs/sp_schedule_hourly.csv"
                encoded_schedule = pd.read_csv(
                    path, index_col=0, parse_dates=True, date_format="%m/%d/%y %H:%M"
                )[project]
            else:
                path = f"{DATASETS_PATH}/csvs/{experiment_year}_experiment_csvs/sp_schedule_daily.csv"
                encoded_schedule = pd.read_csv(
                    path, index_col=0, parse_dates=True, date_format="%m/%d/%y"
                )[project]

    encoded_schedule.index = pd.to_datetime(encoded_schedule.index)

    # Prepare
    if experiment_year == "2021":
        sps = [74, 76]
    elif experiment_year == "2022":
        sps = [74, 76, 78]
    binary_df = pd.DataFrame(index=encoded_schedule.index)

    # Create binary columns
    for sp in sps:
        binary_df[f"CSP = {sp}F"] = (encoded_schedule == sp).astype(float)

    # handle weekends
    binary_df["Weekend"] = (binary_df.index.dayofweek >= 5).astype(int)
    if no_weekends:
        binary_df.loc[binary_df["Weekend"] == 1, :] = np.nan
        control_for_weekends = False

    if not control_for_weekends:
        binary_df.drop(columns="Weekend", inplace=True)

    # Handle NaNs: if value is NaN → make entire row NaN
    binary_df.loc[encoded_schedule.isna(), :] = np.nan

    # Mask values not in sps
    valid_mask = encoded_schedule.isin(sps)
    binary_df.loc[~valid_mask, :] = np.nan

    if drop_baseline_column:
        binary_df.drop(columns=baseline_column, inplace=True)

    return binary_df


def general_Delta_fn(df, T, binary, mode="Absolute Change", summary_statistic="Mean"):
    """
    General purpose function to find delta due to setpoint change

    Parameters
    ----------
    df : pd.DataFrame
        df with time as index (hourly) and equips as cols
    T : pd.Series
        outside temperature data of interest (hourly)
    binary : pd.Series or pd.DataFrame
        0 if not test day, 1 if test day
        each column represents a different test, e.g. all zone day vs dominant zone day
    mode : str
        "Absolute Change" or "Percent Change"
    summary_statistic : str
        "Mean" or "Sum"

    Returns
    -------
    pd.DataFrame, results of regression
    """
    # allow for pd.DataFrame input as T
    if T is not None and isinstance(T, pd.DataFrame):
        T = T["temperture"]

    # allow for pd.Series input as binary
    if isinstance(binary, pd.Series):
        binary = binary.to_frame()
        binary.columns = ["High SP"]

    # prep raw data
    if summary_statistic == "Mean":
        df = df.groupby(df.index.date).mean()
    if summary_statistic == "Sum":
        df = df.groupby(df.index.date).sum(min_count=1)
    df.index = pd.DatetimeIndex(df.index)
    if mode == "Percent Change":
        df[df <= 0] = np.nan  # causes error with log

    binary.dropna(how="all", inplace=True)
    binary.index = pd.DatetimeIndex(binary.index)
    if T is not None:
        T = T.groupby(T.index.date).mean()
        T.index = pd.DatetimeIndex(T.index)
        common = binary.index.intersection(df.index).intersection(T.index, sort=True)
        binary = binary.loc[common, :]
        df = df.loc[common, :]
        T = T.loc[common]
        vars = ["OAT", "Intercept"]
    else:
        common = binary.index.intersection(df.index, sort=True)
        binary = binary.loc[common, :]
        df = df.loc[common, :]
        vars = ["Intercept"]
    # initializations
    equips = list(df.columns)
    tests = list(binary.columns)
    vars.extend(tests)
    cols = []
    for var in vars:
        cols.append(f"Slope {var}")
        cols.append(f"Slope Low {var}")
        cols.append(f"Slope High {var}")
        cols.append(f"Std Err {var}")
        cols.append(f"P-Value {var}")
    for test in tests:
        cols.append(f"Delta {test}")
        cols.append(f"Delta Low {test}")
        cols.append(f"Delta High {test}")
    cols.append("R2")
    all_results = pd.DataFrame(data=np.nan, index=equips, columns=cols)

    # run regression for each equip
    for equip in equips:
        # filter data for this equip
        ys = df[equip]
        ys = ys.dropna()
        if len(ys) < 3:
            all_results.loc[equip, :] = np.nan
            continue
        if mode == "Percent Change":
            ys = np.log(ys)
        intercept = pd.Series(1, index=ys.index)
        bins = binary.loc[ys.index, :]
        if T is not None:
            ts = T.loc[ys.index]
            xs = pd.concat([ts, intercept, bins], axis=1)
        else:
            xs = pd.concat([intercept, bins], axis=1)
        xs.columns = vars

        # fit model and store info
        model = sm.OLS(ys, xs)
        results = model.fit()
        for var in vars:
            all_results.loc[equip, f"Slope {var}"] = results.params[var]
            all_results.loc[equip, f"Std Err {var}"] = results.bse[var]
            all_results.loc[equip, f"Slope Low {var}"] = (
                results.params[var] - 1.96 * results.bse[var]
            )
            all_results.loc[equip, f"Slope High {var}"] = (
                results.params[var] + 1.96 * results.bse[var]
            )
            all_results.loc[equip, f"P-Value {var}"] = results.pvalues[var]
        all_results.loc[equip, "R2"] = results.rsquared

    if mode == "Percent Change":
        for test in tests:
            for err in [" ", " Low ", " High "]:
                all_results[f"Delta{err}{test}"] = 100 * (
                    np.exp(all_results[f"Slope{err}{test}"].astype(float)) - 1
                )
    else:
        for test in tests:
            for err in [" ", " Low ", " High "]:
                all_results[f"Delta{err}{test}"] = all_results[f"Slope{err}{test}"]
    return all_results


def general_regression_fn(
    y_data,
    x_data,
):
    """
    General purpose function to regress y onto x by equip

    Parameters
    ----------
    y_data : pd.DataFrame
        y_data with time as index (hourly) and equips as cols
    x_data : pd.Series
        x_data with time as index (hourly) and equips as cols

    Returns
    -------
    pd.DataFrame, results of regression
    """
    # prep raw data
    y_data = y_data.groupby(y_data.index.date).mean()
    y_data.index = pd.DatetimeIndex(y_data.index)
    x_data = x_data.groupby(x_data.index.date).mean()
    x_data.index = pd.DatetimeIndex(x_data.index)
    common = y_data.index.intersection(x_data.index, sort=True)
    y_data = y_data.loc[common, :]
    x_data = x_data.loc[common, :]
    # initializations
    equips = list(y_data.columns)
    all_results = pd.DataFrame(
        index=equips,
        columns=[
            "Slope X",
            "Slope Intercept",
            "Std Err X",
            "Std Err Intercept",
            "P-Value X",
            "P-Value Intercept",
            "R2",
        ],
    )
    # run regression for each equip
    for equip in equips:
        # filter data for this equip
        ys = y_data[equip]
        ys = ys.dropna()
        xs = x_data[equip]
        xs = xs.dropna()
        common = ys.index.intersection(xs.index, sort=True)
        ys = ys[common]
        xs = xs[common]
        if len(ys) < 3:
            all_results.loc[equip, :] = np.nan
            continue
        xs = xs.to_frame()
        xs["Intercept"] = pd.Series(1, index=xs.index)
        vars = ["X", "Intercept"]
        xs.columns = vars
        # fit model and store info
        model = sm.OLS(ys, xs)
        results = model.fit()
        for var in vars:
            all_results.loc[equip, f"Slope {var}"] = results.params[var]
            all_results.loc[equip, f"Std Err {var}"] = results.bse[var]
            all_results.loc[equip, f"P-Value {var}"] = results.pvalues[var]
        all_results.loc[equip, "R2"] = results.rsquared
    return all_results
