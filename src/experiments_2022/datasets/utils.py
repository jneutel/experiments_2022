import pandas as pd
import warnings

from experiments_2022.zone_level_analysis.cleaning import clean_columns
from experiments_2022.zone_level_analysis.base import trim_to_common_elements

warnings.filterwarnings("ignore")


AVAILABLE_PROJECTS = [
    "OFF-1",
    "OFF-2",
    "OFF-3",
    "OFF-4",
    "OFF-5",
    "OFF-6",
    "OFF-7",
    "OFF-8",
    "LAB-1",
    "LAB-2",
    "LAB-3",
]

AVAILABLE_VARIABLES = [
    "weather-oat",
    "weather-rh",
    "ahu-chwv_cmd",  # ahu core
    "ahu-dap",
    "ahu-dapsp",
    "ahu-fanspeed",
    "ahu-econ_cmd",
    "ahu-dat",
    "ahu-datsp",
    "ahu-rat",
    "ahu-mat",
    "ahu-oat",
    "zone-dat",  # zone core
    "zone-datsp",
    "zone-airflow",
    "zone-airflowsp",
    "zone-coolsp",
    "zone-heatsp",
    "zone-temps",
    "zone-tloads",
    "zone-damper",
    "zone-rhv",
    "zone-zonesp",
    "zone-cool_offset",
    "zone-heat_offset",
    "zone-map",
    "zone-deadband_top",  # zone derived
    "zone-deadband_bottom",
    "zone-local_offset",
    "zone-deviation_coolsp",
    "zone-deviation_heatsp",
    "zone-simple_cooling_requests",
    "zone-simple_heating_requests",
    "ahu-airflow",
]

VARIABLE_DEPENDENCIES = {
    "zone-deadband_top": ["zone-zonesp", "zone-cool_offset"],
    "zone-deadband_bottom": ["zone-zonesp", "zone-heat_offset"],
    "zone-local_offset": ["zone-zonesp", "zone-cool_offset", "zone-coolsp"],
    "zone-deviation_coolsp": ["zone-temps", "zone-coolsp"],
    "zone-deviation_heatsp": ["zone-temps", "zone-heatsp"],
    "zone-simple_cooling_requests": ["zone-tloads"],
    "zone-simple_heating_requests": ["zone-tloads"],
    "ahu-airflow": ["zone-map", "zone-airflow"],
}


def compute_zone_deadband_top(project, data_dict):
    """
    Computes the top of the deadband without local offset

    Parameters
    ----------
    project : str
        name of project
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-zonesp": df1, "zone-cool_offset": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-zonesp, zone-cool_offset
    - Returns data in F.
    """
    if project in ["OFF-3"]:
        return data_dict["zone-zonesp"]
    if project in ["OFF-6"]:
        return data_dict["zone-zonesp"] + 2
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    middle = data_dict["zone-zonesp"]
    offset = data_dict["zone-cool_offset"]
    middle, offset = trim_to_common_elements([middle, offset])
    return middle + offset.abs()


def compute_zone_deadband_bottom(project, data_dict):
    """
    Computes the bottom of the deadband without local offset

    Parameters
    ----------
    project : str
        name of project
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-zonesp": df1, "zone-heat_offset": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-zonesp, zone-heat_offset
    - Returns data in F.
    """
    if project in ["OFF-3"]:
        return data_dict["zone-zonesp"]
    if project in ["OFF-6"]:
        return data_dict["zone-zonesp"] + 2
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    middle = data_dict["zone-zonesp"]
    offset = data_dict["zone-heat_offset"]
    middle, offset = trim_to_common_elements([middle, offset])
    return middle - offset.abs()


def compute_zone_local_offset(project, data_dict):
    """
    Computes the local offset

    Parameters
    ----------
    project : str
        name of project
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-zonesp": df1, "zone-cool_offset": df2, "zone-coolsp" : df3}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-zonesp, zone-cool_offset, zone-coolsp
    - Returns data in F.
    """
    for this_var in ["zone-coolsp"]:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    eff_coolsp = data_dict["zone-coolsp"]
    top_deadband = compute_zone_deadband_top(
        project,
        {
            "zone-zonesp": data_dict["zone-zonesp"],
            "zone-cool_offset": data_dict["zone-cool_offset"],
        },
    )
    eff_coolsp, top_deadband = trim_to_common_elements([eff_coolsp, top_deadband])
    return eff_coolsp - top_deadband


def compute_zone_deviation_coolsp(data_dict):
    """
    Computes the deviation between zone temperatures and zone cooling setpoints.

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-temps": df1, "zone-coolsp": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-temps, zone-coolsp.
    - Returns data in F.
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    temp = data_dict["zone-temps"]
    coolsp = data_dict["zone-coolsp"]
    temp, coolsp = trim_to_common_elements([temp, coolsp])
    return temp - coolsp


def compute_zone_deviation_heatsp(data_dict):
    """
    Computes the deviation between zone temperatures and zone heating setpoints.

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-temps": df1, "zone-heatsp": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-temps, zone-heatsp.
    - Returns data in F.
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    temp = data_dict["zone-temps"]
    heatsp = data_dict["zone-heatsp"]
    temp, heatsp = trim_to_common_elements([temp, heatsp])
    return temp - heatsp


def compute_zone_simple_cooling_requests(data_dict):
    """
    1 if tload above 70%, 0 if not

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-tloads", df1}

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-tloads
    - Unitless
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    tload = data_dict["zone-tloads"]
    df = pd.DataFrame(0, index=tload.index, columns=tload.columns)
    df[tload >= 70] = 1
    return df


def compute_zone_simple_heating_requests(data_dict):
    """
    1 if tload below -70%, 0 if not

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-tloads", df1}

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-tloads
    - Unitless
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    tload = data_dict["zone-tloads"]
    df = pd.DataFrame(0, index=tload.index, columns=tload.columns)
    df[tload <= -70] = 1
    return df


def compute_ahu_airflow(data_dict):
    """
    Calculates total airflow supply bottom up using zone airflow

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case ...
        {
            "zone-map" : df1,
            "zone-airflow" : df2,
        }

    Returns
    -------
    ahu-airflow in units of CFM
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    zone_map = data_dict["zone-map"]
    zone_airflow = data_dict["zone-airflow"]
    common = list(zone_map.index.intersection(zone_airflow.columns))
    zone_map = zone_map.loc[common, :]
    zone_airflow = zone_airflow.loc[:, common]
    ahus = list(set(zone_map["AHU"]))
    summed_supply_air = pd.DataFrame(index=zone_airflow.index, columns=ahus)
    for ahu in ahus:
        these_zones = list(((zone_map[zone_map["AHU"] == ahu]).dropna()).index)
        this_airflow = zone_airflow.loc[:, these_zones]
        summed_supply_air[ahu] = this_airflow.sum(axis=1)
    summed_supply_air = clean_columns(summed_supply_air, "ahu-airflow")
    return summed_supply_air


# vars --> functions

FUNCTIONS = {
    "zone-deadband_top": compute_zone_deadband_top,
    "zone-deadband_bottom": compute_zone_deadband_bottom,
    "zone-local_offset": compute_zone_local_offset,
    "zone-deviation_coolsp": compute_zone_deviation_coolsp,
    "zone-deviation_heatsp": compute_zone_deviation_heatsp,
    "zone-simple_cooling_requests": compute_zone_simple_cooling_requests,
    "zone-simple_heating_requests": compute_zone_simple_heating_requests,
    "ahu-airflow": compute_ahu_airflow,
}
