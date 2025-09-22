import numpy as np
import pandas as pd
from experiments_2022.zone_level_analysis import (
    cleaning,
    viz,
    regression_functions,
)
from experiments_2022.datasets import load_zones
import copy

PROJECTS_2022 = [
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

PROJECTS_2022_TOTAL = [
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
    "TOTAL",
]

PROJECTS_2021 = ["OFF-1", "OFF-3", "OFF-4", "OFF-6", "LAB-1", "LAB-3"]

PROJECTS_VAV = [
    "OFF-1",
    "OFF-2",
    "OFF-3",
    "OFF-4",
    "OFF-5",
    "OFF-6",
    "OFF-7",
]

PROJECTS_FC = [
    "LAB-1",
    "LAB-2",
    "LAB-3",
]

PROJECTS_VAV_TOTAL = [
    "OFF-1",
    "OFF-2",
    "OFF-3",
    "OFF-4",
    "OFF-5",
    "OFF-6",
    "OFF-7",
    "TOTAL",
]

NO_WEEKENDS = {
    "OFF-1": True,
    "OFF-2": True,
    "OFF-3": True,
    "OFF-4": True,
    "OFF-5": True,
    "OFF-6": True,
    "OFF-7": True,
    "LAB-1": True,
    "LAB-2": True,
    "LAB-3": True,
}

CONTROL_FOR_WEEKENDS = {
    "OFF-1": False,
    "OFF-2": False,
    "OFF-3": False,
    "OFF-4": False,
    "OFF-5": False,
    "OFF-6": False,
    "OFF-7": False,
    "LAB-1": False,
    "LAB-2": False,
    "LAB-3": False,
}

SUMMER_START_2021 = pd.Timestamp("05-01-2021")
SUMMER_END_2021 = pd.Timestamp("10-01-2021")

SUMMER_START_2022 = pd.Timestamp("05-01-2022")
SUMMER_END_2022 = pd.Timestamp("10-01-2022")

DOMINANT_THRESH = 0.85
ROGUE_THRESH = 0.7
REACTIVE_THRESH = {"Pos": 20, "Neg": -20, "High Constant": 30, "Heating": -10}

RESPONSE_COLORS = {
    f"Reduced zonal load {abs(REACTIVE_THRESH['Neg'])}% or more": "ForestGreen",
    "Small change zonal load": "RoyalBlue",
    "Small change zonal load (remained high)": "DarkOrange",
    f"Increased zonal load {abs(REACTIVE_THRESH['Pos'])}% or more": "Firebrick",
    "Typically in heating": "Gray",
}

DOMINANT_COLORS = {
    "Dominated": "Teal",
    "Dominant": "Navy",
    "Rogue": "Coral",
}

SINGLE_PLOT_LEGEND_SIZE = 26
SINGLE_PLOT_TXT_SIZE = 23
SINGLE_PLOT_ANNOTATION_SIZE = 23

MULTI_PLOT_TITLE_SIZE = 32
MULTI_PLOT_LEGEND_SIZE = 30
MULTI_PLOT_ANNOTATION_SIZE = 24
MULTI_PLOT_TXT_SIZE = 24


def gini(x):
    x = np.array(x)
    x_sorted = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def add_vertical_boxes(fig, x_points, background_color="lightgray"):
    for i in range(len(x_points) - 1):
        fig.add_shape(
            type="rect",
            x0=x_points[i],
            x1=x_points[i + 1],
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor=background_color
            if i % 2 == 0
            else "rgba(0, 0, 0, 0)",  # Alternates between gray and transparent
            line=dict(width=0),  # No border for the rectangles,
            layer="below",
            opacity=0.7,
        )
    return fig


def add_line_to_subplots(fig, x_range, y_range, total_subplots, dash="dash", width=4):
    for i in range(1, total_subplots + 1):
        xref = "x" if i == 1 else f"x{i}"
        yref = "y" if i == 1 else f"y{i}"

        fig.add_shape(
            type="line",
            x0=x_range[0],
            y0=y_range[0],
            x1=x_range[1],
            y1=y_range[1],
            line=dict(color="black", dash=dash, width=width),
            xref=xref,
            yref=yref,
            layer="above",
        )

    return fig


def get_2022_control_data(
    project,
    variable,
    start_date=SUMMER_START_2022,
    end_date=SUMMER_END_2022,
    no_weekends=True,
    only_business_hours=True,
    SI_units=True,
    remove_FCUs=False,
    clean_underyling_data=False,
):
    df = load_zones("2022", project, variable, clean_data=clean_underyling_data)
    zonal_schedule = regression_functions.get_zonal_sp_schedule(
        project, experiment_year="2022", freq="hourly", df=df
    )
    df_filter = cleaning.create_sp_filter(
        zonal_schedule, sps=[76, 78], reverse_filter=True
    )

    df = cleaning.clean_df(
        df=df,
        this_var=variable,
        start_date=start_date,
        end_date=end_date,
        only_business_hours=only_business_hours,
        no_weekends=no_weekends,
        remove_FCUs=remove_FCUs,
        SI_units=SI_units,
        df_filter=df_filter,
    )
    return df


def run_building_regressions(
    df_dict,
    T,
    year="2022",
    mode="Percent Change",
    summary_statistic="Mean",
    y_axis_title="Y Axis Title",
    use_raw=False,
):
    projects = list(df_dict.keys())

    summary = pd.DataFrame(
        index=[
            "CSP = 76F",
            "CSP = 78F",
            "OAT",
            "Weekend",
            "P-Value CSP = 76F",
            "P-Value CSP = 78F",
            "P-Value OAT",
            "P-Value Weekend",
            "R2",
        ],
        columns=projects,
    )
    if year == "2022":
        sps = ["76", "78"]
        line_legend = {
            "name": {
                "Control": "CSP = 23.3°C",
                "CSP = 76F": "CSP = 24.4°C",
                "CSP = 78F": "CSP = 25.5°C",
            },
            "color": {
                "Control": "RoyalBlue",
                "CSP = 76F": "DarkOrange",
                "CSP = 78F": "Firebrick",
            },
        }
    else:
        sps = ["76"]
        line_legend = {
            "name": {
                "Control": "CSP = 23.3°C",
                "CSP = 76F": "CSP = 24.4°C",
            },
            "color": {
                "Control": "RoyalBlue",
                "CSP = 76F": "DarkOrange",
            },
        }

    deltas = pd.DataFrame(index=projects, columns=sps)
    deltas_high = pd.DataFrame(index=projects, columns=sps)
    deltas_low = pd.DataFrame(index=projects, columns=sps)
    figs = {}

    for project in projects:
        binary_df = regression_functions.get_2021_2022_binary_df(
            project=project,
            experiment_year=year,
            freq="daily",
            baseline_column="CSP = 74F",
            drop_baseline_column=True,
            no_weekends=NO_WEEKENDS[project],
            control_for_weekends=CONTROL_FOR_WEEKENDS[project],
            use_raw=use_raw,
        )
        reg_results = regression_functions.general_Delta_fn(
            df=df_dict[project],
            T=T,
            binary=binary_df,
            mode=mode,
            summary_statistic=summary_statistic,
        )

        for sp in sps:
            deltas.loc[project, sp] = reg_results.loc[project, f"Delta CSP = {sp}F"]
            deltas_high.loc[project, sp] = (
                reg_results.loc[project, f"Delta Low CSP = {sp}F"]
                - deltas.loc[project, sp]
            )
            deltas_low.loc[project, sp] = deltas.loc[project, sp] - (
                reg_results.loc[project, f"Delta High CSP = {sp}F"]
            )
        if NO_WEEKENDS[project]:
            if year == "2022":
                conditions = ["CSP = 76F", "CSP = 78F", "OAT"]
            else:
                conditions = ["CSP = 76F", "OAT"]
            shape_legend = None
            additive_column_dict = None
            dont_add_to_legend = []
        else:
            days = pd.DatetimeIndex(df_dict[project].index.date).unique()
            shapes = pd.Series(0, index=days)
            shapes[days.dayofweek >= 5] = 1
            shape_legend = {
                "series": shapes,
                "name": {0: "Weekday", 1: "Weekend"},
                "shape": {0: "circle", 1: "x"},
            }
            dont_add_to_legend = ["Weekday"]
            if year == "2022":
                additive_column_dict = {
                    "Weekend": ["Control", "CSP = 76F", "CSP = 78F"]
                }
                conditions = ["CSP = 76F", "CSP = 78F", "OAT", "Weekend"]
            else:
                additive_column_dict = {"Weekend": ["Control", "CSP = 76F"]}
                conditions = ["CSP = 76F", "OAT", "Weekend"]

        for condition in conditions:
            summary.loc[
                condition, project
            ] = f"{round(reg_results.loc[project, f'Slope {condition}'], 3)} ({round(reg_results.loc[project, f'Std Err {condition}'], 3)})"
            summary.loc[
                f"P-Value {condition}", project
            ] = f"{round(reg_results.loc[project, f'P-Value {condition}'], 3)}"
        summary.loc["R2", project] = round(reg_results.loc[project, "R2"], 3)
        fig = viz.plot_experiment_regression(
            experiment_results=reg_results,
            df=df_dict[project],
            T=T,
            binary=binary_df,
            line_legend=line_legend,
            shape_legend=shape_legend,
            additive_column_dict=additive_column_dict,
            mode=mode,
            summary_statistic=summary_statistic,
            marker_size=10,
            line_width=2.5,
            y_axis_title=y_axis_title,
            x_axis_title="Daytime Average OAT [°C]",
            dont_add_to_legend=dont_add_to_legend,
            title_size=MULTI_PLOT_TITLE_SIZE,
            text_size=MULTI_PLOT_TXT_SIZE,
            legend_size=MULTI_PLOT_LEGEND_SIZE,
        )
        figs[project] = fig

    # clean summary
    new_index = list(summary.index)
    for i in range(len(new_index)):
        idx = new_index[i]
        if "CSP = 76F" in idx:
            new_index[i] = idx.replace("CSP = 76F", "CSP = 24.4°C")
        if "CSP = 78F" in idx:
            new_index[i] = idx.replace("CSP = 78F", "CSP = 25.5°C")

    summary.index = new_index
    summary.columns = [word.capitalize() for word in list(summary.columns)]
    summary.dropna(inplace=True, how="all", axis=0)

    # combine regression figs
    regression_fig = viz.combine_figs(
        figs,
        y_axis_title=y_axis_title,
        x_axis_title="Daytime Average OAT [°C]",
        force_same_yaxes=False,
        force_same_xaxes=False,
        num_cols=3,
        horizontal_spacing=0.125,
        vertical_spacing=0.1,
    )

    return deltas, deltas_high, deltas_low, summary, regression_fig


def run_group_regressions(
    dict_df,
    T,
    mode,
    year="2022",
    summary_statistic="Mean",
    no_weekends=NO_WEEKENDS,
    control_for_weekends=CONTROL_FOR_WEEKENDS,
    use_raw=False,
):
    deltas_76 = {}
    deltas_low_76 = {}
    deltas_high_76 = {}

    deltas_78 = {}
    deltas_low_78 = {}
    deltas_high_78 = {}

    for project in dict_df:
        binary_df = regression_functions.get_2021_2022_binary_df(
            project=project,
            experiment_year=year,
            freq="daily",
            baseline_column="CSP = 74F",
            drop_baseline_column=True,
            no_weekends=no_weekends[project],
            control_for_weekends=control_for_weekends[project],
            use_raw=use_raw,
        )
        reg_results = regression_functions.general_Delta_fn(
            df=dict_df[project],
            T=T,
            binary=binary_df,
            mode=mode,
            summary_statistic=summary_statistic,
        )
        # grab results
        deltas_76[project] = reg_results["Delta CSP = 76F"].to_frame()
        deltas_low_76[project] = (
            reg_results["Delta CSP = 76F"] - reg_results["Delta Low CSP = 76F"]
        ).to_frame()
        deltas_high_76[project] = (
            reg_results["Delta High CSP = 76F"] - reg_results["Delta CSP = 76F"]
        ).to_frame()
        deltas_76[project].columns = ["76"]
        deltas_low_76[project].columns = ["76"]
        deltas_high_76[project].columns = ["76"]

        if year == "2022":
            deltas_78[project] = reg_results["Delta CSP = 78F"].to_frame()
            deltas_low_78[project] = (
                reg_results["Delta CSP = 78F"] - reg_results["Delta Low CSP = 78F"]
            ).to_frame()
            deltas_high_78[project] = (
                reg_results["Delta High CSP = 78F"] - reg_results["Delta CSP = 78F"]
            ).to_frame()
            deltas_78[project].columns = ["78"]
            deltas_low_78[project].columns = ["78"]
            deltas_high_78[project].columns = ["78"]
    if year == "2022":
        return (
            deltas_76,
            deltas_low_76,
            deltas_high_76,
            deltas_78,
            deltas_low_78,
            deltas_high_78,
        )
    return (
        deltas_76,
        deltas_low_76,
        deltas_high_76,
    )


def run_equip_regressions(
    dict_df,
    T,
    mode,
    year="2022",
    summary_statistic="Mean",
    no_weekends=NO_WEEKENDS,
    control_for_weekends=CONTROL_FOR_WEEKENDS,
):
    deltas_76 = {}
    deltas_low_76 = {}
    deltas_high_76 = {}

    deltas_78 = {}
    deltas_low_78 = {}
    deltas_high_78 = {}

    for project in dict_df:
        df = dict_df[project]
        zones = list(df.columns)

        this_deltas_76 = pd.Series(index=zones)
        this_deltas_low_76 = pd.Series(index=zones)
        this_deltas_high_76 = pd.Series(index=zones)
        if year == "2022":
            this_deltas_78 = pd.Series(index=zones)
            this_deltas_low_78 = pd.Series(index=zones)
            this_deltas_high_78 = pd.Series(index=zones)

        for zone in zones:
            binary_df = regression_functions.get_2021_2022_binary_df(
                project=project,
                experiment_year=year,
                freq="daily",
                baseline_column="CSP = 74F",
                drop_baseline_column=True,
                no_weekends=no_weekends[project],
                control_for_weekends=control_for_weekends[project],
                zone=zone,
            )
            reg_results = regression_functions.general_Delta_fn(
                df=df[zone].to_frame(),
                T=T,
                binary=binary_df,
                mode=mode,
                summary_statistic=summary_statistic,
            )
            # grab results
            this_deltas_76[zone] = reg_results.loc[zone, "Delta CSP = 76F"]
            this_deltas_low_76[zone] = (
                reg_results.loc[zone, "Delta CSP = 76F"]
                - reg_results.loc[zone, "Delta Low CSP = 76F"]
            )
            this_deltas_high_76[zone] = (
                reg_results.loc[zone, "Delta High CSP = 76F"]
                - reg_results.loc[zone, "Delta CSP = 76F"]
            )
            if year == "2022":
                this_deltas_78[zone] = reg_results.loc[zone, "Delta CSP = 78F"]
                this_deltas_low_78[zone] = (
                    reg_results.loc[zone, "Delta CSP = 78F"]
                    - reg_results.loc[zone, "Delta Low CSP = 78F"]
                )
                this_deltas_high_78[zone] = (
                    reg_results.loc[zone, "Delta High CSP = 78F"]
                    - reg_results.loc[zone, "Delta CSP = 78F"]
                )

        this_deltas_76 = this_deltas_76.to_frame()
        this_deltas_low_76 = this_deltas_low_76.to_frame()
        this_deltas_high_76 = this_deltas_high_76.to_frame()

        this_deltas_76.columns = ["76"]
        this_deltas_low_76.columns = ["76"]
        this_deltas_high_76.columns = ["76"]

        deltas_76[project] = this_deltas_76
        deltas_low_76[project] = this_deltas_low_76
        deltas_high_76[project] = this_deltas_high_76

        if year == "2022":
            this_deltas_78 = this_deltas_78.to_frame()
            this_deltas_low_78 = this_deltas_low_78.to_frame()
            this_deltas_high_78 = this_deltas_high_78.to_frame()

            this_deltas_78.columns = ["78"]
            this_deltas_low_78.columns = ["78"]
            this_deltas_high_78.columns = ["78"]

            deltas_78[project] = this_deltas_78
            deltas_low_78[project] = this_deltas_low_78
            deltas_high_78[project] = this_deltas_high_78
        print(f"Done with {project}")

    if year == "2022":
        return (
            deltas_76,
            deltas_low_76,
            deltas_high_76,
            deltas_78,
            deltas_low_78,
            deltas_high_78,
        )
    return (
        deltas_76,
        deltas_low_76,
        deltas_high_76,
    )


def add_total_to_dfs(dfs):
    these_dfs = copy.deepcopy(dfs)
    total_df = []
    for project in these_dfs:
        df = copy.deepcopy(these_dfs[project])
        new_col = [f"{project} {col}" for col in df.columns]
        df.columns = new_col
        total_df.append(df)
    these_dfs["TOTAL"] = pd.concat(total_df, axis=0)
    return these_dfs


def add_total_to_results_dict(results_dict):
    this_results_dict = copy.deepcopy(results_dict)
    total_df = []
    for project in this_results_dict:
        df = copy.deepcopy(this_results_dict[project])
        new_idx = [f"{project} {idx}" for idx in df.index]
        df.index = new_idx
        total_df.append(df)
    this_results_dict["TOTAL"] = pd.concat(total_df, axis=0)
    return this_results_dict
