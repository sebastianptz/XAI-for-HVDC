import numpy as np
import pandas as pd
import os


def get_flowdata_entsoe(
    feature, inname, outname, start_time="2015", end_time="2019/12/31"
):
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    entsoe_data_folder = "../../External_data/ENTSO-E/"
    filenames = {
        "flow": "PhysicalFlows_12.1.G",
        "sch_flow": "TotalCommercialSchedules_12.1.F",
        "outage": "UnavailabilityInTransmissionGrid_10.1.A_B",
    }
    featurenames = {"flow": "FlowValue", "sch_flow": "Capacity", "outage": "NewNTC"}
    filename = filenames[feature]
    featurename = featurenames[feature]

    df = pd.DataFrame()
    for i in np.arange(15, 20):
        for j in np.arange(1, 13):
            _df = pd.read_csv(
                entsoe_data_folder + "20{}_{:02d}_{}.csv".format(i, j, filename),
                sep="\t",
                encoding="utf-8",
                header=0,
                index_col=False,
            )
            df_temp = _df.loc[(_df.OutAreaName == outname) & (_df.InAreaName == inname)]
            if feature in ["flow", "sch_flow"]:
                df_temp.loc[:, "DateTime"] = pd.to_datetime(df_temp.loc[:, "DateTime"])
                df_temp = (
                    df_temp.sort_values(by="DateTime")
                    .set_index("DateTime")
                    .resample("1H")
                    .asfreq()
                )
                df_temp2 = _df.loc[
                    (_df.OutAreaName == inname) & (_df.InAreaName == outname)
                ]
                df_temp2.loc[:, "DateTime"] = pd.to_datetime(
                    df_temp2.loc[:, "DateTime"]
                )
                df_temp2 = (
                    df_temp2.sort_values(by="DateTime")
                    .set_index("DateTime")
                    .resample("1H")
                    .asfreq()
                )
                df_temp.loc[:, featurename] = (
                    df_temp[featurename] - df_temp2[featurename]
                )
            elif feature in ["outage"]:
                df_temp.loc[:, "StartOutage"] = pd.to_datetime(
                    df_temp.loc[:, "StartOutage"]
                )
                df_temp = df_temp.sort_values(by="StartOutage")
                df_temp2 = _df.loc[
                    (_df.OutAreaName == inname) & (_df.InAreaName == outname)
                ]
                df_temp2.loc[:, "StartOutage"] = pd.to_datetime(
                    df_temp2.loc[:, "StartOutage"]
                )
                df_temp2 = df_temp2.sort_values(by="StartOutage")
                df_temp = df_temp.append(df_temp2)
                df_temp = df_temp.sort_values(by="StartOutage")

            df = df.append(df_temp)
    if feature in ["outage"]:
        return df
    else:
        df = df.loc[(df.index >= start_time) & (df.index <= end_time)]
        df.index = df.index.tz_localize("UTC")
        return df[featurename]


def main():
    links = {
        "IFA": ["GB", "FR"],
        "Britned": ["GB", "NL"],
        "Norned": ["NO2", "NL"],
        "Nordbalt": ["SE4", "LT"],
        "Estlink": ["FI", "EE"],
        "Swepol": ["SE4", "PL"],
        "Storebaelt": ["DK2", "DK1"],
        "Skagerrak": ["NO2", "DK1"],
        "Kontiskan": ["SE3", "DK1"],
    }

    sch_flow_data = pd.DataFrame()

    if not os.path.exists("../data/HVDClinks/"):
        os.makedirs("../data/HVDClinks/")

    for link, areas in links.items():
        print("--- {} ---".format(link))
        df_flow = get_flowdata_entsoe(
            "flow", "{} BZN".format(areas[0]), "{} BZN".format(areas[1])
        )
        df_sch_flow = get_flowdata_entsoe(
            "sch_flow", "{} BZN".format(areas[0]), "{} BZN".format(areas[1])
        )
        df_outage = get_flowdata_entsoe(
            "outage", "{} BZN".format(areas[0]), "{} BZN".format(areas[1])
        )

        df_outage["StartOutage"] = pd.to_datetime(df_outage["StartOutage"])
        df_outage["EndOutage"] = pd.to_datetime(df_outage["EndOutage"])

        df_outage["StartOutage"] = df_outage["StartOutage"].dt.tz_localize("CET")
        df_outage["EndOutage"] = df_outage["EndOutage"].dt.tz_localize("CET")

        df_clean = df_sch_flow - df_flow

        for i in range(
            len(
                df_outage.loc[
                    (df_outage.Status == "Active")
                    & (df_outage.ProductionType == "DC Link")
                ]
            )
        ):
            start = df_outage.loc[
                (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
            ].iloc[i]["StartOutage"]
            end = df_outage.loc[
                (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
            ].iloc[i]["EndOutage"]
            if pd.to_datetime(end) - pd.to_datetime(start) < pd.to_timedelta("60 days"):
                df_clean[(start):(end)] = np.nan

        if os.path.exists("../data/HVDClinks/unscheduled_flows.pkl"):
            clean_data = pd.read_pickle("../data/HVDClinks/unscheduled_flows.pkl")
        else:
            clean_data = pd.DataFrame()
        if os.path.exists("../data/HVDClinks/scheduled_flows.pkl"):
            sch_flow_data = pd.read_pickle("../data/HVDClinks/scheduled_flows.pkl")
        else:
            sch_flow_data = pd.DataFrame()

        clean_data[link] = df_clean
        sch_flow_data[link] = df_sch_flow

        clean_data.to_pickle("../data/HVDClinks/unscheduled_flows.pkl")
        sch_flow_data.to_pickle("../data/HVDClinks/scheduled_flows.pkl")

    links = {"EWIC": ["UK(National Grid)", "IE"], "Moyle": ["UK(National Grid)", "NIE"]}

    sch_flow_data = pd.DataFrame()

    for link, areas in links.items():
        print("--- {} ---".format(link))
        df_flow = get_flowdata_entsoe(
            "flow", "{} CTA".format(areas[0]), "{} CTA".format(areas[1])
        )
        df_sch_flow = get_flowdata_entsoe(
            "sch_flow", "{} CTA".format(areas[0]), "{} CTA".format(areas[1])
        )
        df_outage = get_flowdata_entsoe(
            "outage", "{} CTA".format(areas[0]), "{} CTA".format(areas[1])
        )

        df_outage["StartOutage"] = pd.to_datetime(df_outage["StartOutage"])
        df_outage["EndOutage"] = pd.to_datetime(df_outage["EndOutage"])

        df_outage["StartOutage"] = df_outage["StartOutage"].dt.tz_localize("CET")
        df_outage["EndOutage"] = df_outage["EndOutage"].dt.tz_localize("CET")

        df_clean = df_sch_flow - df_flow

        for i in range(
            len(
                df_outage.loc[
                    (df_outage.Status == "Active")
                    & (df_outage.ProductionType == "DC Link")
                ]
            )
        ):
            start = df_outage.loc[
                (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
            ].iloc[i]["StartOutage"]
            end = df_outage.loc[
                (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
            ].iloc[i]["EndOutage"]
            df_clean[
                (start - pd.Timedelta(hours=2)) : (end + pd.Timedelta(hours=2))
            ] = np.nan
            if pd.to_datetime(end) - pd.to_datetime(start) < pd.to_timedelta("60 days"):
                df_clean[(start):(end)] = np.nan

        if os.path.exists("../data/HVDClinks/unscheduled_flows.pkl"):
            clean_data = pd.read_pickle("../data/HVDClinks/unscheduled_flows.pkl")
        if os.path.exists("../data/HVDClinks/scheduled_flows.pkl"):
            sch_flow_data = pd.read_pickle("../data/HVDClinks/scheduled_flows.pkl")

        clean_data[link] = df_clean
        sch_flow_data[link] = df_sch_flow

        clean_data.to_pickle("../data/HVDClinks/unscheduled_flows.pkl")
        sch_flow_data.to_pickle("../data/HVDClinks/scheduled_flows.pkl")

    link = "Kontek"
    konteklist = [["DK2", "DE-LU"], ["DK2", "DE-AT-LU"]]

    sch_flow_data = pd.DataFrame()

    print("--- {} ---".format(link))
    df_flow = get_flowdata_entsoe(
        "flow", "{} BZN".format(konteklist[0][0]), "{} BZN".format(konteklist[0][1])
    )
    df_sch_flow = get_flowdata_entsoe(
        "sch_flow", "{} BZN".format(konteklist[0][0]), "{} BZN".format(konteklist[0][1])
    )
    df_outage = get_flowdata_entsoe(
        "outage", "{} BZN".format(konteklist[0][0]), "{} BZN".format(konteklist[0][1])
    )
    df_flow = df_flow.append(
        get_flowdata_entsoe(
            "flow", "{} BZN".format(konteklist[1][0]), "{} BZN".format(konteklist[1][1])
        )
    )
    df_flow = df_flow[~df_flow.index.duplicated(keep="last")]
    df_sch_flow = df_sch_flow.append(
        get_flowdata_entsoe(
            "sch_flow",
            "{} BZN".format(konteklist[1][0]),
            "{} BZN".format(konteklist[1][1]),
        )
    )
    df_sch_flow = df_sch_flow[~df_sch_flow.index.duplicated(keep="last")]
    df_outage = df_outage.append(
        get_flowdata_entsoe(
            "outage",
            "{} BZN".format(konteklist[1][0]),
            "{} BZN".format(konteklist[1][1]),
        )
    )
    df_outage = df_outage[~df_outage.index.duplicated(keep="last")]

    df_outage["StartOutage"] = pd.to_datetime(df_outage["StartOutage"])
    df_outage["EndOutage"] = pd.to_datetime(df_outage["EndOutage"])

    df_outage["StartOutage"] = df_outage["StartOutage"].dt.tz_localize("CET")
    df_outage["EndOutage"] = df_outage["EndOutage"].dt.tz_localize("CET")

    df_clean = df_sch_flow - df_flow

    for i in range(
        len(
            df_outage.loc[
                (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
            ]
        )
    ):
        start = df_outage.loc[
            (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
        ].iloc[i]["StartOutage"]
        end = df_outage.loc[
            (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
        ].iloc[i]["EndOutage"]
        if pd.to_datetime(end) - pd.to_datetime(start) < pd.to_timedelta("60 days"):
            df_clean[(start):(end)] = np.nan

    if os.path.exists("../data/HVDClinks/unscheduled_flows.pkl"):
        clean_data = pd.read_pickle("../data/HVDClinks/unscheduled_flows.pkl")
    if os.path.exists("../data/HVDClinks/scheduled_flows.pkl"):
        sch_flow_data = pd.read_pickle("../data/HVDClinks/scheduled_flows.pkl")

    clean_data[link] = df_clean
    sch_flow_data[link] = df_sch_flow

    clean_data.to_pickle("../data/HVDClinks/unscheduled_flows.pkl")
    sch_flow_data.to_pickle("../data/HVDClinks/scheduled_flows.pkl")

    link = "Baltic Cable"
    balticlist = [["SE4", "DE-LU"], ["SE4", "DE-AT-LU"]]

    sch_flow_data = pd.DataFrame()

    print("--- {} ---".format(link))
    df_flow = get_flowdata_entsoe(
        "flow", "{} BZN".format(balticlist[0][0]), "{} BZN".format(balticlist[0][1])
    )
    df_sch_flow = get_flowdata_entsoe(
        "sch_flow", "{} BZN".format(balticlist[0][0]), "{} BZN".format(balticlist[0][1])
    )
    df_outage = get_flowdata_entsoe(
        "outage", "{} BZN".format(balticlist[0][0]), "{} BZN".format(balticlist[0][1])
    )
    df_flow = df_flow.append(
        get_flowdata_entsoe(
            "flow", "{} BZN".format(balticlist[1][0]), "{} BZN".format(balticlist[1][1])
        )
    )
    df_flow = df_flow[~df_flow.index.duplicated(keep="last")]
    df_sch_flow = df_sch_flow.append(
        get_flowdata_entsoe(
            "sch_flow",
            "{} BZN".format(balticlist[1][0]),
            "{} BZN".format(balticlist[1][1]),
        )
    )
    df_sch_flow = df_sch_flow[~df_sch_flow.index.duplicated(keep="last")]
    df_outage = df_outage.append(
        get_flowdata_entsoe(
            "outage",
            "{} BZN".format(balticlist[1][0]),
            "{} BZN".format(balticlist[1][1]),
        )
    )
    df_outage = df_outage[~df_outage.index.duplicated(keep="last")]

    df_outage["StartOutage"] = pd.to_datetime(df_outage["StartOutage"])
    df_outage["EndOutage"] = pd.to_datetime(df_outage["EndOutage"])

    df_outage["StartOutage"] = df_outage["StartOutage"].dt.tz_localize("CET")
    df_outage["EndOutage"] = df_outage["EndOutage"].dt.tz_localize("CET")

    df_clean = df_sch_flow - df_flow

    for i in range(
        len(
            df_outage.loc[
                (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
            ]
        )
    ):
        start = df_outage.loc[
            (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
        ].iloc[i]["StartOutage"]
        end = df_outage.loc[
            (df_outage.Status == "Active") & (df_outage.ProductionType == "DC Link")
        ].iloc[i]["EndOutage"]
        if pd.to_datetime(end) - pd.to_datetime(start) < pd.to_timedelta("60 days"):
            df_clean[(start):(end)] = np.nan

    if os.path.exists("../data/HVDClinks/unscheduled_flows.pkl"):
        clean_data = pd.read_pickle("../data/HVDClinks/unscheduled_flows.pkl")
    if os.path.exists("../data/HVDClinks/scheduled_flows.pkl"):
        sch_flow_data = pd.read_pickle("../data/HVDClinks/scheduled_flows.pkl")

    clean_data[link] = df_clean
    sch_flow_data[link] = df_sch_flow

    clean_data.to_pickle("../data/HVDClinks/unscheduled_flows.pkl")
    sch_flow_data.to_pickle("../data/HVDClinks/scheduled_flows.pkl")


if __name__ == "__main__":
    main()
