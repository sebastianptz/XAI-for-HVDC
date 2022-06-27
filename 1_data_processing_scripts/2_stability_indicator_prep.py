import sys
import os
import numpy as np
import pandas as pd

sys.path.append("../")
from utils.stability_indicators import calc_rocof, make_frequency_data_hdf


def main():
    # Time zones of frequency recordings
    tzs = {"CE": "CET", "Nordic": "Europe/Helsinki", "GB": "GB"}

    # Datetime parameters for output data generation
    start = pd.Timestamp("2015-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2019-12-31 00:00:00", tz="UTC")
    time_resol = pd.Timedelta("1H")

    # Pre-processed frequency csv files
    frequency_csv_folder = "../../Frequency_data_base/"
    tso_names = {"GB": "Nationalgrid", "CE": "TransnetBW", "Nordic": "Fingrid"}

    # HDF frequency files (for faster access than csv files)
    frequency_hdf_folder = {
        "GB": "../../Frequency_data_preparation/Nationalgrid/",
        "CE": "../../Frequency_data_preparation/TransnetBW/",
        "Nordic": "../../Frequency_data_preparation/Fingrid/",
    }

    # Nan treatment
    skip_hour_with_nan = True

    # Parameters for rocof estimation
    smooth_windows = {"CE": 60, "GB": 60, "Nordic": 30}
    lookup_windows = {"CE": 60, "GB": 60, "Nordic": 30}

    for area in ["GB", "CE", "Nordic"]:

        print("\n######", area, "######")

        # If not existent, create HDF file from csv files
        # (for faster access when trying out things)
        hdf_file = make_frequency_data_hdf(
            frequency_csv_folder,
            tso_names[area],
            frequency_hdf_folder[area],
            start,
            end,
            tzs[area],
            delete_existing_hdf=True,
        )

        # Output data folder
        folder = "../data/{}/".format(area)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Load frequency data
        freq = pd.read_hdf(hdf_file).loc[start:end]
        freq = freq - 50

        # Setup datetime index for output data
        index = pd.date_range(start, end, freq=time_resol, tz="UTC").tz_convert(
            tzs[area]
        )
        if os.path.exists(folder + "indicators.h5"):
            indicators = pd.read_hdf(folder + "indicators.h5")
        else:
            indicators = pd.DataFrame(index=index)

        # Extract stability indicators
        print("Extracting stability indicators ...")
        indicators["f_integral"] = freq.groupby(pd.Grouper(freq="1H")).sum()
        indicators["f_ext"] = freq.groupby(pd.Grouper(freq="1H")).apply(
            lambda x: x[x.abs().idxmax()] if x.notnull().any() else np.nan
        )
        indicators["f_rocof"] = calc_rocof(
            freq, smooth_windows[area], lookup_windows[area]
        )
        indicators["f_msd"] = (freq**2).groupby(pd.Grouper(freq="1H")).mean()

        # Set hour to NaN if frequency contains at least one NaN in that hour
        if skip_hour_with_nan:
            hours_with_nans = freq.groupby(pd.Grouper(freq="1H")).apply(
                lambda x: x.isnull().any()
            )
            print("hours_with_nans ", hours_with_nans.sum())
            indicators.loc[hours_with_nans] = np.nan

        # Save data
        indicators.to_hdf(folder + "indicators.h5", key="df")


if __name__ == "__main__":
    main()
