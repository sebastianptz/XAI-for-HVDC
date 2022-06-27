import pandas as pd
from sklearn import model_selection
import os


def main():
    # Areas inlcuding "country" areas
    areas = ["GB", "Nordic"]
    combined_areas = {"GB-CE": ["GB", "CE"], "Nordic-CE": ["Nordic", "CE"]}

    area_links = {
        "GB": [
            "EWIC",
            "Moyle",
        ],
        "Nordic": [
            "Nordbalt",
            "Estlink",
        ],
        "GB-CE": [
            "IFA",
            "Britned",
        ],
        "Nordic-CE": [
            "Norned",
            "Swepol",
            "Baltic Cable",
            "Storebaelt",
            "Skagerrak",
            "Kontiskan",
            "Kontek",
        ],
    }

    for area in areas:
        for target in area_links[area]:
            print("Processing external features from", area)

            # Setup folder for this specific version of train-test data
            input_folder = "../data/{}/".format(area)
            folder = "../data/flow_input/{}/{}/".format(area, target)
            version_folder = (
                folder + "version_" + pd.Timestamp("today").strftime("%Y-%m-%d") + "/"
            )
            if not os.path.exists(version_folder):
                os.makedirs(version_folder)

            # Load actual and forecast (day-ahead available) data
            X_actual = pd.read_hdf(input_folder + "input_actual.h5").filter(
                regex="(?<!_flow)$(?<!_flow_ramp)$(?<!_total_ramp)$(?<!_import_export_total)$"
            )
            X_forecast = pd.read_hdf(input_folder + "input_forecast.h5")
            X_indicator = pd.read_hdf(input_folder + "indicators.h5")
            X_forecast = X_forecast.join(
                pd.read_pickle("../data/HVDClinks/scheduled_flows.pkl")[target]
            )
            y = pd.read_pickle("../data/HVDClinks/unscheduled_flows.pkl").loc[:, target]

            # Drop nan values
            valid_ind = (
                ~pd.concat([X_forecast, X_actual, X_indicator, y], axis=1)
                .isnull()
                .any(axis=1)
            )
            X_forecast, X_actual, X_indicator, y = (
                X_forecast[valid_ind],
                X_actual[valid_ind],
                X_indicator[valid_ind],
                y[valid_ind],
            )

            # Join features for full model
            X_full = X_actual.join(X_forecast).join(X_indicator)

            # Train-test split
            (
                X_train_full,
                X_test_full,
                y_train,
                y_test,
            ) = model_selection.train_test_split(
                X_full, y, test_size=0.2, random_state=42
            )
            X_train_day_ahead = X_forecast.loc[X_train_full.index]
            X_test_day_ahead = X_forecast.loc[X_test_full.index]
            X_train_indicators = X_indicator.loc[X_train_full.index]
            X_test_indicators = X_indicator.loc[X_test_full.index]
            y_pred = pd.DataFrame(index=y_test.index)

            # Save data for full model and restricted models
            X_train_full.to_hdf(version_folder + "X_train_full_links.h5", key="df")
            X_train_day_ahead.to_hdf(
                version_folder + "X_train_day_ahead_links.h5", key="df"
            )
            X_train_indicators.to_hdf(
                version_folder + "X_train_indicators_links.h5", key="df"
            )
            y_train.to_hdf(version_folder + "y_train_links.h5", key="df")
            y_test.to_hdf(version_folder + "y_test_links.h5", key="df")
            y_pred.to_hdf(version_folder + "y_pred_links.h5", key="df")
            X_test_full.to_hdf(version_folder + "X_test_full_links.h5", key="df")
            X_test_day_ahead.to_hdf(
                version_folder + "X_test_day_ahead_links.h5", key="df"
            )
            X_test_indicators.to_hdf(
                version_folder + "X_test_indicators_links.h5", key="df"
            )

    for area_name, areas in combined_areas.items():
        for target in area_links[area_name]:
            print("Processing external features from", area_name)

            version_folder = (
                "../data/flow_input/{}/{}/".format(area_name, target)
                + "version_"
                + pd.Timestamp("today").strftime("%Y-%m-%d")
                + "/"
            )
            if not os.path.exists(version_folder):
                os.makedirs(version_folder)

            # Load actual and forecast (day-ahead available) data
            X_actual_1 = pd.read_hdf(
                "../data/{}/".format(areas[0]) + "input_actual.h5"
            ).filter(
                regex="(?<!_flow)$(?<!_flow_ramp)$(?<!_total_ramp)$(?<!_import_export_total)$"
            )
            X_actual_2 = (
                pd.read_hdf("../data/{}/".format(areas[1]) + "input_actual.h5")
                .filter(
                    regex="(?<!_flow)$(?<!_flow_ramp)$(?<!_total_ramp)$(?<!_import_export_total)$"
                )
                .filter(regex="^(?!{})".format(areas[0]))
            )

            X_actual = pd.merge(
                X_actual_1,
                X_actual_2,
                how="outer",
                suffixes=map(lambda x: "_" + x, areas),
                left_index=True,
                right_index=True,
            )

            X_forecast_1 = pd.read_hdf(
                "../data/{}/".format(areas[0]) + "input_forecast.h5"
            )
            X_forecast_2 = (
                pd.read_hdf("../data/{}/".format(areas[1]) + "input_forecast.h5")
                .filter(regex="^(?!{})".format(areas[0]))
                .drop(columns=["hour", "weekday", "month"])
            )

            X_forecast = pd.merge(
                X_forecast_1,
                X_forecast_2,
                how="outer",
                suffixes=map(lambda x: "_" + x, areas),
                left_index=True,
                right_index=True,
            )
            X_forecast = X_forecast.join(
                pd.read_pickle("../data/HVDClinks/scheduled_flows.pkl")[target]
            )

            X_indicators_1 = pd.read_hdf(
                "../data/{}/".format(areas[0]) + "indicators.h5"
            )
            X_indicators_2 = pd.read_hdf(
                "../data/{}/".format(areas[1]) + "indicators.h5"
            )

            X_indicator = pd.merge(
                X_indicators_1,
                X_indicators_2,
                how="outer",
                suffixes=map(lambda x: "_" + x, areas),
                left_index=True,
                right_index=True,
            )

            y = pd.read_pickle("../data/HVDClinks/unscheduled_flows.pkl").loc[:, target]

            # Drop nan values
            valid_ind = (
                ~pd.concat([X_forecast, X_actual, X_indicator, y], axis=1)
                .isnull()
                .any(axis=1)
            )
            X_forecast, X_actual, X_indicator, y = (
                X_forecast[valid_ind],
                X_actual[valid_ind],
                X_indicator[valid_ind],
                y[valid_ind],
            )

            # Join features for full model
            X_full = X_actual.join(X_forecast).join(X_indicator)

            # Train-test split
            (
                X_train_full,
                X_test_full,
                y_train,
                y_test,
            ) = model_selection.train_test_split(
                X_full, y, test_size=0.2, random_state=42
            )
            X_train_day_ahead = X_forecast.loc[X_train_full.index]
            X_test_day_ahead = X_forecast.loc[X_test_full.index]
            X_train_indicators = X_indicator.loc[X_train_full.index]
            X_test_indicators = X_indicator.loc[X_test_full.index]
            y_pred = pd.DataFrame(index=y_test.index)

            # Save data for full model and restricted models
            X_train_full.to_hdf(version_folder + "X_train_full_links.h5", key="df")
            X_train_day_ahead.to_hdf(
                version_folder + "X_train_day_ahead_links.h5", key="df"
            )
            X_train_indicators.to_hdf(
                version_folder + "X_train_indicators_links.h5", key="df"
            )
            y_train.to_hdf(version_folder + "y_train_links.h5", key="df")
            y_test.to_hdf(version_folder + "y_test_links.h5", key="df")
            y_pred.to_hdf(version_folder + "y_pred_links.h5", key="df")
            X_test_full.to_hdf(version_folder + "X_test_full_links.h5", key="df")
            X_test_day_ahead.to_hdf(
                version_folder + "X_test_day_ahead_links.h5", key="df"
            )
            X_test_indicators.to_hdf(
                version_folder + "X_test_indicators_links.h5", key="df"
            )


if __name__ == "__main__":
    main()
