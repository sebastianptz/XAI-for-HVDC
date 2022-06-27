import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import time
import shap
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import pickle
import multiprocessing as mp
from itertools import product


def save_shap(area, target):

    data_version = "2022-06-24"

    print(
        "---------------------------- ",
        area,
        "-",
        target,
        " - start ------------------------------------",
    )

    data_folder = "../data/stability_input/{}/version_{}/".format(area, data_version)

    # Result folder where prediction, SHAP values and CV results are saved
    res_folder = "../stability_results/model_fit/{}/version_{}/target_{}/".format(
        area, data_version, target
    )

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    # Load target data
    y_train = pd.read_hdf(data_folder + "y_train.h5").loc[:, target]
    y_test = pd.read_hdf(data_folder + "y_test.h5").loc[:, target]
    y_pred = pd.read_hdf(data_folder + "y_pred.h5")  # contains only time index

    for model_type in ["_full"]:

        # Load feature data
        X_train = pd.read_hdf(data_folder + "X_train{}.h5".format(model_type))
        X_test = pd.read_hdf(data_folder + "X_test{}.h5".format(model_type))

        params_grid = {
            "num_leaves": [10, 30, 200, 900],
            "max_depth": [10, 5],
            "subsample": [1, 0.9, 0.5],
            "learning_rate": [0.001, 0.01, 0.05, 0.1],
            "min_child_samples": [50, 100, 500],
        }

        # Gradient boosting regression best model evaluation on test set
        best_params = pd.read_csv(
            res_folder + "cv_best_params_gtb{}.csv".format(model_type),
            usecols=list(params_grid.keys())
            + ["n_estimators", "min_child_weight", "subsample_freq", "n_jobs"],
        )
        best_params = best_params.to_dict("records")[0]
        best_params["n_jobs"] = 1
        print(
            "Best parameters from GridSearchCV:",
            {
                p_name: best_params[p_name]
                for p_name in list(params_grid.keys()) + ["n_estimators"]
            },
        )

        # Train on whole training set (including validation set)
        model = LGBMRegressor(**best_params)
        model.fit(X_train, y_train)

        # Calculate SHAP values on test set
        if area in ["CE", "Nordic", "GB"]:
            if model_type == "_full":

                print("explaining {} in {} with SHAP".format(target, area))
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer(X_test)

                with open(
                    res_folder + "shap_values_gtb{}.pkl".format(model_type), "wb"
                ) as handle:
                    pickle.dump(shap_vals, handle)

                # shap_interactions = explainer.shap_interaction_values(X_test)

                # with open(res_folder + 'shap_interaction_values_gtb{}.pkl'.format(model_type), 'wb') as handle:
                #     pickle.dump(shap_interactions, handle)

        # Prediction on test set
        y_pred["gtb{}".format(model_type)] = model.predict(X_test)

        # Print performances on test sets
        print(
            "{} - {} - {} - ".format(model_type[1:], area, target),
            "R2 score test set: {}".format(
                r2_score(y_test, y_pred["gtb{}".format(model_type)])
            ),
        )

        # Daily profile prediction
        daily_profile = y_train.groupby(X_train.index.time).mean()
        y_pred["daily_profile"] = [daily_profile[time] for time in X_test.index.time]

        # Mean predictor
        y_pred["mean_predictor"] = y_train.mean()

    y_pred.to_hdf(res_folder + "y_pred.h5", key="df")
    return


def main():
    # Setup
    areas = ["GB", "CE", "Nordic"]
    data_version = "2022-06-24"
    targets = ["f_integral", "f_ext", "f_msd", "f_rocof"]

    start_time = time.time()

    for area in areas:

        print('---------------------------- ', area, ' ------------------------------------')

        data_folder = '../data/stability_input/{}/version_{}/'.format(area, data_version)

        for target in targets:

            print('-------- ', target, ' --------')

            # Result folder where prediction, SHAP values and CV results are saved
            res_folder = '../stability_results/model_fit/{}/version_{}/target_{}/'.format(area, data_version, target)

            if not os.path.exists(res_folder):
                os.makedirs(res_folder)

            # Load target data
            y_train = pd.read_hdf(data_folder+'y_train.h5').loc[:, target]
            y_test = pd.read_hdf(data_folder+'y_test.h5').loc[:, target]
            y_pred = pd.read_hdf(data_folder+'y_pred.h5')  # contains only time index

            for model_type in ['_full']:

                # Load feature data
                X_train = pd.read_hdf(data_folder+'X_train{}.h5'.format(model_type))
                X_test = pd.read_hdf(data_folder+'X_test{}.h5'.format(model_type))

                #### Gradient boosting Regressor CV hyperparameter optimization ###

                # Split training set into (smaller) training set and validation set
                X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                # Parameters for hyper-parameter optimization
                params_grid = {
                    'num_leaves': [10, 30, 200, 900],
                    'max_depth': [10, 5],
                    'subsample': [1, 0.9, 0.5],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                    'min_child_samples': [50, 100, 500]
                }

                fit_params = {
                    'eval_set': [(X_train_train, y_train_train), (X_train_val, y_train_val)],
                    'early_stopping_rounds': 20,
                    'verbose': 0
                }

                # Grid search for optimal hyper-parameters
                grid_search = GridSearchCV(LGBMRegressor(n_estimators=1000, min_child_weight=0, subsample_freq=1, n_jobs=5), params_grid, verbose=1, n_jobs=5, cv=5)
                grid_search.fit(X_train_train, y_train_train, **fit_params)

                # Save CV results
                pd.DataFrame(grid_search.cv_results_).to_csv(res_folder+'cv_results_gtb{}.csv'.format(model_type))

                # Save best params (including n_estimators from early stopping on validation set)
                best_params = grid_search.best_estimator_.get_params()
                best_params['n_estimators'] = grid_search.best_estimator_.best_iteration_
                pd.DataFrame(best_params, index=[0]).to_csv(res_folder+'cv_best_params_gtb{}.csv'.format(model_type))

    pool = mp.Pool(len(areas) * len(targets))
    pool.starmap(save_shap, list(product(areas, targets)))
    pool.close()

    print("Execution time [h]: {}".format((time.time() - start_time) / 3600.0))


if __name__ == "__main__":
    main()
