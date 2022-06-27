import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import time

import pickle

version_folder = "../data/flow_input/{}/{}/version_{}/"
fit_folder = "../flow_results/model_fit/{}/version_{}/target_{}/"

version = "2022-06-24"

areas = ["GB", "Nordic", "GB-CE", "Nordic-CE"]
area_names = ["GB/IE", "Nordic/Baltic", "GB/CE", "Nordic/Continental Europe"]
area_name = dict(zip(areas, area_names))

targets = {
    "GB": ["EWIC", "Moyle",],
    "Nordic": ["Nordbalt", "Estlink",],
    "GB-CE": ["IFA", "Britned",],
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

links = [
    "Norned",
    "Swepol",
    "Baltic Cable",
    "Storebaelt",
    "Skagerrak",
    "Kontiskan",
    "Kontek",
    "Estlink",
    "Nordbalt",
    "IFA",
    "Britned",
    "EWIC",
    "Moyle",
]

symbols = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "star",
    "pentagon",
    "cross",
    "x",
    "hexagram",
    "hourglass",
    "bowtie",
    "asterisk-open",
]

linksymbols = dict(zip(links, symbols))


def shap_vals_df(area, version, targ, model_type="_full"):
    version_folder = "../data/flow_input/{}/{}/version_{}/"
    fit_folder = "../flow_results/model_fit/{}/version_{}/target_{}/"
    with open(
        fit_folder.format(area, version, targ)
        + "shap_values_gtb{}.pkl".format(model_type),
        "rb",
    ) as handle:
        sh = pickle.load(handle)
    X_test = pd.read_hdf(
        version_folder.format(area, targ, version)
        + "X_test{}_links.h5".format(model_type)
    )
    return pd.DataFrame(data=sh.values, index=X_test.index, columns=sh.feature_names)


def most_important_shap(area, version, targ, n_features=9, model_type="_full"):
    return list(
        shap_vals_df(area, version, targ, model_type)
        .abs()
        .mean()
        .sort_values(ascending=False)
        .index[:n_features]
    )

