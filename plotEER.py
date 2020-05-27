import os
import tempfile
from pathlib import Path
from typing import Union

import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot
import numpy as np
import fire
import time
import math

"""
Required Data = DataFrame({"id1", "predict", "score"})
"""


def calculate_FAR_FRR_each_thresh(
    data: DataFrame,  # id, is_actual, predict (not require), score
    list_thresh: Union[list, np.array] = np.linspace(0, 1, 999), # [start, end] range
    column_names=["base", "enroll", "score"]
):
    """
    Search a FRR index and value given a target FAR
    :param data:
    :param list_thresh:
    :param column_names:
    :return:
    """
    # generate threshold range for a given range
    if type(list_thresh) is list:
        list_thresh = np.linspace(list_thresh[0], list_thresh[1], 999)

    # apply polymorph to data
    if type(data) is str or type(data) is Path:
        data = pd.read_csv(data)

    # list of calculating values
    TPs = []
    TNs = []
    FPs = []
    FNs = []
    FARs = []
    FRRs = []

    # change data type to increase speed
    np_data = data.values

    # define intuitive variable
    genuine_attempt = (np_data[:, 0] == np_data[:, 1])
    imposter_attempt = (np_data[:, 0] != np_data[:, 1])
    scores = np_data[:, 2]
    for thresh in list_thresh:
        TP = (genuine_attempt & (scores > thresh)).sum()
        TN = (imposter_attempt & (scores <= thresh)).sum()

        FP = (imposter_attempt & (scores > thresh)).sum()
        FN = (genuine_attempt & (scores <= thresh)).sum()

        # appending results
        TPs += [TP]
        TNs += [TN]
        FPs += [FP]
        FNs += [FN]
        FARs += [FP / (FP + TN)]
        FRRs += [FN / (FN + TP)]
        # print(TP, TN, FP, FN, FP / (FP + TN), FN / (FN + TP))

    # parse result to DataFrame
    far_frr_data = DataFrame({
        "thresh": list_thresh,
        "TP": TPs, "TN": TNs, "FP": FPs, "FN": FNs,
        "FAR": FARs,
        "FRR": FRRs
    })
    return far_frr_data


def search_FRR(
    target_FAR: float = None,
    target_upper_bound: float = None, # a threshold in range 0.0 - 1.0 for confidence level 0 - 100%
    far_frr_table: DataFrame = None  # id, is_actual, predict (not require), score
):
    """
    Search a FRR index and value given a target FAR
    :param far_frr_table:
    :param target_FAR:
    :return:
    """
    if type(far_frr_table) is not DataFrame:
        raise ValueError(f"Given far_frr_table is {type(far_frr_table)}")

    if target_FAR is not None:
        round_estimate_idx = far_frr_table[far_frr_table["FAR"] > target_FAR].shape[0]
        bound_idx = np.clip(round_estimate_idx, a_min=0, a_max=far_frr_table.shape[0]-1)
        return bound_idx, far_frr_table.iloc[bound_idx]["FRR"]
    elif target_upper_bound is not None:
        round_estimate_idx = far_frr_table[far_frr_table["thresh"] < target_upper_bound].shape[0]
        bound_idx = np.clip(round_estimate_idx, a_min=0, a_max=far_frr_table.shape[0] - 1)
        return bound_idx, far_frr_table.iloc[bound_idx]["FRR"]
    elif target_FAR is not None and target_upper_bound is not None:
        raise ValueError("Cannot process because both target_FAR and target_upper_bound is given.")
    else:
        raise ValueError("Cannot process because no target_FAR or target_upper_bound is given.")



def get_thresh_data(
    data: Union[DataFrame,str,Path],
    file_name: str = None,
    list_thresh: Union[list, np.array] = np.linspace(0, 1, 999), # [start, end] range
    column_names=["Actual", "Prediction", "Score"],
):
    if type(list_thresh) is list:
        list_thresh= np.linspace(list_thresh[0], list_thresh[1], 999)

    if type(data) is str or type(data) is Path:
        data = pd.read_csv(data)

    far_frr_table = calculate_FAR_FRR_each_thresh(
        data,  # id, is_actual, predict (not require), score
        list_thresh=list_thresh,
        column_names=column_names
    )

    if file_name is None:
        return far_frr_table.to_csv()
    elif file_name is not None and type(file_name) is str:
        far_frr_table.to_csv(file_name, index=False)


def plotEERv2(
    data: Union[DataFrame,str,Path],
    graph_name,
    list_thresh: Union[list, np.array] = np.linspace(0, 1, 999), # [start, end] range
    column_names=["Actual", "Prediction", "Score"],
    list_target_FAR=[0.01, 0.001],
    target_upper_bound=None,
    list_target_color=['b', 'k']
):
    """

    :param data:
    :param list_thresh:
    :param column_names:
    :param list_target_FAR:
    :param list_target_color:
    :return:
    """
    if type(list_thresh) is list:
        list_thresh= np.linspace(list_thresh[0], list_thresh[1], 999)

    if type(data) is str or type(data) is Path:
        data = pd.read_csv(data)

    far_frr_table = calculate_FAR_FRR_each_thresh(
        data,  # id, is_actual, predict (not require), score
        list_thresh=list_thresh,
        column_names=column_names
    )
    results = []

    pyplot.plot(list_thresh, far_frr_table["FAR"], color='r', label='FAR')
    pyplot.plot(list_thresh, far_frr_table["FRR"], color='b', label='FRR')

    for target_FAR in list_target_FAR:
        idx, frr = search_FRR(
            target_FAR=target_FAR,
            far_frr_table=far_frr_table
        )
        threshold = list_thresh[idx]
        # [threshold, target FAR, FRR]
        results += [[threshold, target_FAR, frr]]

        label = f"FRR = {frr:.5}\nFAR = {target_FAR:.5}\nat TH = {threshold:.5}"
        pyplot.axvline(x=threshold, color='k', ls='--', label=label)

    if target_upper_bound is not None:
        idx, frr = search_FRR(
            target_upper_bound=target_upper_bound,
            far_frr_table=far_frr_table
        )
        far = far_frr_table.iloc[idx]["FAR"]
        frr = far_frr_table.iloc[idx]["FRR"]
        threshold = list_thresh[idx]
        label = f"FRR = {frr:.5}\nFAR = {far:.5}\nat Upper bound of {threshold:.5}"
        pyplot.axhline(y=far, color='violet', ls='--', label=label)

    pyplot.legend(loc="upper right")

    pyplot.title(graph_name)
    pyplot.xlabel("Threshold")
    pyplot.ylabel("Error Rate")
    pyplot.savefig(graph_name + ".png")
    return results


def testPlotEER():
    data = pd.read_csv("./test_eer_data.csv")
    far_frr_table = calculate_FAR_FRR_each_thresh(
        data,  # id, is_actual, predict (not require), score
        list_thresh=np.linspace(-1, 1, 999),
        column_names=["Actual", "Prediction", "Score"]
    )
    res = plotEERv2(
        data,
        "TestNewAPI",
        list_thresh=np.linspace(-1, 1, 999),
        column_names=["Actual", "Prediction", "Score"],
        list_target_FAR=[0.15, 0.1],
        list_target_color=['b', 'k']
    )

    np.testing.assert_almost_equal(
        res, [[0.9959919839679356, 0.15, 0.8198198198198198], [0.9979959919839678, 0.1, 0.8828828828828829]],
        decimal=3
    )


if __name__ == "__main__":
    fire.Fire({
        "eer": plotEERv2,
        "get_thresh_data": get_thresh_data
    })

