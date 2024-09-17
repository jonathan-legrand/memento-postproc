import warnings
from pathlib import Path
import os
import json
import joblib

import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.linalg as npl
import math
from statsmodels.stats.multitest import fdrcorrection

from sklearn.utils import Bunch
from sklearn import covariance
from nilearn.connectome import ConnectivityMeasure

from nilearn import plotting

from bold_dementia.data.study import balance_control, balance_control_cat, load_signals
from bold_dementia.data.memento import (
    MementoTS, past_diag_AD, control_M000, converter_M000, converter, control, all_subs, delete_me
)
from neuroginius.atlas import Atlas
from bold_dementia.connectivity.matrices import plot_matrices, reshape_pvalues
from bold_dementia import get_config
from bold_dementia.utils.saving import save_run


config = get_config()

from nilearn.connectome import GroupSparseCovariance

def compute_cov_prec(time_series, kind="covariance"):

    if kind == "group":
        gsc = GroupSparseCovariance(
            alpha=0.01,
            verbose=5,
            max_iter=5,
            memory="/tmp/memento_cache",
            memory_level=2
        )
        gsc.fit(time_series)
        c = np.transpose(gsc.covariances_, (2, 0, 1))
        p = np.transpose(gsc.precisions_, (2, 0, 1))
    else:
        print(f"Computing {kind} measure...", end=" ")
        pipe = ConnectivityMeasure(
            covariance.LedoitWolf(),
            kind=kind
        )
        c = pipe.fit_transform(time_series)
        p = npl.inv(c)

    bunch = Bunch(
        covariances_=c,
        precisions_=p
    )
    print("Done")
    return bunch


# TODO Simplify matrix calculation for the passation
# (=> creating a new project?) and ensure that fMRIprep's
# default thresh of 0.5mm FD is respected (vilain nilearn)
def create_maps(run_config):
    try:
        atlas = Atlas.from_name(
            run_config["ATLAS"],
            run_config["SOFT"]
        )
    except KeyError:
            i_path = Path(config["parcellations"]) / run_config["ATLAS"]
            atlas = Atlas.from_path(i_path, run_config["SOFT"])
    try:
        cache_dir = run_config["cache_dir"]
    except KeyError:
        cache_dir = Path(config["bids_dir"]) / "derivatives" / f"{atlas.name}"
    print(f"Fetching time series in {cache_dir}")
    memento = MementoTS(cache_dir=cache_dir, target_func=lambda row: row)
    posfunc_name = run_config["posfunc"] if "posfunc" in run_config.keys() else "converter"
    negfunc_name = run_config["negfunc"] if "negfunc" in run_config.keys() else "control"
    print(f"Selecting with {posfunc_name} and {negfunc_name}")
    with warnings.catch_warnings(category=FutureWarning, action="ignore"):
        AD_signals_ub, control_signals_ub, pm, nm = load_signals(
            memento,
            eval(posfunc_name),
            eval(negfunc_name),
            clean_signal=run_config["CLEAN_SIGNAL"],
            confounds_strategy=run_config["confounds_strategy"]
        )

    balance_strat = run_config["BALANCE_STRAT"]
    if balance_strat != []:
        print(f"Balancing for {balance_strat}")
        balanced_AD, balanced_meta = pm, nm
        print(balanced_meta)

        if "sub" in balance_strat:
            balanced_AD = balanced_AD.groupby("sub").sample(n=1, random_state=1234)
            balanced_meta = balanced_meta.groupby("sub").sample(n=1, random_state=1234)

        # Balanced age and sex accross groups
        if "AGE" in balance_strat:
            balanced_AD, balanced_meta = balance_control(
                balanced_AD,
                balanced_meta,
                col_name="current_scan_age"
            )
        if "SEX" in balance_strat:
            balanced_AD, balanced_meta = balance_control_cat(
                balanced_AD,
                balanced_meta,
                col_name="SEX"
            )
    else:
        balanced_AD, balanced_meta = pm, nm

    balanced_signals = [control_signals_ub[idx] for idx in balanced_meta.index]
    AD_signals = [AD_signals_ub[idx] for idx in balanced_AD.index]

    time_series = AD_signals + balanced_signals
    AD_indices = list(range(len(AD_signals)))
    control_indices = list(range(len(AD_signals), len(time_series)))

    n = len(time_series)
    print(f"Study on {n} subjects")
    
    kind = run_config["KIND"] if "KIND" in run_config.keys() else "covariance"
    print(f"Computing connectivity with {kind}", end="... ")
    gcov = compute_cov_prec(time_series, kind=kind)
        
    print("Finished, exporting results")

    joblib_export = {
        f"{posfunc_name}.joblib": gcov.covariances_[AD_indices, :, :],
        f"{negfunc_name}.joblib": gcov.covariances_[control_indices, :, :],
        f"{posfunc_name}_prec.joblib": gcov.precisions_[AD_indices, :, :],
        f"{negfunc_name}_prec.joblib": gcov.precisions_[control_indices, :, :],
        f"{posfunc_name}_series_ub.joblib": AD_signals_ub,
        f"{negfunc_name}_series_ub.joblib": control_signals_ub
    }
    csv_export = {
        f"balanced_{posfunc_name}.csv": balanced_AD,
        f"balanced_{negfunc_name}.csv": balanced_meta,
        f"{posfunc_name}_series_ub.csv": pm,
        f"{negfunc_name}_series_ub.csv": nm,
    }

    save_run(run_config, joblib.dump, joblib_export)
    exppath = save_run(run_config, lambda df, fname: df.to_csv(fname), csv_export)
    return exppath

import sys

if __name__ == "__main__":
    try:
        run_config = get_config(sys.argv[1])
        print("Loaded custom config :")
    except IndexError:
        run_config = config["default_run"]
        print("No config path provided, using default :")
    print(run_config)
    
    p = create_maps(run_config)
    print(f"Saved output in {p}")



    


