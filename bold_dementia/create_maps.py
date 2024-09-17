import warnings
from pathlib import Path
import os
import json
import joblib

import pandas as pd
import nibabel as nib
import numpy as np
import numpy.linalg as npl

from sklearn.utils import Bunch
from sklearn import covariance
from nilearn.connectome import ConnectivityMeasure
from bold_dementia.data.study import load_signals
from bold_dementia.data.memento import MementoTS

from neuroginius.atlas import Atlas
from bold_dementia import get_config
from bold_dementia.utils.saving import save_run

config = get_config()


def compute_cov_prec(time_series, kind="covariance"):

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
    atlas = Atlas.from_name(
        run_config["atlas_name"],
    )
    try:
        cache_dir = run_config["cache_dir"]
    except KeyError:
        cache_dir = Path(config["bids_dir"]) / "derivatives" / f"{atlas.name}"

    print(f"Fetching time series in {cache_dir}")

    memento = MementoTS(cache_dir=cache_dir, target_func=lambda row: row)

    with warnings.catch_warnings(category=FutureWarning, action="ignore"):
        time_series, metadata = load_signals(
            memento,
            confounds_strategy=run_config["confounds_strategy"]
        )

    n = len(time_series)
    print(f"Study on {n} subjects")
    
    kind = run_config["kind"] if "kind" in run_config.keys() else "covariance"
    print(f"Computing connectivity with {kind}", end="... ")

    pipe = ConnectivityMeasure(
        covariance.LedoitWolf(),
        kind=kind
    )
    connectivities = pipe.fit_transform(time_series)
        
    print("Finished, exporting results")

    joblib_export = {
        f"connectivities.joblib": connectivities,
        f"connectivity_measure.joblib": pipe,
    }
    csv_export = {
        f"metadata.csv": metadata,
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



    


