import warnings
from pandas import DataFrame
import pandas as pd
from nilearn import signal
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.interfaces.bids import parse_bids_filename
import json
from pathlib import Path


def load_signals(dataset, clean_signal=True, confounds_strategy=None, **clean_kwargs):
    processed_ts, meta = [], []
    for ts, row, fpath in iter(dataset):
        print(f"Processing {fpath}")
        if clean_signal:
            confounds, sample_mask = load_confounds(
                fpath, **confounds_strategy
            )
            cleaned_ts = signal.clean(
                ts,
                sample_mask=sample_mask,
                confounds=confounds,
                standardize="zscore_sample",
                **clean_kwargs
            )
        else:
            cleaned_ts = ts
        processed_ts.append(cleaned_ts)
        meta.append(row)

    meta = DataFrame(meta).reset_index(drop=True)
    return processed_ts, meta
            