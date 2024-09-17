from typing import List
from pathlib import Path
import pandas as pd
import nibabel as nib
import joblib
import os
import json
import math

from nilearn.interfaces.bids import get_bids_files, parse_bids_filename
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds

from bold_dementia.data.phenotypes import days_to_onset, timedelta_to_years
from neuroginius.atlas import Atlas

session_mapping = {
    "IRM_M0": "M000",
    "IRM_M24": "M024",
    "IRM_M48": "M048"
}

from joblib import Memory, Parallel, delayed
memory = Memory("/tmp/Memento/dataset_cache")

def all_subs(row):
    return True

def delete_me(row):
    return False

def converter(row):
    return not pd.isnull(row.DEMENCE_DAT)

def past_diag_AD(row):
    return past_diag(row) and row.MA != 0

def past_diag(row):
    return row.scan_to_onset <= 0

def control(row):
    return math.isnan(row.scan_to_onset)

def non_demented(row):
    return row.scan_to_onset < 0

def control_M000(row):
    return control(row) and row.ses == "M000"

def converter_M000(row):
    return converter(row) and row.ses == "M000"

# TODO Type annotations
# TODO Custom target func like MementoTS
class Memento:
    def __init__(
        self,
        bids_path,
        phenotypes_path,
        atlas:Atlas=None,
        cache_dir="dataset_cache",
        clean_signal=False,
        **confounds_kw
    ):
        
        if atlas is None:
            atlas = Atlas.from_name("harvard-oxford", soft=False)
        self.atlas = atlas
        self.masker = atlas.fit_masker()

        self.scans_ = self.index_bids(bids_path)
        self.phenotypes_ = self.load_phenotypes(phenotypes_path)
        self.rest_dataset = self.make_rest_dataset(self.scans_, self.phenotypes_)
        

        self.cache_dir = Path(cache_dir)
        
        self.clean_signal = clean_signal
        self.confounds_kw = confounds_kw

    
    @staticmethod
    def make_rest_dataset(scans, phenotypes):
        rest_dataset = pd.merge(
            left=scans,
            right=phenotypes,
            how="left",
            on="sub"
        )
        rest_dataset = rest_dataset.dropna(axis=0, subset="CEN_ANOM")
        current_scan = rest_dataset.apply(lambda row: row[row.ses], axis=1)
        
        tdelta = current_scan - rest_dataset["INCCONSDAT_D"]
        rest_dataset["current_scan_age"] = tdelta.map(timedelta_to_years) + rest_dataset["AGE_CONS"]
        
        rest_dataset["scan_to_onset"] = days_to_onset(current_scan, rest_dataset["DEMENCE_DAT"])
        return rest_dataset.reset_index(drop=True)
        
    
    @staticmethod
    def load_phenotypes(ppath, date_format=None, timepoints=None, **loading_kwargs):
        # No more cases please
        if "index_col" not in loading_kwargs.keys():
            loading_kwargs["index_col"] = 0
        phenotypes = pd.read_csv(
            ppath,
            **loading_kwargs
        )
        if timepoints is None:
            timepoints = {"M000", "M024", "M048"}

        # I don't trust date parsing in read_csv
        for date_col in {
                "DEMENCE_DAT",
                "INCCONSDAT_D",
            }.union(timepoints):
            phenotypes[date_col] = pd.to_datetime(
                phenotypes[date_col],
                format=date_format
            )

        phenotypes["sub"] = phenotypes["NUM_ID"].map(lambda x: x[4:])
        return phenotypes.rename(columns=session_mapping)

    @staticmethod
    @memory.cache
    def index_bids(bids_dir):
        fmri_paths = get_bids_files(
        bids_dir / "derivatives/fmriprep-23.2.0",
        "bold",
        file_type="nii.gz",
        filters=[
            ("space", "MNI152NLin6Asym"),
            ],
        )
        scans = pd.DataFrame(map(parse_bids_filename, fmri_paths))

        return scans

    def _extract_ts(self, idx):
        fmri_path = self.rest_dataset["file_path"].iloc[idx]
        
        # For some reason passing the path
        # is slower than loading the image
        # using nibabel, although the masker
        # accepts both
        fmri = nib.load(fmri_path)

        if self.clean_signal:
            confounds, sample_mask = load_confounds(
                fmri_path,
                **self.confounds_kw
            )
        else:
            confounds, sample_mask = None, None

        time_series = self.masker.transform(
            fmri,
            confounds=confounds,
            sample_mask=sample_mask
        )
        return time_series
    
    def __getitem__(self, idx):
        if idx > len(self):
            raise KeyError()
        time_series = self._extract_ts(idx)
        return time_series, self.is_demented(idx)

    def __len__(self):
        return len(self.rest_dataset)

    # TODO Custom is_demented
    def is_demented(self, idx):
        return self.rest_dataset["scan_to_onset"].iloc[idx] <= 0

    def _cache_metadata(self):
        metadata = {
                    "atlas": self.atlas.name,
                    "is_cleaned": self.clean_signal,
                    "confounds_kw": self.confounds_kw
                }
        with open(self.cache_dir / "metadata.json", "w") as stream:
            json.dump(metadata, stream)

    def cache_series(self):
        if not os.path.exists(self.cache_dir / "time_series"):
            os.makedirs(self.cache_dir / "time_series")
            
        self.rest_dataset.to_csv(f"{self.cache_dir}/phenotypes.csv")
        self._cache_metadata()

        for idx, row in self.rest_dataset.iterrows():
            fpath = f"{self.cache_dir}/time_series/{row.file_basename}"
            print(row.file_basename)
            
            # We do not overwrite, if you want a new cache
            # delete the former one yourself
            if not os.path.exists(fpath):
                ts = self._extract_ts(idx)
                joblib.dump(ts, fpath)

    def _parallel_fetch(self, idx):
        row = self.rest_dataset.iloc[idx]
        fpath = f"{self.cache_dir}/time_series/{row.file_basename}"
        if not os.path.exists(fpath):
            return fpath, self._extract_ts(idx)
        else:
            return fpath, None
    
    def parallel_caching(self, n_jobs=8):
        if not os.path.exists(self.cache_dir / "time_series"):
            os.makedirs(self.cache_dir / "time_series")

        self.rest_dataset.to_csv(f"{self.cache_dir}/phenotypes.csv")
        self._cache_metadata()

        parallel = Parallel(n_jobs=n_jobs, verbose=5, return_as="generator")
        calls = [delayed(self._parallel_fetch)(idx) for idx in self.rest_dataset.index]
        for fpath, ts in parallel(calls):
            print(fpath, end=" ... ")
            if ts is None:
                print("Already stored, skipping")
            else:
                joblib.dump(ts, fpath)
                print("Stored")

        
# TODO Mapping from (subject, ses) to ts
class MementoTS(Memento):
    def __init__(self, cache_dir="dataset_cache", target_func=None):
        self.cache = Path(cache_dir)
        self.rest_dataset = pd.read_csv(
            self.cache / "phenotypes.csv",
            index_col=0,
            parse_dates=True,
            date_format="%Y/%m/%d",
        )

        with open(self.cache / "metadata.json", "r") as stream:
            config = json.load(stream)
        for k, w in config.items():
            setattr(self, k, w)

        if target_func is None:
            self.target_func = past_diag
        else:
            self.target_func = target_func

    def __getitem__(self, idx):
        row = self.rest_dataset.iloc[idx, :]
        ts = joblib.load(self.cache / f"time_series/{row.file_basename}")
        return ts, self.target_func(row), row.file_path

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        try:
            item = self.__getitem__(self.counter)
        except IndexError:
            raise StopIteration()
        except FileNotFoundError:
            print("Caching uncomplete, stop iterations")
            raise StopIteration()
        self.counter += 1
        return item