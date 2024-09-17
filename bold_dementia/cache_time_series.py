"""
Our model uses average time series from a given parcellation as input features.
Image loading and parcellation take too much time to be integrated
in the training input pipeline, so it is performed before and
cached in a dedicated directory.
"""

from pathlib import Path
import argparse
from data.memento import Memento
from bold_dementia import get_config
from neuroginius.atlas import Atlas

config = get_config()

def compute_and_cache_ts(atlas:Atlas, bids_dir:Path, ppath:Path):
    psuffix = ppath.stem
    
    memento = Memento(
        bids_dir,
        ppath,
        atlas=atlas,
        cache_dir=bids_dir / "derivatives" / (atlas.name + "_" + "soft-" + str(atlas.is_soft) + "_" + psuffix + "DEBUG"),
    )
    if atlas.is_soft:
        print("is_soft is True, default to serial caching")
        memento.cache_series() # For some reason parallel caching is slow with soft atlases
    else:
        print("Using parallel caching")
        memento.parallel_caching(n_jobs=8)
    

# TODO We should configure n_jobs from command line
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute average time series. No signal cleaning at this point.")
    parser.add_argument(
        "atlas_name",
        help="Name of the atlas"
    )
    parser.add_argument(
        "--is_soft",
        help="Whether the parcellation is soft. Most of the times it is not.",
        type=bool,
        default=False
    )
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    try:
        atlas = Atlas.from_name(args.atlas_name, soft=args.is_soft)
    except KeyError:
        print("Loading custom parcellation")
        i_path = Path(config["parcellations"]) / args.atlas_name
        atlas = Atlas.from_path(i_path, soft=args.is_soft)

    ppath = Path(config["data_dir"]) / "merged_phenotypes.csv"
    print(f"Using phenotypes from {ppath}")

    bids_dir = Path(config["bids_dir"])
    
    compute_and_cache_ts(atlas, bids_dir, ppath)