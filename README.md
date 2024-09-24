# Memento postproc

From raw time series to matrices

## Preprocessing
[fMRIprep 23.2.0](https://fmriprep.org/en/stable/)
was used to preprocessed data on pipeau cluster.
It's called from the `fmriprep.slurm` script.
You first need to download fMRIprep in a singularity
container, as well as a freesurfer license, and
modify paths in the script accordingly.
Next, you can launch preprocessing as a job array :

```sbatch --job-name fmriprep-NLin6 --array=1-2184 fmriprep.slurm```


## Parcellation
```python
python bold_dementia/cache_time_series.py schaefer200
```

## Create maps
Write a yaml config file specifying cleaning operations and the kind
of matrix you want, following the model 
of `create_maps_config/schaefer200_prediction.yaml`.
Then you can call the script, giving the config path as first argument :
```python
python bold_dementia/create_maps.py create_maps_config/schaefer200_prediction.yaml
```
The result is stored in the `output_dir`
specified in the root level config file `config.yml`.

## Null distributions
**TODO**