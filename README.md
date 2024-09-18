# Memento postproc

From raw time series to matrices

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