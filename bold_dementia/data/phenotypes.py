
from datetime import datetime
import pandas as pd

def timedelta_to_years(td):
    return td.days / 365

def days_to_onset(
    reference: pd.Series, demence_dat: pd.Series
) -> pd.Series:
    timedelta = demence_dat - reference
    return timedelta.map(lambda x: x.days)
    