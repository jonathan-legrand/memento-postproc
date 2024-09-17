import pandas as pd

def load_volumes(vpath):
    volumes = pd.read_csv(vpath)
    volumes.loc[:, "subject"] = volumes.loc[:, "subject"].map(lambda x: x.split("-")[-1])
    volumes["sub"] = volumes.loc[:, "subject"].map(lambda x: int(x[4:]))
    return volumes

def add_volumes(df, vpath, raise_on_loss=True):
   volumes = load_volumes(vpath)
   merge = pd.merge(
       df,
       volumes,
       how="inner",
       on="sub"
   )
   if raise_on_loss and len(merge) != len(df):
       diff = len(df) - len(merge)
       raise ValueError(f"{diff} subjects are missing from the volume file.")
   return merge