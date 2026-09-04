from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def pd_read_any(fpath: PathLike, *args, **kwargs) -> pd.DataFrame:
    """
    Read a file in any format supported by pandas.
    """
    fpath = Path(fpath)
    if fpath.suffix == '.csv':
        return pd.read_csv(fpath, *args, **kwargs)
    elif fpath.suffix == '.parquet':
        return pd.read_parquet(fpath, *args, **kwargs)
    else:
        raise ValueError("Unsupported file format.")
