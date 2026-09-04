from functools import cached_property
from pathlib import Path
from typing import Tuple, Any, Union

import h5py
import torch
from torch import Tensor as T

from naturallab.utils.misc import PathLike


class H5Container:
    def __init__(
            self,
            h5_file: PathLike,
    ):
        self._hdf_fpath = Path(h5_file)
        self._hdf_file = None
        self.open()

    @cached_property
    def keys(self) -> Tuple[str, ...]:
        if self.is_open():
            return tuple(self._hdf_file.keys())
        else:
            raise RuntimeError("Cannot get keys from a closed HDF file.")

    def is_open(self) -> bool:
        return self._hdf_file is not None and bool(self._hdf_file)

    def open(self) -> None:
        if not self._hdf_fpath.exists():
            raise FileNotFoundError(f"{self._hdf_fpath} does not exist.")
        if not self._hdf_fpath.is_file() or self._hdf_fpath.suffix != ".h5":
            raise ValueError(f"{self._hdf_fpath} is not a valid HDF file.")
        if self.is_open():
            raise RuntimeError("Cannot open the HDF file twice.")
        self._hdf_file = h5py.File(self._hdf_fpath, "r")

    def close(self) -> None:
        if not self.is_open():
            raise RuntimeError("Cannot close the HDF file twice.")
        self._hdf_file.close()
        self._hdf_file = None

    def __del__(self) -> None:
        if self.is_open():
            self.close()

    def __getitem__(self, key: str) -> Any:
        return self._hdf_file[key]


def agg_prototypes(proto_container: H5Container, agg: str = "mean") -> Union[T, T]:
    """
    Aggregate the prototypes in the container.

    Args:
        proto_container: The container with the prototypes.
        agg: The aggregation method. Can be one of "mean", "median", "max", "min", "sum", or an integer. If None,
        all views are returned.

    Returns:
        The aggregated prototypes and the categories.
    """
    prototypes = []
    categories = []
    for cat in proto_container.keys:
        proto = torch.from_numpy(proto_container[cat][:])  # shape: (n_prototypes, dim)
        if agg == "mean":
            proto = proto.mean(dim=0)
        elif agg == "median":
            proto = proto.median(dim=0)
        elif agg == "max":
            proto = proto.max(dim=0)[0]
        elif agg == "min":
            proto = proto.min(dim=0)
        elif agg == "sum":
            proto = proto.sum(dim=0)
        elif isinstance(agg, int):
            proto = proto[agg]
        elif agg is None:
            cat = [cat] * proto.size(0)

        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

        prototypes.append(proto)
        categories.append(cat)
    if agg is None:
        prototypes = torch.cat(prototypes, dim=0)
        categories = [
            x
            for xs in categories
            for x in xs
        ]
    else:
        prototypes = torch.stack(prototypes)
    return prototypes, categories
