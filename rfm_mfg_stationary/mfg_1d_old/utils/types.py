import numpy as np
from typing import Protocol

from rfm_mfg_stationary.mfg_1d_old.classes.error_tracker import ErrorTracker1D


# Define a protocol for array-like objects that are sized and support indexing with assignment
class Sized(Protocol):
    def __len__(self) -> int:
        ...

class ErrorArray(Sized, Protocol):
    def __getitem__(self, index: int) -> ErrorTracker1D:
        ...

    def __setitem__(self, index: int, value: ErrorTracker1D) -> None:
        ...

# Create a pre-allocated array of ErrorTracker1D instances
def create_error_array(size: int) -> ErrorArray:
    return np.empty(size, dtype=object)  # type: ErrorArray  # Array of ErrorTracker1D instances
