"""pydicom data manager

borrowed from the PyDicom project at: https://github.com/pydicom/
"""

from .data_manager import get_testdata_files
from .data_manager import DATA_ROOT

__all__ = ['get_testdata_files']
