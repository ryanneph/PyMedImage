"""Test the data manager"""

import os
import unittest
from os.path import basename
from pymedimage.data import get_testdata_files

from pymedimage.data.data_manager import DATA_ROOT


class TestGetData(unittest.TestCase):

    def test_get_dataset(self):
        """Test the different functions to get lists of data files."""

        testbase = os.path.join(DATA_ROOT, 'test_files')
        self.assertTrue(os.path.exists(testbase))

        # Test that subdirectory files included
        testdata = get_testdata_files()
        bases = [basename(x) for x in testdata]
        self.assertTrue('2693' in bases)
        self.assertTrue(len(testdata) > 70)

        # The files should be from their respective bases
        [self.assertTrue(testbase in x) for x in testdata]

    def test_get_dataset_pattern(self):
        """Test that pattern is working properly."""

        pattern = 'CT_small'
        filename = get_testdata_files(pattern)
        self.assertTrue(filename[0].endswith('CT_small.dcm'))
