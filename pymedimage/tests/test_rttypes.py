# coding: utf-8

# test_rttypes.py
# -*- coding: utf-8 -*-
"""unittest tests for pymedimage.rttypes module"""

import os
import sys
import tempfile
import unittest
import random
import string
import warnings

from pymedimage.data import get_testdata_files
import dicom
import h5py
import pymedimage.rttypes as rttypes
try:
    unittest.skipUnless
except AttributeError:
    try:
        import unittest2 as unittest
    except ImportError:
        print("unittest2 is required for testing in python2.6")

try:
    import numpy  # NOQA
except ImportError:
    numpy = None

try:
    from PIL import Image as PILImg
except ImportError:
    # If that failed, try the alternate import syntax for PIL.
    try:
        import Image as PILImg
    except ImportError:
        # Neither worked, so it's likely not installed.
        PILImg = None

have_numpy = numpy is not None
have_pillow = PILImg is not None

rtplan_name = get_testdata_files("rtplan.dcm")[0]
rtdose_name = get_testdata_files("rtdose.dcm")[0]
ct_name = get_testdata_files("CT_small.dcm")[0]
ct_hdf5_name = get_testdata_files("CT_1slice.h5")[0]
mr_name = get_testdata_files("MR_small.dcm")[0]
rtstruct_name = get_testdata_files("rtstruct.dcm")[0]
rtstruct_hdf5_name = get_testdata_files("rtstruct.h5")[0]
gzip_name = get_testdata_files("zipMR.gz")[0]
emri_name = get_testdata_files("emri_small.dcm")[0]
dir_name = os.path.dirname(sys.argv[0])
save_dir = os.getcwd()


def isClose(a, b, epsilon=0.000001):
    """Compare within some tolerance, to avoid machine roundoff differences"""
    try:
        a.append  # see if is a list
    except BaseException:  # (is not)
        return abs(a - b) < epsilon
    else:
        if len(a) != len(b):
            return False
        for ai, bi in zip(a, b):
            if abs(ai - bi) > epsilon:
                return False
        return True

def random_string(length=40):
    pool = string.ascii_letters + string.digits
    return ''.join(random.choice(pool) for i in range(length))

def random_file_path(length=40, extension=''):
    n = random_string(length)
    if len(extension): n = '.'.join(n, extension)
    return os.path.join(tempfile.gettempdir(), n)

class ExtendedTestCase(unittest.TestCase):
    def assertTupleAlmostEqual(self, tupleA, tupleB, places=None, delta=None):
        if len(tupleA) != len(tupleB):
            raise AssertionError("Tuples of unequal sizes provided: lengths: {:d} & {:d}".format(len(tupleA), len(tupleB)))
            for a, b in zip(tupleA, tupleB):
                self.assertAlmostEqual(a, b, places=places, delta=delta)


class ROITests(ExtendedTestCase):
    def test_toHDF5(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            roi = rttypes.ROI.collectionFromFile(rtstruct_name)['patient']

        # write to hdf5
        p = random_file_path() + '.h5'
        roi.toHDF5(p)
        with h5py.File(p, 'r') as f:
            a = f.attrs
            self.assertEqual(a['refforuid'], '1.2.826.0.1.3680043.8.498.2010020400001.2')
            self.assertEqual(a['roiname'], 'patient')
            self.assertEqual(a['roinumber'], 1)
            # test first 3 tuples in coordslices
            N = 3
            slices = f['coordslices']
            for i, k in enumerate(sorted(slices.keys())):
                data = slices[k]
                npdata = numpy.empty((N, 3), dtype=data.dtype)
                data.read_direct(npdata, numpy.s_[:N, :])
                npdata = npdata.flatten()
                for j in range(N*3):
                    self.assertAlmostEqual(npdata[j], roi.coordslices[i][int(j/3)][j%3])

    def test_fromHDF5(self):
        roi = rttypes.ROI.fromHDF5(rtstruct_hdf5_name)
        with h5py.File(rtstruct_hdf5_name, 'r') as f:
            a = f.attrs
            self.assertEqual(roi.refforuid, a['refforuid'])
            self.assertEqual(roi.roiname, a['roiname'])
            self.assertEqual(roi.roinumber, a['roinumber'])

            N = 3
            slices = f['coordslices']
            for i, k in enumerate(sorted(slices.keys())):
                data = slices[k]
                npdata = numpy.empty((N, 3), dtype=data.dtype)
                data.read_direct(npdata, numpy.s_[:N, :])
                npdata = npdata.flatten()
                for j in range(N*3):
                    self.assertAlmostEqual(npdata[j], roi.coordslices[i][int(j/3)][j%3])


class BaseVolumeTests(ExtendedTestCase):
    def test_toHDF5(self):
        vol = rttypes.BaseVolume.fromDatasetList([dicom.read_file(ct_name)])

        self.assertEqual(vol.modality, 'CT')
        # write to hdf5
        p = random_file_path() + '.h5'
        vol.toHDF5(p)
        with h5py.File(p, 'r') as f:
            a = f.attrs
            self.assertEqual(a['modality'], 'CT')
            self.assertTupleEqual(tuple(a['size'])[::-1], (128,128,1))
            self.assertTupleAlmostEqual(tuple(a['spacing'][::-1]), (0.661, 0.661, 5.000), places=3)
            self.assertTupleAlmostEqual(tuple(a['start'])[::-1], (-158.136, -179.036, -75.700))
            self.assertEqual(a['scale'], 1)
            self.assertEqual(a['offset'], -1024)
            data = f['arraydata']
            self.assertTupleEqual(data.shape, (1, 128, 128))
            testarr = numpy.array(
                [[-849., -844., -858.],
                 [-838., -841., -867.],
                 [-840., -844., -853.]]
            )
            npdata = numpy.empty((3, 3), dtype=data.dtype)
            data.read_direct(npdata, numpy.s_[0, :3, :3])
            self.assertListEqual(testarr.flatten().tolist(), npdata.flatten().tolist())

    def test_fromHDF5(self):
        vol = rttypes.BaseVolume.fromHDF5(ct_hdf5_name)
        self.assertEqual(vol.modality, 'CT')
        vfor = vol.frameofreference
        self.assertTupleEqual(vfor.size, (128,128,1))
        self.assertTupleAlmostEqual(vfor.spacing, (0.661, 0.661, 5.000), places=3)
        self.assertTupleAlmostEqual(vfor.start, (-158.136, -179.036, -75.700), places=3)
        self.assertEqual(vol.rescaleparams.scale, 1)
        self.assertEqual(vol.rescaleparams.offset, -1024)
        self.assertTupleEqual(vol.array.shape, (1, 128, 128))
        testarr = numpy.array(
            [[-849., -844., -858.],
             [-838., -841., -867.],
             [-840., -844., -853.]]
        )
        self.assertListEqual(testarr.flatten().tolist(), vol.array[0, :3, :3].flatten().tolist())


if __name__ == "__main__":
    # This is called if run alone, but not if loaded through run_tests.py
    # If not run from the directory where the sample images are, then need
    # to switch there
    unittest.main()
