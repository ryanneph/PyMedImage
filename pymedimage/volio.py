import sys
import os
from os.path import join, exists
import struct
from abc import ABCMeta, abstractmethod
import logging

import numpy as np
from scipy.io import loadmat, whosmat
import h5py

logger = logging.getLogger(__name__)

class BaseDataProvider:
    __metaclass__ = ABCMeta
    def __init__(self):
        self._cached_image = None
        self._cached_image_path = None

    def _check_cached(self, filepath):
        if (self._cached_image is not None and filepath == self._cached_image_path):
            return True
        else: return False

    @abstractmethod
    def _load_file(self, filepath):
        return

    def load(self, filepath, size=None):
        if (self._load_file(filepath, size)):
            return self._cached_image
        else: return None

    def _load_file(self, filepath, size=None):
        try:
            status = False
            if filepath:
                status = True
                if not self._check_cached(filepath):
                    status = False
                    # [re]load data volume from file
                    if not os.path.exists(filepath):
                        raise FileNotFoundError
                    del self._cached_image
                    self._cached_image = self._load_file(filepath, size)
                    if self._cached_image is not None:
                        status = True
                        self._cached_image_path = filepath
        except Exception as e:
            logger.error(e)
            status = False
        return status

    def get_image_slice(self, filepath, slicenum, orientation=0, size=None):
        try:
            if self._load_file(filepath, size=size):
                if self._cached_image is not None:
                    if orientation==0:
                        slice = self._cached_image[slicenum, :, :]
                    elif orientation==1:
                        slice = self._cached_image[:, slicenum, :]
                    else:
                        slice = self._cached_image[:, :, slicenum]
                    return slice
        except Exception as e:
            logger.error(e)

    def get_slice_count(self, filepath, orientation=0, size=None):
        if self._load_file(filepath, size=size):
            return self._cached_image.shape[orientation]
        else: return 0

class ImageDataProvider(BaseDataProvider):
    def __init__(self):
        super().__init__()
        self.valid_exts = set()
        self.loaders = []
        self._add_loader(self._load_from_mat, ['.mat'])
        self._add_loader(self._load_from_legacy_dose_mat, ['.mat'])
        self._add_loader(self._loadFromNpy, ['.npy', '.npz'])
        self._add_loader(self._load_from_h5, ['.h5', '.hdf5', '.dose', '.fmap'])
        self._add_loader(self._load_from_dicom, ['', '.dcm', '.dicom'])
        self._add_loader(self._load_from_bin_with_size, ['', '.bin', '.raw'])
        self._add_loader(self._load_from_cti_bin, ['.cti', '.ctislice', '.seg'])
        self._add_loader(self._load_from_bin, ['', '.bin', '.raw'])

    def _add_loader(self, callable, valid_exts=[]):
        self.loaders.append({"callable": callable, "valid_exts": [str(x).lower() for x in valid_exts]})
        for ext in valid_exts:
            self.valid_exts.add(ext)

    def get_size(self):
        return self._cached_image.frameofreference

    def get_valid_extensions(self):
        return list(self.valid_exts)

    def reset_cache(self):
        self._cached_image = None
        self._cached_image_path = None

    def _load_from_bin_with_size(self, filepath, *args, **kwargs):
        with open(filepath, 'rb') as fd:
            sizebuf = fd.read(struct.calcsize('I'*3))
            databuf = fd.read()
        size = np.array(struct.unpack('I'*3, sizebuf))
        arr = np.array(struct.unpack('f'*np.product(size), databuf)).reshape(size[::-1])
        #  arr = np.transpose(arr, [0, 2, 1])
        return arr

    def _load_from_bin(self, filepath, size, *args, **kwargs):
        if size is None:
            raise ValueError("size must be 3-tuple")
        with open(filepath, 'rb') as fd:
            buf = fd.read()
        except_msgs = []
        for type in ['f', 'd']:
            try:
                arr = np.array(struct.unpack(type*np.product(size), buf)).reshape(size[::-1])
                break
            except Exception as e:
                except_msgs.append(str(e))
                continue
        if arr is None:
            raise Exception("\n".join(except_msgs))
        return arr

    def _load_from_cti_bin(self, filepath, size, *args, **kwargs):
        if size is None:
            raise ValueError("size must be 3-tuple")
        with open(filepath, 'rb') as fd:
            buf = fd.read()
        arr = np.array(struct.unpack('h'*np.product(size), buf)).reshape(size[::-1])
        arr = np.transpose(arr, [0, 2, 1])
        return arr

    def _load_from_h5(self, filepath, *args, **kwargs):
        with h5py.File(filepath, 'r') as fd:
            excepts = []
            for k in ["data", "volume", "arraydata"]:
                try:
                    return fd[k][:]
                except Exception as e:
                    excepts.append(e)
                    continue
            raise Exception('\n'.join(excepts))

    def _load_from_mat(self, filepath, *args, **kwargs):
        # Load from matlab (matrad "cube")
        d = loadmat(filepath)
        return d['ct']['cube'][0,0][0,0].transpose((2,0,1))

    def _load_from_legacy_dose_mat(self, filepath, *args, **kwargs):
        import sparse2dense.recon
        vol = sparse2dense.recon.reconstruct_from_dosecalc_mat(filepath)
        return vol

    def _loadFromNpy(self, filepath, *args, **kwargs):
        data = np.load(filepath)
        if isinstance(data, np.ndarray):
            return data
        else:
            return next(iter(data.values()))

    def _load_from_dicom(self, filepath, *args, **kwargs):
        import pymedimage.rttypes as rttypes
        if os.path.splitext(filepath)[1] == '':
            if not os.path.isdir(filepath):
                raise TypeError('file must be a directory containing dicom files or a single dicom file')
            return rttypes.BaseVolume.fromDir(filepath).data
        else:
            return rttypes.BaseVolume.fromDicom(filepath).data
        return None

    def _load_file(self, filepath, size=None):
        excepts = []
        attempts = 0
        while attempts < len(self.loaders):
            attempts += 1
            try:
                loader = self.loaders[attempts-1]
                if os.path.splitext(filepath)[1].lower() not in loader['valid_exts']:
                    raise ValueError("file doesn't match valid valid extensions: [{!s}]".format(', '.join(loader['valid_exts'])))
                vol = loader['callable'](filepath, size)
                if vol is not None:
                    return vol
            except Exception as e:
                excepts.append(e)
                self.reset_cace()

        logger.error("Failed to load image with errors:")
        for e in excepts:
            logger.error(e, '\n')
        return None # failed to open
