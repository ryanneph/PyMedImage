import sys
import os
import re
import logging
from numbers import Number
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma

from ..rttypes import MaskableVolume
from .strutils import isFileByExt, matchtype
#  from ..visualgui import multi_slice_viewer as view3d

# initialize module logger
logger = logging.getLogger(__name__)

def loadImageSet(dname, mask_types=None, multichannel=False, exts=None, type_order=None, typegetter=None, asarray=False, resize_factor=None, resample_order=3):
    """load all images in directory and create vector valued (multichannel) image given a specified order"""
    logger.debug('loading images in "{}"'.format(dname))
    fnames = [os.path.join(dname, f) for f in os.listdir(dname) if isFileByExt(f, exts)]


    if isinstance(resize_factor, Number):
        max_resize = 4
        if not (0<resize_factor<=max_resize):
            raise ValueError('resize factor ({}) must be in the range (0, {}]'.format(resize_factor, max_resize))
        if resize_factor != 1.0:
            resize_factor = ([float(resize_factor)]*2)+[1.0]
        else: resize_factor = None
    if resize_factor is not None and not isinstance(resize_factor, list): resize_factor = list(resize_factor)

    def loadvol(fname, mask=None, resample_order=3):
        vol = MaskableVolume.load(fname)
        if isinstance(resize_factor, list):
            vol = vol.resample(zoom_factors=resize_factor, order=resample_order)
        #  view3d(vol.data, cmap='gray')
        if asarray:
            if mask is not None:
                return ma.array(vol.data, mask=mask)
            else: return vol.data
        else: return vol

    if isinstance(type_order, str): type_order = [type_order]
    if not isinstance(type_order, list): type_order = None
    if isinstance(mask_types, str): mask_types = [mask_types]
    if not isinstance(mask_types, list): mask_types = None

    # attempt load mask
    mask = None
    if mask_types:
        for mtype in mask_types:
            for fname in fnames:
                if matchtype(fname, mtype, typegetter):
                    mask = loadvol(fname, resample_order=0).astype('uint8')
                    logger.debug('loaded mask from: "{}"'.format(fname))
                    break
            if mask is not None: break

    # load images from file
    images = []
    image_fnames = []
    for fname in fnames:
        if type_order and not typegetter(fname) in type_order: continue
        if mask is not None and typegetter(fname) in mask_types: continue
        im = loadvol(fname, mask, resample_order)
        images.append(im)
        image_fnames.append(fname)

    # reorder
    if isinstance(type_order, list):
        dim = len(type_order)
        # reorder
        temp_images = []
        temp_fnames = []
        for type in type_order:
            for _im, _f in zip(images, image_fnames):
                if matchtype(_f, type, typegetter=typegetter):
                    temp_images.append(_im)
                    temp_fnames.append(_f)
                    break
        if len(temp_images) < dim:
            logger.error('Failed to find all types')
            return None
        images = temp_images
        image_fnames = temp_fnames
    for _f in image_fnames:
        logger.debug('loaded image from "{}"'.format(_f))

    if multichannel:
        im = np.stack(images)
        logger.debug('created multichannel image with shape: ' + str(im.shape))
    else:
        im = images
        logger.debug('loaded {} images (shape={})'.format(len(images), images[0].shape))

    if mask is not None:
        return (im, mask)
    else:
        return im

def loadImageCollection(root, recursive=True, **kwargs):
    """load a collection of images, each contained in their own directory and loaded as a multichannel image
    array or a list of single-channel images arrays"""
    doi_paths = []
    # collect all full paths to dirs containing medical image files
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for f in filenames:
            if isFileByExt(f, kwargs['exts']):
                doi_paths.append(dirpath)
                break
        if not recursive: break

    images = {}
    for dname in sorted(doi_paths):
        im = loadImageSet(dname, **kwargs)
        if im is not None:
            images[dname.lstrip(root)] = im
            logger.debug('')
    return images
