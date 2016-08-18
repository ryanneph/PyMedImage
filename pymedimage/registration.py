"""registration.py

functions and classes for working with image registation
"""
from utils.rttypes import *
import SimpleITK as sitk

def register_MultiModality(ref, volume_list)
    """performs registration between ref BaseVolume and each BaseVolume in volume_list

    Args:
        ref          -- BaseVolume to which others will be registered
        volume_list  -- list of BaseVolumes that will be registered to ref
        ?HYPERPARAMS?
    """
    ### check for valid inputs
    if (not isinstance(ref, rttypes.BaseVolume)):
        print('ref of type: {:s} is an invalid type. Must be a BaseVolume or a subtype'.format(type(ref)))
        raise TypeError
    if (isinstance(volume_list, list)):
        # check each element in volume_list
        for i, check_volume in enumerate(volume_list):
            if (not isinstance(check_volume, BaseVolume)):
                print('volume[{idx:d}] of type: {type:s} is an invalid type. Must be a BaseVolume or a ' \
                        'subtype'.format(idx=i, type=type(ref)))
                raise TypeError
    else:
        # volume_list was specified as a single volume, not a list holding a single volume
        if (not isinstance(volume_list, rttypes.BaseVolume)):
            print('Volume of type: {:s} is an invalid type. Must be a BaseVolume or a subtype'.format(type(volume_list)))
            raise TypeError

    ### extract image volumes and convert to an ITK friendly type
    ref_array = sitk.GetImageFromArray(ref.vectorize(asmatrix=True))
    print(ref_array.GetSize())
    ### upsample to same resolution/shape?

    ### perform affine registration using Mutual Information similarity metric

    ### inject registration parameters into each BaseVolume, Identity parameters for ref
