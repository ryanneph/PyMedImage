"""
dcmio.py

handles all dicom input/output tasks including reading dicom files and building
workable datasets for later use in projects
"""

import os
import sys
import pickle
import dicom # pydicom
import numpy as np
from utils.imvector import imvector
from utils import rttypes
from utils import features as features
from utils.logging import print_indent, g_indents
from string import Template


def write_dicom(path, dataset):
    """write a pydicom dataset to dicom file"""
    if not os.path.splitext(path)[1] == '.dcm':
        path += '.dcm'
    dicom.write_file(path, dataset)


def read_dicom(path):
    """read a dicom slice using pydicom and return the dataset object"""
    ds = None
    if (os.path.exists(path)):
        ds = dicom.read_file(path)
    return ds


def read_dicom_dir(path, recursive=False, verbose=0):
    """read all dicom files in directory and return a list of the dataset objects.

    Keyword arguments:
    recursive -- should we walk into subdirs?
    verbose -- 1: low, 2: high, 3: full
    """
    ds_list = []
    dicom_paths = []
    if (not os.path.exists(path)):
        print('Execution Terminated. Supplied path did not exist: {:s}'.format(path))
        sys.exit(1)
    else:
        l1_indent = g_indents[2]
        l2_indent = g_indents[3]
        printstring = Template('Reading dicoms in specified path$extra:\n"{:s}"')
        extra = ''
        if recursive:
            extra = ' and subdirs'
        printstring = printstring.substitute(extra=extra).format(path)
        print_indent(printstring, l1_indent)
        for root, dirs, files in os.walk(path, topdown=True):
            # build the list of valid dicom file paths then load them after walk
            for file in files:
                _, file_extension = os.path.splitext(file)
                if file_extension in ['.dcm', '.dicom']:
                    dicom_paths.append(root + '/' + file)
            if (not recursive):
                # clear dirs so that walk stops after this level
                del dirs[:]
    
        # Now read the dicom files that were located within path
        if verbose == 1:
            #low verbosity
            print_indent(dicom_paths[:5],l2_indent)
        elif verbose == 2:
            #high verbosity
            print_indent(dicom_paths[:20],l2_indent)
        elif verbose > 2:
            #full verbosity
            print_indent(dicom_paths,l2_indent)

        if (len(dicom_paths)>0):
            for file in dicom_paths:
                file_dataset = read_dicom(file)
                if file_dataset is not None:
                    ds_list.append(file_dataset)
            return ds_list
        else:
            return None

def loadImages(images_path, modalities):
    """takes a list of modality strings and loads dicoms into an imvolume dataset from images_path

    Args:
        images_path --  Full path to patient specific directory containing various modality dicom images
            each modality imageset is contained in a directory within images_path where the modality string
            in modalities must match the directory name. This subdir is recursively searched for all dicoms
        modalities  --  list of modality strings that are used to identify subdirectories from which dicoms
            are loaded
    Returns:
        dictionary of {modality: imvolume} that contains loaded image data for each modality supported
    """
    # check if path specified exists
    if (not os.path.exists(images_path)):
        print('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # load imvector and store to dictionary for each modality
        if (modalities is None or len(modalities)==0):
            print('No modalities supplied. skipping')
            return None
        else:
            volumes = {}
            l1_indent = g_indents[1]
            l2_indent = g_indents[2]
            for mod in modalities:
                print_indent('Importing {mod:s} images'.format(mod=mod.upper()),l1_indent)
                dicom_path = os.path.join(images_path, '{mod:s}'.format(mod=mod))

                if (os.path.exists(dicom_path)):
                    # recursively walk modality path for dicom images, and build a dataset from it
                    volumes[mod] = imvolume(dicom_path, recursive=True)
                    volume = volumes[mod]
                    if (volume is not None):
                        print_indent('stacked {len:d} datasets of shape: ({d:d}, {r:d}, {c:d})'.format(
                                len=volume.numberOfSlices,
                                d=1,
                                r=volume.rows,
                                c=volume.columns
                            ), l2_indent)
                else:
                    print_indent('path to {mod:s} dicoms doesn\'t exist. skipping\n'
                        '(path: {path:s}'.format(mod=mod, path=dicom_path), l2_indent)
                print()
            return volumes


def loadEntropy(entropy_pickle_path, image_vectors, radius=4, savePickle=True, verbose=False):
    """Checks if entropy vector has already been pickled at path specified and
    loads the files if so, or computes entropy for each modality and pickles for later access.
    Returns tuple of entropy imvectors (CT_entropy, PET_entropy)
    
    Args:
        entropy_pickle_path --  should be the full path to the patient specific "precomputed" dir.
            pickle file names are searched for occurence of pet, ct, and entropy and will be loaded if a 
            modality string and "entropy" are both present.
        image_vectors       --  dictionary of {modality, imvector} that contains loaded image data for 
            each modality supported
    """
    # check if path specified exists
    if (not os.path.exists(entropy_pickle_path)):
        print('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # extract modalities from image_vectors
        if (image_vectors is None or len(image_vectors)==0):
            print('No image data was provided. Skipping')
            return None
        modalities = image_vectors.keys()

        # get list of files in immediate path (not recursive)
        files = [
            f
            for f in os.listdir(entropy_pickle_path)
            if os.path.isfile(os.path.join(entropy_pickle_path,f))
            and ('entropy' in f.lower())
            and ('.pickle' == os.path.splitext(f)[1])
        ]

        # load first file that matches the search and move to next modality
        entropy_vectors = {}
        for mod in modalities:
            l1_indent = g_indents[1]
            l2_indent = g_indents[2]
            print_indent('Loading {mod:s} entropy:'.format(mod=mod.upper()),l1_indent)
            # initialize to None
            entropy_vectors[mod] = None
            # find first pickle that matches modality string or compute entropy fresh for that modality
            match = next((f for f in files if mod in f.lower()), None) # gets first match and stops
            if (match is not None):
                # found pickled entropy vector, load it and add to dict
                print_indent('Pickled entropy vector found ({mod:s}). Loading.'.format(mod=mod),l2_indent)
                try:
                    with open(os.path.join(entropy_pickle_path, match), 'rb') as p:
                        entropy_vectors[mod] = pickle.load(p)
                except:
                    print_indent('there was a problem loading the file: {path:s}'.format(path=p),l2_indent)
                    entropy_vectors[mod] = None
                else:
                    print_indent('Pickled {mod:s} entropy vector loaded successfully.'.format(
                        mod=mod.upper()),l2_indent)
            else:
                # if no file is matched for that modality, calculate instead if image dicom files are
                #   present for that modality
                # no match, compute entropy
                print_indent('No pickled entropy vector found ({mod:s})'.format(mod=mod),l2_indent)
                # check for presence of image vector in modality
                image = image_vectors[mod]
                if image is not None:
                    print_indent('Computing entropy now...'.format(mod=mod),l2_indent)
                    entropy_vectors[mod] = features.image_entropy(image, radius, verbose)
                    if entropy_vectors[mod] is None:
                        print_indent('Failed to compute entropy for {mod:s} images.'.format(
                            mod=mod.upper()),l2_indent)
                    else:
                        print_indent('Entropy computed successfully',l2_indent)
                        # pickle for later recall
                        try:
                            pickle_dump_path = os.path.join(entropy_pickle_path,
                                                            '{mod:s}_entropy.pickle'.format(mod=mod))
                            with open(pickle_dump_path, 'wb') as p:
                                pickle.dump(entropy_vectors[mod], p)
                        except:
                            print_indent('error pickling: {:s}'.format(pickle_dump_path),l2_indent)
                        else:
                            print_indent('entropy pickled successfully to:\n{:s}'.format(pickle_dump_path),l2_indent)
                else:
                    print_indent('No {mod:s} image vector was supplied.'
                        ' Could not compute entropy.'.format(mod=mod.upper()),l2_indent)
            print()

        # return dict of modality specific entropy imvectors with keys defined by keys for image_vectors arg.
        return entropy_vectors
