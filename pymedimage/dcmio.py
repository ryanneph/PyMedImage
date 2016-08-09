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
from utils import rttypes, features
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

