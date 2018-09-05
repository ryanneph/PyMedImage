"""
dcmio.py

handles all dicom input/output tasks including reading dicom files and building
workable datasets for later use in projects
"""

import os
import sys
import logging
import warnings
from datetime import datetime
import pydicom
import pydicom.dataset
from string import Template
from .misc import indent, g_indents, ensure_extension

# initialize module logger
logger = logging.getLogger(__name__)

RTIMAGE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.1"
CTIMAGE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.2"
MRIMAGE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.4"

def get_roi_contour_module(ds):
    ds.ROIContourSequence = []
    return ds

def add_roi_to_roi_contour(ds, roi, contours, ref_images):
    newroi = pypydicom.dataset.Dataset()
    ds.ROIContourSequence.append(newroi)
    newroi.ReferencedROINumber = roi.ROINumber
    newroi.ROIDisplayColor = roicolors[(roi.ROINumber-1) % len(roicolors)]
    newroi.ContourSequence = []
    for i, contour in enumerate(contours, 1):
        c = pydicom.dataset.Dataset()
        newroi.ContourSequence.append(c)
        c.ContourNumber = i
        c.ContourGeometricType = 'CLOSED_PLANAR'
        # c.AttachedContours = [] # T3
        if ref_images != None:
            c.ContourImageSequence = [] # T3
            for image in ref_images:
                if image.ImagePositionPatient[2] == contour[0,2]:
                    imgref = pydicom.dataset.Dataset()
                    imgref.ReferencedSOPInstanceUID = image.SOPInstanceUID
                    imgref.ReferencedSOPClassUID = image.SOPClassUID
                    # imgref.ReferencedFrameNumber = "" # T1C on multiframe
                    # imgref.ReferencedSegmentNumber = "" # T1C on segmentation
                    c.ContourImageSequence.append(imgref)
        # c.ContourSlabThickness = "" # T3
        # c.ContourOffsetVector = [0,0,0] # T3
        c.NumberofContourPoints = len(contour)
        c.ContourData = "\\".join(["%g" % x for x in contour.ravel().tolist()])
    return newroi


roicolors = [[255,0,0],
             [0,255,0],
             [0,0,255],
             [255,255,0],
             [0,255,255],
             [255,0,255],
             [255,127,0],
             [127,255,0],
             [0,255,127],
             [0,127,255],
             [127,0,255],
             [255,0,127],
             [255,127,127],
             [127,255,127],
             [127,127,255],
             [255,255,127],
             [255,127,255],
             [127,255,255]]

def get_structure_set_module(ds, DT, TM, ref_images, current_study):
    ds.StructureSetLabel = "Structure Set" # T1
    # ds.StructureSetName = "" # T3
    # ds.StructureSetDescription = "" # T3
    # ds.InstanceNumber = "" # T3
    ds.StructureSetDate = DT # T2
    ds.StructureSetTime = TM # T2
    if ref_images != None and len(ref_images) > 0:
        reffor = pydicom.dataset.Dataset()
        reffor.FrameofReferenceUID = get_current_study_uid('FrameofReferenceUID', current_study)
        refstudy = pydicom.dataset.Dataset()
        refstudy.ReferencedSOPClassUID = get_uid("Detached Study Management SOP Class") # T1, but completely bogus.
        refstudy.ReferencedSOPInstanceUID = get_current_study_uid('StudyUID', current_study) # T1
        assert len(set(x.SeriesInstanceUID for x in ref_images)) == 1
        refseries = pydicom.dataset.Dataset()
        refseries.SeriesInstanceUID = ref_images[0].SeriesInstanceUID
        refseries.ContourImageSequence = [] # T3
        for image in ref_images:
            imgref = pydicom.dataset.Dataset()
            imgref.ReferencedSOPInstanceUID = image.SOPInstanceUID
            imgref.ReferencedSOPClassUID = image.SOPClassUID
            # imgref.ReferencedFrameNumber = "" # T1C on multiframe
            # imgref.ReferencedSegmentNumber = "" # T1C on segmentation
            refseries.ContourImageSequence.append(imgref)
        refstudy.RTReferencedSeriesSequence = [refseries]
        reffor.RTReferencedStudySequence = [refstudy]
        ds.ReferencedFrameOfReferenceSequence = [reffor] # T3
    ds.StructureSetROISequence = []

    return ds

def make_dicom_boilerplate(SeriesInstanceUID=None, StudyInstanceUID=None, FrameOfReferenceUID=None):
    # Populate required values for file meta information
    file_meta = pydicom.dataset.Dataset()
    file_meta.FileMetaInformationGroupLength = 204
    file_meta.FileMetaInformationVersion = b'00\01'
    file_meta.MediaStorageSOPClassUID = CTIMAGE_SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = '2.25.229451600072090404564844894284998027179' #arbitrary specific to this library
    file_meta.ImplementationVersionName = "PyMedImage"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds = pydicom.dataset.Dataset()
    ds.preamble = b"\0" * 128
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    datestr = datetime.now().strftime('%Y%m%d')
    timestr = datetime.now().strftime('%H%M%S')
    ds.ContentDate = datestr
    ds.ContentTime = timestr
    ds.StudyDate = datestr
    ds.StudyTime = timestr
    ds.PatientID = 'ANON0001'
    ds.StudyID = 'ANON0001'
    ds.SeriesNumber = '0001'
    ds.StudyDate = datestr
    ds.StudyTime = timestr
    ds.AccessionNumber = ''
    ds.ReferringPhysiciansName = ''
    ds.PatientName = 'ANON0001'
    ds.PatientSex = ''
    ds.PatientAge = ''
    ds.PatientBirthDate = ''
    ds.PatientOrientation = 'LA'
    ds.PatientPosition = 'HFS'
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.InstanceNumber = 1
    ds.StudyInstanceUID = pydicom.uid.generate_uid() if StudyInstanceUID is None else StudyInstanceUID
    ds.SeriesInstanceUID = pydicom.uid.generate_uid() if SeriesInstanceUID is None else SeriesInstanceUID
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid() if FrameOfReferenceUID is None else FrameOfReferenceUID
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
    ds.Modality = ''
    ds.SOPClassUID = CTIMAGE_SOP_CLASS_UID
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1.0
    ds.KVP = ''
    ds.AcquisitionNumber = 1
    ds.PixelRepresentation = 0
    ds.SliceLocation = 0.0
    ds.Rows = 0
    ds.Columns = 0
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.Units = 'HU'
    ds.RescaleType = 'HU'
    return ds

def write_dicom(path, dataset):
    """write a pydicom dataset to dicom file"""
    ensure_extension(path, '.dcm')
    pydicom.dcmwrite(path, dataset, write_like_original=False)

def read_dicom(path, only_header=False):
    """read a dicom slice using pydicom and return the dataset object"""
    if (not os.path.exists(path)):
        raise FileNotFoundError('file at {!s} does not exist'.format(path))
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=only_header)
    except pydicom.errors.InvalidDicomError as e:
        warnings.warn('pydicom.read_dicom() failed with error: "{!s}". Trying again with force=True'.format(e))
        ds = pydicom.dcmread(path, stop_before_pixels=only_header, force=True)
    return ds

def read_dicom_dir(path, recursive=False, only_headers=False, verbosity=0):
    """read all dicom files in directory and return a list of the dataset objects.

    Keyword arguments:
    recursive -- should we walk into subdirs?
    verbosity -- 1: low, 2: high, 3: full
    """
    ds_list = []
    dicom_paths = []
    if (not os.path.exists(path)):
        logger.info('Execution Terminated. Supplied path did not exist: {:s}'.format(path))
        sys.exit(1)
    else:
        l1_indent = g_indents[2]
        l2_indent = g_indents[3]
        printstring = Template('Reading dicoms in specified path${extra}:\n"{:s}"')
        extra = ''
        if recursive:
            extra = ' and subdirs'
        printstring = printstring.substitute(extra=extra).format(path)
        logger.debug(indent(printstring, l1_indent))
        for root, dirs, files in os.walk(path, topdown=True):
            # build the list of valid dicom file paths then load them after walk
            for file in files:
                _, file_extension = os.path.splitext(file)
                if file_extension in ['.dcm', '.dicom', '.mag']:
                    dicom_paths.append(root + '/' + file)
            if (not recursive):
                # clear dirs so that walk stops after this level
                del dirs[:]

        # Now read the dicom files that were located within path
        if verbosity == 0:
            #low verbosity
            logger.debug(indent(dicom_paths[:5],l2_indent))
        elif verbosity == 1:
            #high verbosity
            logger.debug(indent(dicom_paths[:20],l2_indent))
        elif verbosity > 2:
            #full verbosity
            logger.debug(indent(dicom_paths,l2_indent))

        if (len(dicom_paths)>0):
            for file in dicom_paths:
                file_dataset = read_dicom(file, only_header=only_headers)
                if file_dataset is not None:
                    ds_list.append(file_dataset)
            return ds_list
        else:
            return None

def probeDicomProperties(root, prop_label_list, recursive=True, silent=False):
    """probe all dicoms in root for unique values of the properties defined in prop_label_list

    Returns:
        dict<k: prop_label, v: set()>: a set for each property is accumulated showing the unique values
            encountered across the entire dataset within root
    """
    sets = {}
    for l in prop_label_list:
        sets[l] = set()

    dcm_counter = 0
    for r, dirs, files in os.walk(root, topdown=True):
        # build the list of valid dicom file paths then load them after walk
        for file in files:
            _, file_extension = os.path.splitext(file)
            if file_extension in ['.dcm', '.dicom']:
                try:
                    ds = read_dicom(os.path.join(r, file))
                    dcm_counter += 1

                    for l, s in sets.items():
                        #print(l)
                        val = ds.get(l)
                        if isinstance(val, pydicom.multival.MultiValue):
                            val = tuple(val)
                        #print(type(val))
                        s.add(val)
                except:
                    continue

        if (not recursive):
            # clear dirs so that walk stops after this level
            del dirs[:]

    if not silent:
        print('Finished probing {:d} dicom files.'.format(dcm_counter))
        print('')
        print('Probe Results:')
        print('--------------')
        for l, s in sets.items():
            print('| SET: {!s}'.format(l))
            for idx, item in enumerate(s):
                print('|   {!s}.  {!s}'.format(idx+1, item))
        print('--------------')

    return sets
