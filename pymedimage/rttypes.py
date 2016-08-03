"""rttypes.py

Contains datatype definitions necessary for working with Dicom volumes, slices, and contours[masks]
"""

import numpy as np

class imslice():
"""Data type for a single dicom slice

Contains the dicom file dataset as well as convenience functions for extracting commonly used fields
"""
    def __init__(self, dataset):
        """store the dataset"""
        self.__dataset = dataset
        self.mask = None

    ## Get methods
    def pixelData(mask=False, rescale=False, vectorize=False):
        """get numpy ndarray of pixel intensities.

        Optional Args:
            mask       -- return the element-wise product of pixel_array and binary mask
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
            vectorize  -- return a 1darray in row-major order
        """
        pass

    def sliceThickness():
        """gets thickness of this slice in mm

        """
        pass

    def imagePositionPatient():
        """gets (x,y,z) tuple indicating spatial position in RCS of top left (first) pixel in mm

        """
        pass

    def imageOrientationPatient():
        """gets six-tuple of direction cosines of first row and first column w.r.t. the patient

        """
        pass

    def pixelSpacing():
        """gets (r,c) tuple indicating pixel spacing betwen adjacent rows (r) and columns (c) in mm

        """
        pass

    def sliceLocation():
        """gets location of this slice plane w.r.t. unspecified reference position in mm.

        """
        pass

    def numberOfSlices():
        """gets number of slices for this series instance (should match number of dicom files found)

        """
        pass

    def rows():
        """gets number of rows in this slice

        """
        pass

    def columns():
        """gets number of columns in this slice

        """
        pass

    def seriesInstanceUID():
        """gets the UID for this imaging series as a string.
        
        all slices found in a directory should have matching values.
        """
        pass

    def SOPInstanceUID():
        """gets the UID for this slice as a string.

        This should be unique to this slice within this seriesInstance.
        """
        pass

    def instanceNumber():
        """gets the integer identifying the occurence of this slice within the series.

        This should be unique within the series instance.
        if the slice is an axial plane then increments from feet->head
        if coronal then increments from anterior->posterior
        if sagittal then increments from right->left (patient)
        """
        pass

    def modality():
        """gets the modality of the image slice as a string

        """
        pass

    def rescaleSlope():
        """gets the value by which the raw pixel data should be rescaled (as float)

        """
        pass

    def rescaleIntercept():
        """gets the value by which the raw pixel data should be offset (as float) after factor rescaling

        """
        pass


class maskslice():
    """

    """
    def __init__():
        pass


class imvolume():
"""Data container for a dicom series instance containing a set of imslices

contains a (sortable) list of imslices and sorting functions as well as convenience functions for
performing further processing with the volume intensities
"""
    def __init__(self, imslice_list):
        self.imslice_list = imslice_list
        # check that seriesinstaceuid,Rows,Columns,Modality,numberofslices,rescales agree for
        # all imslices in imslice_list
        self.numberOfSlices = None
        self.rows = None
        self.columns = None
        self.modality = None
        self.seriesInstanceUID = None
        self.rescaleSlope = None
        self.rescaleIntercept = None

    def getSlice(ID, asdataset=False, mask=False, rescale=False, vectorize=False):
        """takes ID as SOPInstanceUID[string] or InstanceNumber[int] and returns a numpy ndarray or\
                the dataset object

        Args:
            ID      -- SOPInstanceUID[string] or InstanceNumber[int] identifying the slice
        Optional Args:
            asdataset  -- False: return numpy ndarray using remaining opt-args to format
                           True: return imslice object
            mask       -- return the element-wise product of pixel_array and binary mask
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
            vectorize  -- return a 1darray in row-major order
        """
        pass

    def vectorize(mask=False, rescale=False):
        """constructs vector (np 1darray) of all contained imslices.

        Shape will be (numberOfSlices*rows*columns, 1)
        
        Optional Args:
            mask       -- return the element-wise product of pixel_array and binary mask
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
        """
        pass

