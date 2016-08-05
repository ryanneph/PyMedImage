"""rttypes.py

Contains datatype definitions necessary for working with Dicom volumes, slices, and contours[masks]
"""

import numpy as np
import dicom # pydicom
from utils import dcmio
from operator import attrgetter, methodcaller

class imslice():
    """Data type for a single dicom slice

    Contains the dicom file dataset as well as convenience functions for extracting commonly used fields
    """
    def __init__(self, dataset):
        """store the dataset"""
        self._dataset = None
        self.mask = None
        if (isinstance(dataset, dicom.dataset.Dataset)):
            self._dataset = dataset
        else:
            print('dataset is not a valid pydicom dataset')
            raise ValueError

    ## Get methods
    def _safeGetAttr(self, tag):
        """defines safe dicom attribute querying"""
        if (self._dataset is None):
            print('dataset is not a valid pydicom dataset')
            raise ValueError

        if (not isinstance(tag, str)):
            print('tag must be a string')
            raise TypeError

        if (not tag in self._dataset.dir()):
            print('couldn\'t locate attribute: "{attr:s}" in dataset')
            raise KeyError
        else:
            return self._dataset.data_element(tag).value


    def pixelData(self, mask=False, rescale=False, vectorize=False):
        """get numpy ndarray of pixel intensities.

        Optional Args:
            mask       -- return the element-wise product of pixel_array and binary mask
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
            vectorize  -- return a 1darray in row-major order
        """
        # check that dataset exists
        if (self._dataset is None):
            print('No dataset loaded. Can\'t get pixel data.')
            return None

        pixel_data = self._dataset.pixel_array
        ##TODO Add handling for when pixel_array is jpeg compressed

        # MASK
        if (mask):
            if (not self.mask is None):
                pixel_data = np.multiply(pixel_data, self.mask)
                print('image mask applied')
            else:
                print('No mask has been defined. returning unmasked data')

        # RESCALE
        if (rescale):
            f = float(self.rescaleSlope())
            i = float(self.rescaleIntercept())
            pixel_data = np.add(np.multiply(pixel_data, f), i)
            print('data rescaled using: ({f:0.3f} * data[i]) + {i:0.3f}'.format(f=f, i=i) )

        # VECTORIZE
        if (vectorize):
            # flatten ndarray in row-major order (c-style)
            pixel_data = pixel_data.flatten(order='C').reshape((-1, 1), order='C')

        return pixel_data

    def sliceThickness(self):
        """gets thickness of this slice in mm
        
        Returns:
            float
        """
        return self._safeGetAttr('SliceThickness')

    def imagePositionPatient(self):
        """gets (x,y,z) tuple indicating spatial position in RCS of top left (first) pixel in mm

        Returns:
            list<float>[3]
        """
        return self._safeGetAttr('ImagePositionPatient')

    def imageOrientationPatient(self):
        """gets six-tuple of direction cosines of first row and first column w.r.t. the patient
        
        Returns:
            list<float>[6]
        """
        return self._safeGetAttr('ImageOrientationPatient')

    def pixelSpacing(self):
        """gets (r,c) tuple indicating pixel spacing betwen adjacent rows (r) and columns (c) in mm
        
        Returns:
            list<float>[2]
        """
        return self._safeGetAttr('PixelSpacing')

    def sliceLocation(self):
        """gets location of this slice plane w.r.t. unspecified reference position in mm.
        
        Returns:
            float
        """
        return self._safeGetAttr('SliceLocation')

    def numberOfSlices(self):
        """gets number of slices for this series instance (should match number of dicom files found)

        Returns:
            float
        """
        if ('NumberOfSlices' in self.dataset.dir()):
            return self._safeGetAttr('NumberOfSlices')
        else:
            print('# slices not defined for this image series')
            return None

    def rows(self):
        """gets number of rows in this slice
        
        Returns:
            int
        """
        return self._safeGetAttr('Rows')

    def columns(self):
        """gets number of columns in this slice

        Returns:
            int
        """
        return self._safeGetAttr('Columns')

    def seriesInstanceUID(self):
        """gets the UID for this imaging series as a string.
        
        all slices found in a directory should have matching values.

        Returns:
            str
        """
        return self._safeGetAttr('SeriesInstanceUID')

    def SOPInstanceUID(self):
        """gets the UID for this slice as a string.

        This should be unique to this slice within this seriesInstance.

        Returns:
            str
        """
        return self._safeGetAttr('SOPInstanceUID')

    def instanceNumber(self):
        """gets the integer identifying the occurence of this slice within the series.

        This should be unique within the series instance.
        if the slice is an axial plane then increments from feet->head
        if coronal then increments from anterior->posterior
        if sagittal then increments from right->left (patient)

        Returns:
            int
        """
        return self._safeGetAttr('InstanceNumber')

    def modality(self):
        """gets the modality of the image slice as a string
            
        Returns:
            str
        """
        return self._safeGetAttr('Modality')

    def rescaleSlope(self):
        """gets the value by which the raw pixel data should be rescaled (as float)

        Returns:
            float
        """
        return self._safeGetAttr('RescaleSlope')

    def rescaleIntercept(self):
        """gets the value by which the raw pixel data should be offset (as float) after factor rescaling

        Returns:
            float
        """
        return self._safeGetAttr('RescaleIntercept')


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
    def __init__(self, slices, recursive=False):
        self.__dict_instanceNumber = {}
        self.__dict_SOPInstanceUID = {}
        self.numberOfSlices = None
        self.rows = None
        self.columns = None
        self.modality = None
        self.seriesInstanceUID = None
        self.rescaleSlope = None
        self.rescaleIntercept = None

        if (isinstance(slices, str)):
            # slices is a path to a directory containing a series of dicom files
            path = slices
            self._fromDir(path, recursive=recursive)

        elif (isinstance(slices, list)):
            # pass to constructor
            self._fromSliceList(slices)

        else:
            print('must supply a list of imslices or a valid path to a dicom series')
            raise TypeError

    def _fromDir(self, path, recursive=False):
        """constructor: takes path to directory containing dicom files and builds a list of imslices

        Args:
            recursive      -- find dicom files in all subdirectories?
        """
        # get the datasets from files
        dataset_list = dcmio.read_dicom_dir(path, recursive=recursive)

        # generate imslices and build a list
        imslice_list = []
        for slice_dataset in dataset_list:
            imslice_list.append(imslice(slice_dataset))

        # pass imslice list to constructor
        self._fromSliceList(imslice_list)

    def _fromSliceList(self, slices):
        """constructor: takes a list of imslices and builds imvolume object properties"""
        # check that all elements are valid slices, if not remove and continue
        nRemoved = 0
        for i, slice in enumerate(slices):
            if (not isinstance(slice, imslice)):
                print('invalid type ({t:s}) at idx {i:d}. removing.'.format(
                    t=str(type(slice)),
                    i=i ) )
                slices.remove(slice)
                nRemoved += 1
        if (nRemoved > 0):
            print('# slices removed with invalid types: {:d}'.format(nRemoved))

        # build object properties
        self.numberOfSlices = len(slices)
        self.rows = slices[0].rows()
        self.columns = slices[0].columns()
        self.modality = slices[0].modality()
        self.seriesInstanceUID = slices[0].seriesInstanceUID()
        self.rescaleSlope = slices[0].rescaleSlope()
        self.rescaleIntercept = slices[0].rescaleIntercept()
        # add slices to dicts
        for slice in slices:
            self.__dict_instanceNumber[slice.instanceNumber()] = slice
            self.__dict_SOPInstanceUID[slice.SOPInstanceUID()] = slice

    def _sliceList(self):
        """Function allowing extraction of a list of imslices from volume dictionary
        
        Returns:
            list<imslices>[self.numberOfSlices]
        """
        if (len(self.__dict_instanceNumber) == len(self.__dict_SOPInstanceUID)):
            return list(self.__dict_instanceNumber.values())
        else:
            print('ERROR: dictionaries do not match')
            raise Exception

    def getSlice(self, ID, asdataset=False, mask=False, rescale=False, vectorize=False):
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
        if (isinstance(ID, str)):
            # check SOPInstanceUID dict for existence
            __dict = self.__dict_SOPInstanceUID
        elif (isinstance(ID, int)):
            # check instanceNumber dict for existence
            __dict = self.__dict_instanceNumber
        else:
            print('invalid type: "{:s}". ID must be a "str" or "int"'.format(str(type(ID))))
            raise TypeError

        # check for existence
        if (not ID in __dict):
            print('ID: "{:s}" not found in volume'.format(str(ID)))
            raise KeyError

        if (asdataset):
            return __dict[ID]
        else:
            return __dict[ID].pixelData(mask=mask, rescale=rescale, vectorize=vectorize)


    def sortedSliceList(self, sortkey='instanceNumber', ascend=True):
        """returns a sorted version of the volume's slice list where the key is the slice instanceNumber

        Args:
            sortkey      -- method that takes as input an imslice object and returns a sortkey
            ascend       -- sort in ascending order?

        Returns
            list<imslice>[self.numberOfSlices]
        """
        return sorted(self._sliceList(), key=methodcaller(sortkey), reverse=(not ascend))

    def vectorize(self, mask=False, rescale=False):
        """constructs vector (np 1darray) of all contained imslices

        Shape will be (numberOfSlices*rows*colums, 1)

        Optional Args:
            mask       -- return the element-wise product of pixel_array and binary mask
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
            vectorize  -- return a 1darray in row-major order

        Returns:
            np.ndarray of shape (numberOfSlices*rows*colums, 1)
        """
        # sort by slice location (from low to high -> inferior axial to superior axial) 
        # begin vectorization
        vect_list = []
        for slice in self.sortedSliceList(sortkey='sliceLocation', ascend=True):
            vect_list.append(slice.pixelData(mask=mask, rescale=rescale, vectorize=True))
        full_vect = np.vstack(vect_list)

        return full_vect

