"""rttypes.py

Contains datatype definitions necessary for working with Dicom volumes, slices, and contours[masks]
"""

import numpy as np
import dicom # pydicom
from utils import dcmio
from operator import attrgetter, methodcaller

from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

class imslice():
    """Data type for a single dicom slice

    Contains the dicom file dataset as well as convenience functions for extracting commonly used fields
    """
    def __init__(self, dataset):
        """store the dataset"""
        self._dataset = None
        self.maskslice_dict = {}
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


    def pixelData(self, mask=False, ROIName=None, rescale=False, vectorize=False):
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

        # RESCALE
        if (rescale):
            f = float(self.rescaleSlope())
            i = float(self.rescaleIntercept())
            pixel_data = np.add(np.multiply(pixel_data, f), i)
            print('data rescaled using: ({f:0.3f} * data[i]) + {i:0.3f}'.format(f=f, i=i) )

        # MASK
        if (mask):
            if (not self.mask is None):
                thismaskslice = self.maskslice_dict[ROIName]
                pixel_data = np.multiply(pixel_data, thismaskslice.pixelData(vectorize=False))
                print('image mask applied')
            else:
                print('No mask has been defined. returning unmasked data')

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


class triplet:
    """Storage class for 3d points"""
    def __init__(self, x,y,z):
        self.__attrs__ = [ float(x),
                           float(y),
                           float(z) ]

class contourPoints:
    def __init__(self, raw_contour_data):
        """takes contour data from rtstruct and creates ordered list of 3d coord triplets"""
        self.raw_contour_data = None
        self.contour_points = None
        if (raw_contour_data is not None and isinstance(raw_contour_data, list)):
            self.raw_contour_data = raw_contour_data
            self.contour_points = self.unpackContourData(raw_contour_data)
        else:
            print('contour_data is not of the appropriate type')

    def unpackContourData(self, raw_contour_data):
        """take raw contour_data from rtstruct and return ordered list of 3d coord triplets"""
        if (raw_contour_data is not None and isinstance(raw_contour_data, list)):
            points_list = []
            for x, y, z in grouper(3, raw_contour_data):
                points_list.append(triplet(x,y,z))
            return points_list
        else:
            return None

    def __str__(self):
        outstr = ''
        for point in self.contour_points:
            outstr += '('
            first = True
            for value in point.__attrs__:
                if first==True:
                    first=False
                else:
                    outstr += ', '
                outstr += '{:0.3f}'.format(value)
            outstr += ')\n'
        return outstr


class maskslice():
    """takes a contour dataset and extracts properties
    """
    def __init__(self, contour_dataset):
        # set properties
        self.numberOfPoints = contour_dataset.NumberOfContourPoints
        if (self.numberOfPoints <=0):
            print('no points found')
            raise ValueError
        self.contourGeometricType = contour_dataset.ContourGeometricType
        self.contourPoints = contourPoints(contour_dataset.ContourData)
        self.SOPInstanceUID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
        self._cached_pixelData = None

    def pixelData(self, startlocation, pixelspacing, vectorize=False):
        """returns the mask binary pixel data as a matrix or vector

        Args:
            startlocation  -- volume start location (of first pixel) for accompanying image data as 
                                a tuple of coordinate measures (z, y, x) in mm
            pixelspacing   -- spacing between adjacent pixels as a tuple (z, y, x) in mm

        Optional Args:
            vectorize  -- return a 1darray in row-major order
        """
        # for each coordinate tuple in contour points, convert to actual coordinate indices and
        # generate binary mask using pillow draw polygon

        # if cached result already exists then load that instead
        if (self._cached_pixelData is not None):
            return self._cached_pixelData
        else:
            # calculate new mask from contourpoints and image geometry
            #calcmask = 
            #TODO

            # cache result for use later without recalc
            self._cached_pixelData = calcmask
            return calcmask


class maskvolume():
    """Takes dicom dataset: ROIContour and StructureSetROI for a single ROI and constructs/stores maskslices
    for each slice in a dict by key=SOPInstanceUID
    """
    def __init__(self, ROIContour, StructureSetROI):
        """takes ROIContour and supplementary info in StructureSetROI and creates a dict of maskslices
        accessible by key=SOPInstanceUID
        """
        self.ROIName = None
        self.referencedFrameOfReferenceUID = None
        self.modality = None
        self.ROINumber = None
        self._dict_SOPInstanceUID = {}

        # assign properties
        if (ROIContour is not None and StructureSetROI is not None):
            self.ROIName = StructureSetROI.ROIName
            self.ROINumber = StructureSetROI.ROINumber
            self.referencedFrameOfReferenceUID = StructureSetROI.ReferencedFrameOfReferenceUID

            # populate slice dict
            for slicedataset in ROIContour.ContourSequence:
                maskslice_single = maskslice(slicedataset)
                self._dict_SOPInstanceUID[maskslice_single.SOPInstanceUID] = maskslice_single
        else:
            print('invalid datasets provided')
            raise ValueError


class imvolume():
    """Data container for a dicom series instance containing a set of imslices

    contains a (sortable) list of imslices and sorting functions as well as convenience functions for
    performing further processing with the volume intensities
    """
    def __init__(self, slices, recursive=False, sortkey='instanceNumber', ascend=True, maskvolume_dict=None,
                 ROIName=None):
        self.__dict_instanceNumber = {}
        self.__dict_SOPInstanceUID = {}
        self._imvector = None
        self._maskvector = None
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

        # pair mask slices with slices
        self._pairMasks(maskvolume_dict)

        # store static vectorized voxel intensities
        self.__vectorize_static(ROIName)


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

    def _pairMasks(self, maskvolume_dict):
        """takes a dict of mask volumes and matches each slice in each volume with the corresponding
        imslice, storing its maskslice object to imslice.maskslice

        Args:
            maskvolume_dict     -- dict of key=contour name, value=maskvolume object

        """
        if (maskvolume_dict is not None and len(maskvolume_dict) > 0):
            # make dict of key=SOPInstanceUID and value: maskslice_dict (key=ROIName, value=maskslice) for
            # injection into slice.maskslice_dict
            maskslice_dict = {}
            for ROIName, thismaskvolume in maskvolume_dict.items():
                for SOPInstanceUID, thismaskslice in thismaskvolume._dict_SOPInstanceUID.items():
                    if (not SOPInstanceUID in maskslice_dict):
                        #initialize value as empty dict, add to it for subequent matches of SOPInstanceUID
                        maskslice_dict[SOPInstanceUID] = {}
                    else:
                        # add thismaskslice with SOPInstanceUID as value to underlying dict with key=ROIName
                        # result should be a dict of dicts where top level key is SOPInstanceUID and
                        # second level key is ROIName
                        # second level dict will be copied to imslice.maskslice_dict for use in masking tasks
                        # with ROI specification later on
                        (maskslice_dict[SOPInstanceUID])[ROIName] = thismaskslice

            #check output
            for SOPInstanceUID, maskslice_dict_final in maskslice_dict.items():
                print(str(SOPInstanceUID) + ' len: ' + str(len(maskslice_dict_final)))

            # store to imslice maskslice_dict for each slice in volume
            for SOPInstanceUID, slice in self.__dict_SOPInstanceUID.items():
                print(SOPInstanceUID)
                print(slice._safeGetAttr('FrameOfReferenceUID'))
                print(list(self.__dict_SOPInstanceUID.keys())[0])
                if (SOPInstanceUID in maskslice_dict):
                    print('storing')
                    slice.maskslice_dict = maskslice_dict[SOPInstanceUID]
                else:
                    print('nothing found here')

            #check by grabbing random slice from this volume and printing the maskslice dict
            for i in range(len(self.__dict_SOPInstanceUID.values())):
                temp = list(self.__dict_SOPInstanceUID.values())[i]
                if (len(temp.maskslice_dict.values()) > 0):
                    print(temp.maskslice_dict)
                    break


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


    def sortedSliceList(self, sortkey='sliceLocation', ascend=True):
        """returns a sorted version of the volume's slice list where the key is the slice instanceNumber

        Args:
            sortkey      -- method that takes as input an imslice object and returns a sortkey
            ascend       -- sort in ascending order?

        Returns
            list<imslice>[self.numberOfSlices]
        """
        return sorted(self._sliceList(), key=methodcaller(sortkey), reverse=(not ascend))

    def __vectorize_static(self, ROIName):
        """constructs vector (np 1darray) of all contained imslices and corresponding mask vector (if masks are available)

        Shape will be (numberOfSlices*rows*colums, 1)

        Returns:
            np.ndarray of shape (numberOfSlices*rows*colums, 1)
        """
        # sort by slice location (from low to high -> inferior axial to superior axial) 
        # begin vectorization
        vect_list = []
        mask_list = []
        for slice in self.sortedSliceList(sortkey='sliceLocation', ascend=True):
            vect_list.append(slice.pixelData(vectorize=True))
            if (slice.maskslice_dict is not None and len(slice.maskslice_dict) > 0):
                mask_list.append(slice.maskslice_dict[ROIName])
        self._imvector = np.vstack(vect_list)
        if (len(mask_list) > 0):
            self._maskvector = np.vstack(mask_list)


    def vectorize(self, mask=False, rescale=False):
        """Apply mask and rescale options to vectorized voxel intensities from slices in the volume

        Shape will be (numberOfSlices*rows*colums, 1)

        Optional Args:
            mask       -- return the element-wise product of pixel_array and binary mask
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()

        Returns:
            np.ndarray of shape (numberOfSlices*rows*colums, 1)
        """
        vect = self._imvector

        # RESCALE
        if (rescale):
            f = float(self.rescaleSlope())
            i = float(self.rescaleIntercept())
            vect = np.add(np.multiply(vect, f), i)
            print('data rescaled using: ({f:0.3f} * data[i]) + {i:0.3f}'.format(f=f, i=i) )

        # MASK
        if (mask):
            if (not self.mask is None):
                vect = np.multiply(vect, self.mask)
                print('image mask applied')
            else:
                print('No mask has been defined. returning unmasked data')

        return vect

    def get_val(self, z, y, x, mask=False):
        """returns the voxel intensity from the static vectorized image at a location

        Uses depth-row major ordering:
        depth: axial slices inf->sup
        rows: coronal slices anterior->posterior
        cols: sagittal slices: pt.right->pt.left
        """
        r = self.rows
        c = self.columns
        d = self.numberOfSlices
        if (z<0 or y<0 or x<0) or (z>=d or y>=r or x>=c):
            return 0
        else:
            pos = r*c*z + c*y + x
            return self._imvector[pos] * self._maskvector[pos] if mask else 1


def featvolume():
    def __init__(self, shape):
        """initialize an empty feature volume of shape specified

        Args:
            shape      -- shape in numpy format as a tuple
        """
        if (len(shape) == 2):
            self.numberOfSlices = 1
            self.rows = shape[0]
            self.columns = shape[1]
        elif (len(shape) == 3):
            self.numberOfSlices = shape[0]
            self.rows = shape[1]
            self.columns = shape[2]

        self._vector = np.zeros((self.numberOfSlices * self.rows * self.columns, 1))

    def get_val(self, z, y, x):
        """convenience function for returning vector intensity at location

        Uses depth-row major ordering:
        depth: axial slices inf->sup
        rows: coronal slices anterior->posterior
        cols: sagittal slices: pt.right->pt.left
        """
        r = self.rows
        c = self.columns
        d = self.numberOfSlices
        if (z<0 or y<0 or x<0) or (z>=d or y>=r or x>=c):
            return 0
        else:
            pos = r*c*z + c*y + x
            return self._vector[pos]

    def set_val(self, z, y, x, value):
        """convenience function for reassigning feature intensity at location

        Uses depth-row major ordering:
        depth: axial slices inf->sup
        rows: coronal slices anterior->posterior
        cols: sagittal slices: pt.right->pt.left
        """
        r = self.rows
        c = self.columns
        d = self.numberOfSlices
        if not (z<0 or y<0 or x<0) and not (z>=d or y>=r or x>=c):
            pos = r*c*z + c*y + x
            self._vector[pos] = value

    def getSlice(self, idx=0, axis=0, vectorize=False):
        """extract a slice along the axis specified in numpy matrix form

        Args:
            idx     --  index of the slice
            axis    --  specifies axis along which to extract 
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            vectorize --  flatten to 1Darray?
        """
        # perform index bounding
        if axis==1:
            idx = 0 if (idx < 0) else (self.rows-1 if idx >= self.rows else idx)
            slice = self.array.reshape((self.numberOfSlices, self.rows, self.columns))[:, idx, :]
        elif axis==2:
            idx = 0 if (idx < 0) else (self.columns-1 if idx >= self.columns else idx)
            slice = self.array.reshape((self.numberOfSlices, self.rows, self.columns))[:, :, idx]
        else:
            idx = 0 if (idx < 0) else (self.numberOfSlices-1 if idx >= self.depth else idx)
            slice = self.array.reshape((self.numberOfSlices, self.rows, self.columns))[idx, :, :]

        if vectorize:
            slice = slice.flatten()
        return slice

    def vectorize(self):
        """consistency method returning the vectorized voxels in the feature volume
        """
        return self._vector
