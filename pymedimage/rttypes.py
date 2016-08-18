"""rttypes.py

Contains datatype definitions necessary for working with Dicom volumes, slices, and contours[masks]
"""

import os, sys
import numpy as np
import dicom # pydicom
from . import dcmio
from operator import attrgetter, methodcaller
import pickle
import PIL
from PIL.ImageDraw import Draw

from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

class imslice:
    """Data type for a single dicom slice

    Contains the dicom file dataset as well as convenience functions for extracting commonly used fields
    """
    def __init__(self, dataset):
        """store the dataset"""
        self._dataset = None
        self.maskslice_dict = {}
        self._densemaskslice = None
        self.__densemaskslice_ROIName = None
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


    def pixelData(self, ROIName=None, rescale=False, flatten=False, verbose=False):
        """get numpy ndarray of pixel intensities.

        Optional Args:
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
            flatten  -- return a 1darray in row-major order
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
        if (ROIName is not None):
            if (self._densemaskslice is not None
                and (ROIName == self.__densemaskslice_ROIName
                     or ROIName is None)):
                thisdensemaskslice = self._densemaskslice
            else:
                thisdensemaskslice = self.makeDenseMaskSlice(ROIName)
                self.__densemaskslice_ROIName = ROIName

            pixel_data = np.multiply(pixel_data, thisdensemaskslice)
            if verbose:
                print('image mask applied')

        # flatten
        if (flatten):
            # flatten ndarray in row-major order (c-style)
            pixel_data = pixel_data.flatten(order='C').reshape((-1, 1), order='C')

        return pixel_data

    def makeDenseMaskSlice(self, ROIName, flatten=False, maskbydefault=False, verbose=False):
        """use slice size and location info to construct dense binary mask based on the contour points
        defined in ROIName
        """
        # get maskslice from dict
        if (ROIName is None or ROIName not in self.maskslice_dict):
            # this slice must not contain the specified ROI or ROI wasnt specified
            if (maskbydefault):
                extra = 'Using mask of zeros instead.'
                thisdensemask = np.zeros((self.rows(), self.columns()))
            else:
                extra = 'leaving unmasked (using mask of ones).'
                thisdensemask = np.ones((self.rows(), self.columns()))
            if verbose:
                print('contour named: "{name:s}" not found. {extra:s}'.format(name=ROIName, extra=extra))
        else:
            slice = self.maskslice_dict[ROIName]
            if verbose:
                print('found ROI in slice')

            # create 2d coords list with index coordinates rather than relative coordinates
            (y_space, x_space) = self.pixelSpacing()
            z_space = self.sliceThickness()
            (x_start, y_start, z_start) = self.imagePositionPatient()

            contour_points = slice.contour_points
            index_coords = []
            for (x, y, z) in contour_points:
                # shift x and y and scale appropriately
                x_idx = round((x-x_start)/x_space)
                y_idx = round((y-y_start)/y_space)
                index_coords.append( (x_idx, y_idx) )

            # use PIL to draw the polygon as a dense image
            im = PIL.Image.new('1', (self.rows(), self.columns()), color=0)
            imdraw = Draw(im)
            imdraw.polygon(index_coords, fill=1, outline=None)
            del imdraw

            # convert from PIL image to np.ndarray and threshold to binary
            thisdensemask = np.array(im.getdata()).reshape((self.rows(), self.columns()))

        # flatten
        if flatten:
            thisdensemask = thisdensemask.flatten(order='C').reshape((-1, 1), order='C')

        return thisdensemask


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
        if ('NumberOfSlices' in self._dataset.dir()):
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


class maskslice:
    def __init__(self, slicedataset):
        """takes contour data from rtstruct and creates ordered list of 3d coord triplets"""
        self.raw_contour_data = None
        self.contour_points = None
        if (slicedataset is not None and isinstance(slicedataset, dicom.dataset.Dataset)):
            self.raw_contour_data = slicedataset.ContourData
            self.contour_points = self._unpackContourData(self.raw_contour_data)
        else:
            print('contour_data is not of the appropriate type')

    def _unpackContourData(self, raw_contour_data):
        """take raw contour_data from rtstruct and return ordered list of 3d coord triplets"""
        if (raw_contour_data is not None and isinstance(raw_contour_data, list)):
            points_list = []
            for x, y, z in grouper(3, raw_contour_data):
                points_list.append( (x,y,z) )
            return points_list
        else:
            return None

    def sliceLocation(self):
        """returns relative coordinates for the slice orthogonal direction
        """
        return self.contour_points[0][2]

    def __str__(self):
        outstr = ''
        for point in self.contour_points:
            outstr += '('
            first = True
            for value in point:
                if first==True:
                    first=False
                else:
                    outstr += ', '
                outstr += '{:0.3f}'.format(value)
            outstr += ')\n'
        return outstr


class maskvolume:
    """Takes dicom dataset: ROIContour and StructureSetROI for a single ROI and constructs/stores maskslices
    for each slice in a dict by key=SOPInstanceUID
    """
    def __init__(self, ROIContour, StructureSetROI):
        """takes ROIContour and supplementary info in StructureSetROI and creates a dict of maskslices
        accessible by key=SOPInstanceUID
        """
        self._dataset_ROIContour = ROIContour
        self._dataset_StructureSetROI = StructureSetROI
        self.ROIName = None
        self.referencedFrameOfReferenceUID = None
        self.ROINumber = None
        self.slicelist = []

        # assign properties
        if (ROIContour is not None and StructureSetROI is not None):
            self.ROIName = StructureSetROI.ROIName
            self.ROINumber = StructureSetROI.ROINumber
            self.referencedFrameOfReferenceUID = StructureSetROI.ReferencedFrameOfReferenceUID

            # populate slicelist
            for slicedataset in ROIContour.ContourSequence:
                self.slicelist.append(maskslice(slicedataset))
        else:
            print('invalid datasets provided')
            raise ValueError


class BaseVolume:
    """Contains basic volume functionality including storage and vectorization of image slices

    Sorting of the slices is done by InstanceNumber or SliceLocation and is specified in constructor with
    "sortkey=". The order is specified with "ascend=True"
    """
    def __init__(self, slices, recursive=False, sortkey='sliceLocation', ascend=True):
        """Constructor - takes slicelist or path to dicom files and initializes volume

        Args:
            slices      -- list of Slice objects or path to a directory of dicom files
        Optional Args:
            recursive   -- if slices is path: descend into subdirs looking for dicoms?
            sortkey     -- slice property to sort on (sliceLocation, index)
            ascend      -- sort order

        """
        # class members
        self._slicedict_index = {}
        self._slicedict_sliceLocation = {}
        self._cache_vector = None
        self.numberOfSlices = None
        self.rows = None
        self.columns = None
        self.modality = None
        self.imageOrientationPatient = None
        self.imagePositionPatient = None
        self.sliceStartPosition = None
        self.pixelSpacing = None
        self.sliceThickness = None
        self.rescaleSlope = None
        self.rescaleIntercept = None

        # CONSTRUCTOR
        if (isinstance(slices, str)):
            # slices is a path to a directory containing a series of dicom files
            path = slices
            self.__fromDir(path, recursive=recursive)

        elif (isinstance(slices, list)):
            # pass to constructor
            self.__fromSliceList(slices)

        else:
            print('must supply a list of imslices or a valid path to a dicom series')
            raise TypeError

        # store static flattend voxel intensities
        self.__vectorize_image_tocache()



    # CONSTRUCTOR METHODS
    def __fromDir(self, path, recursive=False):
        """constructor: takes path to directory containing dicom files and builds a list of imslices

        Args:
            recursive -- find dicom files in all subdirectories?
        """
        # get the datasets from files
        dataset_list = dcmio.read_dicom_dir(path, recursive=recursive)

        # generate imslices and build a list
        imslice_list = []
        for slice_dataset in dataset_list:
            imslice_list.append(imslice(slice_dataset))

        # pass imslice list to constructor
        self.__fromSliceList(imslice_list)

    def __fromSliceList(self, slices):
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
        self.rescaleSlope = slices[0].rescaleSlope()
        self.rescaleIntercept = slices[0].rescaleIntercept()
        self.imageOrientationPatient = slices[0].imageOrientationPatient()
        self.imagePositionPatient = slices[0].imagePositionPatient()
        self.pixelSpacing = slices[0].pixelSpacing()
        self.sliceThickness = slices[0].sliceThickness()
        # add slices to dicts
        for i, slice in enumerate(sorted(slices, key=methodcaller('sliceLocation'), reverse=False)):
            self._slicedict_index[i] = slice
            self._slicedict_sliceLocation[slice.sliceLocation()] = slice

        # get the real start position along the slice-orthogonal direction
        self.sliceStartPosition = self.__getSliceStartPosition()


    # PRIVATE METHODS
    def __getSliceStartPosition(self):
        """gets the lowest (most negative) z-axis slice position and reports that as the volume z-start
        
        the z location reported in the dicom tag: 'ImagePositionPatient' repeats the value already in 
        the dicom tag: 'sliceLocation'. We need the true volume start position along the z axis so we can 
        convert contour coordinates to indices for masking
        """
        return self.sortedSliceList(sortkey='sliceLocation', ascend=True)[0].sliceLocation()


    def __vectorize_image_tocache(self):
        """constructs vector (np 1darray) of all contained imslices and corresponding mask vector (if masks are available)

        Shape will be (numberOfSlices*rows*colums, 1)

        Returns:
            np.ndarray of shape (numberOfSlices*rows*columns, 1)
        """
        # sort by slice location (from low to high -> inferior axial to superior axial) 
        # begin vectorization
        vect_list = []
        for slice in self.sortedSliceList(sortkey='sliceLocation', ascend=True):
            vect_list.append(slice.pixelData(flatten=True))
        self._cache_vector = np.vstack(vect_list)


    # PROTECTED METHODS
    def _sliceList(self):
        """Function allowing extraction of a list of imslices from volume dictionary
        
        Returns:
            list<imslices>[self.numberOfSlices]
        """
        if (len(self._slicedict_index) == len(self._slicedict_sliceLocation)):
            return list(self._slicedict_index.values())
        else:
            print('ERROR: dictionaries do not match')
            raise Exception

    def _select_slicedict(self, IDX):
        """checks slicedicts for the one that will provide meaningful results for key=IDX
        """
        if (IDX in self._slicedict_index):
            # check index dict for existence
            slicedict = self._slicedict_index
        elif (IDX in self._slicedict_sliceLocation):
            # check sliceLocation dict for existence
            slicedict = self._slicedict_sliceLocation
        else:
            print('invalid type: "{:s}". IDX must be a "float" or "int"'.format(str(type(IDX))))
            raise TypeError
        return slicedict

    def _crop(self, array, cropextents=None, flatten=False):
        if (cropextents is None):
            croparray = array
        else:
            # crop
            xmin = cropextents['xmin']
            xmax = cropextents['xmax']
            if (xmin > xmax):
                xmin, xmax = xmax, xmin
            ymin = cropextents['ymin']
            ymax = cropextents['ymax']
            if (ymin > ymax):
                ymin, ymax = ymax, ymin
            zmin = cropextents['zmin']
            zmax = cropextents['zmax']
            if (zmin > zmax):
                zmin, zmax = zmax, zmin

            # convert to ndarray for easier slicing
            if (array.ndim == 3 or (array.ndim == 2 and array.shape[0] == self.numberOfSlices * self.rows * self.columns)):
                # must be a vectorized 3d array
                array = array.reshape((self.numberOfSlices, self.rows, self.columns))
                croparray = array[zmin:zmax, ymin:ymax, xmin:xmax]
            elif (array.ndim == 2):
                # must be a vectorized 2d array
                array = array.reshape((self.rows, self.columns))
                croparray = array[ymin:ymax, xmin:xmax]

        # RESHAPE
        if (flatten):
            croparray = croparray.flatten(order='C').reshape((-1, 1))

        return croparray

    # PUBLIC METHODS
    def get_val(self, z, y, x):
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
            return np.asscalar(self._cache_vector[pos])

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
            self._cache_vector[pos] = value

    def getSlice(self, idx, asdataset=False, rescale=False, flatten=False):
        """takes ID as sliceLocation[float] or InstanceNumber[int] and returns a numpy ndarray or\
                the dataset object

        Args:
            idx      -- idx identifying the slice
        Optional Args:
            asdataset  -- False: return numpy ndarray using remaining opt-args to format
                           True: return imslice object
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
            flatten  -- return a 1darray in row-major order
        """
        # get the appropriate dict
        slicedict = self._select_slicedict(idx)

        # check for existence
        if (not idx in slicedict):
            print('idx: "{:s}" not found in volume'.format(str(idx)))
            raise KeyError

        if (asdataset):
            thisslice = slicedict[idx]
        else:
            thisslice = slicedict[idx].pixelData(rescale=rescale, flatten=flatten)

            # CROP
            if (cropextents is not None):
                thisslice = self._crop(thisslice, cropextents, flatten=flatten)
            else:
                # RESHAPE
                if (flatten):
                    thisslice = thisslice.flatten(order='C').reshape((-1, 1))
                else:
                    thisslice = thisslice.reshape((self.rows, self.columns))

        return thisslice

    def sortedSliceList(self, sortkey='sliceLocation', ascend=True):
        """returns a sorted version of the volume's slice list where the key is the slice location in mm

        Args:
            sortkey      -- method that takes as input an imslice object and returns a sortkey
            ascend       -- sort in ascending order?

        Returns
            list<imslice>[self.numberOfSlices]
        """
        return sorted(self._sliceList(), key=methodcaller(sortkey), reverse=(not ascend))

    def vectorize(self, rescale=False, cropextents=None, asmatrix=False, verbose=False):
        """Apply rescale options to vectorized voxel intensities from slices in the volume

        Shape will be (numberOfSlices*rows*colums, 1)

        Optional Args:
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()

        Returns:
            np.ndarray of shape (numberOfSlices*rows*colums, 1)
        """
        vect = self._cache_vector

        # CROP
        if (cropextents is not None):
            vect = self._crop(vect, cropextents, flatten=(not asmatrix))
        else:
            # RESHAPE as matrix?
            if (asmatrix):
                vect = vect.reshape((self.numberOfSlices, self.rows, self.columns))
            else:
                vect = vect.flatten(order='C').reshape((-1, 1))

        # RESCALE
        if (rescale):
            f = float(self.rescaleSlope())
            i = float(self.rescaleIntercept())
            vect = np.add(np.multiply(vect, f), i)
            if verbose:
                print('data rescaled using: ({f:0.3f} * data[i]) + {i:0.3f}'.format(f=f, i=i) )

        return vect

class MaskableVolume(BaseVolume):
    """adds ROI masking to BaseVolume"""
    def __init__(self, slices, recursive=False, sortkey='sliceLocation', ascend=True, maskvolume_dict=None):
        # call to BaseClass constructor
        super().__init__(slices, recursive=recursive, sortkey=sortkey, ascend=ascend)

        # derived class members
        self._cache_lastROIName = None
        self._cache_lastmaskextents = None
        self._cache_lastmaskextents_ROIName = None
        self._cache_maskvector = None
        self.ROINames = []

        # CONSTRUCTOR
        # pair mask slices with slices
        self.ROINames = list(maskvolume_dict.keys())
        self.__injectMaskSlices(maskvolume_dict)


    # PRIVATE METHODS
    def __injectMaskSlices(self, maskvolume_dict):
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
                for thismaskslice in thismaskvolume.slicelist:
                    sliceLocation = thismaskslice.sliceLocation()
                    if (not sliceLocation in maskslice_dict):
                        #initialize value as empty dict, add to it for subequent matches of SOPInstanceUID
                        maskslice_dict[sliceLocation] = {}
                    else:
                        # add thismaskslice with SOPInstanceUID as value to underlying dict with key=ROIName
                        # result should be a dict of dicts where top level key is SOPInstanceUID and
                        # second level key is ROIName
                        # second level dict will be copied to imslice.maskslice_dict for use in masking tasks
                        # with ROI specification later on
                        (maskslice_dict[sliceLocation])[ROIName] = thismaskslice

            # check output
            #for sliceLocation, maskslice_dict_final in maskslice_dict.items():
            #    print(str(sliceLocation) + ' len: ' + str(len(maskslice_dict_final)))

            # store to imslice maskslice_dict for each slice in volume
            for sliceLocation, slice in self._slicedict_sliceLocation.items():
                if (sliceLocation in maskslice_dict):
                    #print('storing')
                    slice.maskslice_dict = maskslice_dict[sliceLocation]
                else:
                    #print('nothing found here')
                    pass

    def __vectorize_mask_tocache(self, ROIName):
        """constructs vector (np 1darray) of all contained mask vectors (if available)

        Checks cache for valid mask vector before continuing. This is considered safe to call whenever
        we are unsure which mask is currently cached as the checking is wrapped in here
        
        Shape will be (numberOfSlices*rows*columns, 1)
        """
        if (ROIName != self._cache_lastROIName):
            # check for valid ROIName
            if (not ROIName in self.ROINames):
                print('ROI "{:s}" not found'.format(ROIName))
                raise KeyError
            print('Performing costly mask vectorization for image volume')

            # sort by slice location (from low to high -> inferior axial to superior axial) 
            # begin vectorization
            mask_list = []
            for slice in self.sortedSliceList(sortkey='sliceLocation', ascend=True):
                if ((slice.maskslice_dict is not None and len(slice.maskslice_dict) > 0)
                     and ROIName in slice.maskslice_dict):

                    if (ROIName in slice.maskslice_dict):
                        mask_list.append(slice.makeDenseMaskSlice(ROIName, flatten=True))
                else:
                    # the ROI may be valid but not contained in this slice, use zeros instead
                    mask_list.append(np.zeros((slice.rows()*slice.columns(),1)))
            if (len(mask_list) > 0):
                self._cache_maskvector = np.vstack(mask_list)
                self._cache_lastROIName = ROIName


    # PROTECTED METHODS
    def _crop(self, array, cropextents=None, flatten=False):
        if (isinstance(cropextents, str)):
            # requesting automatic crop to mask extents
            cropextents = self.getMaskExtents(ROIName=cropextents, padding=0)
        return super()._crop(array, cropextents, flatten=flatten)


    # PUBLIC METHODS
    def get_val(self, z, y, x, ROIName=None):
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
            if (ROIName is not None):
                # ensure whole volume has proper mask applied to cached _maskvector
                self.__vectorize_mask_tocache(ROIName=ROIName)
                return np.asscalar(self._cache_vector[pos] * self._cache_maskvector[pos])
            else:
                return np.asscalar(self._cache_vector[pos])

    def getSlice(self, IDX, asdataset=False, ROIName=None, cropextents=None, rescale=False, flatten=False):
        """takes IDX as sliceLocation[float] or InstanceNumber[int] and returns a numpy ndarray or\
                the dataset object

        Args:
            IDX      -- depth based index or sliceLocation in mm identifying the slice
        Optional Args:
            asdataset  -- False: return numpy ndarray using remaining opt-args to format
                           True: return imslice object
            mask       -- return the element-wise product of pixel_array and binary mask
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
            flatten  -- return a 1darray in row-major order
        """
        # get the appropriate dict
        slicedict = self._select_slicedict(IDX)

        # check for existence
        if (not IDX in slicedict):
            print('IDX: "{:s}" not found in volume'.format(str(IDX)))
            raise KeyError

        if (asdataset):
            thisslice = slicedict[IDX]
        else:
            thisslice = slicedict[IDX].pixelData(ROIName=ROIName, rescale=rescale, flatten=flatten)

            # CROP
            if (cropextents is not None):
                thisslice = self._crop(thisslice, cropextents, flatten=flatten)
            else:
                # RESHAPE
                if (flatten):
                    thisslice = thisslice.flatten(order='C').reshape((-1, 1))
                else:
                    thisslice = thisslice.reshape((self.rows, self.columns))

        return thisslice

    def getMaskSlice(self, IDX, asdataset=False, ROIName=None, cropextents=None, flatten=False, maskbydefault=False,
            verbose=False):
        """get the binary mask associated with slice[IDX] and ROIName

        Args:
            IDX        -- can be the sliceLocation or index
        
        Optional Args:
            asdataset -- returnt the maskslice object
            ROIName   -- string referencing an ROI. if None, the first available ROI will be used
            flatten -- return the maskslice as a flattened vector?
        """
        thismask = None
        # get the appropriate dict
        slicedict = self._select_slicedict(IDX)
        # check for existence
        if (not IDX in slicedict):
            print('IDX: "{:s}" not found in volume'.format(str(IDX)))
            raise KeyError

        thisslice = slicedict[IDX]
        if (asdataset):
            if (ROIName not in thisslice.maskslice_dict):
                if verbose:
                    print('contour named: "{name:s}" not found. no mask applied'.format(name=ROIName))
                raise KeyError
            else:
                thismask = thisslice.maskslice_dict[ROIName]
        else:
            thismask = thisslice.makeDenseMaskSlice(ROIName,maskbydefault=maskbydefault, verbose=verbose)

            # CROP
            if (cropextents is not None):
                thismask = self._crop(thismask, cropextents, flatten=flatten)
            else:
                # RESHAPE
                if (flatten):
                    thismask = thismask.flatten(order='C').reshape((-1, 1))
                else:
                    thismask = thismask.reshape((self.rows, self.columns))

        return thismask

    def getMaskExtents(self, ROIName, padding=0, verbose=False):
        """finds the maximum index extents in each direction that the mask volume is non-masking within

        Args:
            ROIName   -- name of the ROI for which the extents are evaluated. If the same ROIName is given
                         in two or more contiguous requests, the exents are taken from a cache of the first
                         evaluation
        Optional Args:
            padding   -- number of voxels to pad the extents with in all directions

        Returns:
            dict[6]   -- keys are 'xmin', 'xmax', y... z... describing the index of start/stop for the volume
        """
        if (ROIName is not None and ROIName in self.ROINames):
            if (ROIName == self._cache_lastmaskextents_ROIName):
                # same as last request, use cached version
                if verbose:
                    print('using cached crop extents')
                return self._cache_lastmaskextents
            else:
                if verbose:
                    print('evaluating new crop extents')
                global_limits = {'xmax': -4000,
                                 'ymax': -4000,
                                 'zmax': -4000,
                                 'xmin': 4000,
                                 'ymin': 4000,
                                 'zmin': 4000 }
                for i in range(self.numberOfSlices):
                    try:
                        # ROIName validity is checked in getMaskSlice
                        maskds = self.getMaskSlice(i, asdataset=True, ROIName=ROIName, flatten=False)
                    except:
                        # ROIName doesn't occur in this slice
                        continue

                    #convert coords list to ndarray
                    coords = np.array(maskds.contour_points)
                    (xmin, ymin, zmin) = tuple(coords.min(axis=0, keepdims=False))
                    (xmax, ymax, zmax) = tuple(coords.max(axis=0, keepdims=False))

                    #update limits
                    if xmin < global_limits['xmin']:
                        global_limits['xmin'] = xmin
                    if ymin < global_limits['ymin']:
                        global_limits['ymin'] = ymin
                    if zmin < global_limits['zmin']:
                        global_limits['zmin'] = zmin
                    if xmax > global_limits['xmax']:
                        global_limits['xmax'] = xmax
                    if ymax > global_limits['ymax']:
                        global_limits['ymax'] = ymax
                    if zmax > global_limits['zmax']:
                        global_limits['zmax'] = zmax

                (x_rel_start, y_rel_start, _) = self.imagePositionPatient
                z_rel_start = self.sliceStartPosition
                (y_space, x_space) = self.pixelSpacing
                z_space = self.sliceThickness
                global_limits['zmin'] = int(round((global_limits['zmin']-z_rel_start)/z_space)) - padding
                global_limits['zmax'] = int(round((global_limits['zmax']-z_rel_start)/z_space)) + padding
                global_limits['ymin'] = int(round((global_limits['ymin']-y_rel_start)/y_space)) - padding
                global_limits['ymax'] = int(round((global_limits['ymax']-y_rel_start)/y_space)) + padding
                global_limits['xmin'] = int(round((global_limits['xmin']-x_rel_start)/x_space)) - padding
                global_limits['xmax'] = int(round((global_limits['xmax']-x_rel_start)/x_space)) + padding

                self._cache_lastmaskextents = global_limits
                self._cache_lastmaskextents_ROIName = ROIName

        else:
            global_limits = {'xmax': self.columns,
                             'ymax': self.rows,
                             'zmax': self.numberOfSlices,
                             'xmin': 0,
                             'ymin': 0,
                             'zmin': 0 }

        return global_limits

    def vectorize(self, rescale=False, ROIName=None, cropextents=None, asmatrix=False, verbose=False):
        """Apply rescale options to vectorized voxel intensities from slices in the volume

        Shape will be (numberOfSlices*rows*colums, 1)

        Optional Args:
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
            cropextents -- dict of indices for starting and stopping the crop
                            keys are 'xmin', 'xmax', y..., z...

        Returns:
            np.ndarray of shape (numberOfSlices*rows*colums, 1)
        """

        vect = super().vectorize(rescale=rescale, asmatrix=asmatrix, verbose=verbose)

        # CROP
        if (cropextents is not None):
            vect = self._crop(vect, cropextents, flatten=(not asmatrix))

        # MASK
        if (ROIName is not None):
            # get the mask with the same shape and cropping as the image
            maskvect = self.vectorizeMask(ROIName, cropextents=cropextents,
                    asmatrix=asmatrix, verbose=verbose)
            vect = np.multiply(vect, maskvect)

        return vect

    def vectorizeMask(self, ROIName=None, cropextents=None, asmatrix=False, verbose=False):
        """Apply rescale options to vectorized voxel intensities from slices in the volume

        Shape will be (numberOfSlices*rows*colums, 1)

        Optional Args:
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()

        Returns:
            np.ndarray of shape (numberOfSlices*rows*colums, 1)
        """
        if (ROIName is not None):
            # need to revectorize proper mask
            self.__vectorize_mask_tocache(ROIName)
            if (not self._cache_maskvector is None):
                vect = self._cache_maskvector
        else:
            if verbose:
                print('ROI "{:s}" couldnt be found'.format(ROIName))
                raise KeyError

        # CROP
        if (cropextents is not None):
            vect = self._crop(vect, cropextents, flatten=(not asmatrix))
        else:
            # RESHAPE
            if (asmatrix):
                vect = vect.reshape((self.numberOfSlices, self.rows, self.columns))
            else:
                vect = vect.flatten(order='C').reshape((-1, 1))

        return vect

class FeatureVolume(BaseVolume):
    def __init__(self):
        pass

    def fromArray(self, array):
        """initialize with values in array)"""
        self._cache_vector = array.flatten(order='C').reshape((-1, 1))
        self.numberOfSlices = array.shape[0]
        self.rows = array.shape[1]
        self.columns = array.shape[2]
        return self

    def fromZeros(self, shape):
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

        self._cache_vector = np.zeros((self.numberOfSlices * self.rows * self.columns, 1))
        return self

    def fromPickle(self, pickle_path):
        """import unchanging dataformat for storing entropy calculation results

        When the definition of this class changes, we can still import old pickle files
        """
        if (not os.path.exists(pickle_path)):
            print('file at path: {:s} doesn\'t exists'.format(pickle_path))
        with open(pickle_path, 'rb') as p:
            feature_pickle = pickle.load(p)

        #import data to this object
        self._cache_vector = feature_pickle.datavector
        self.numberOfSlices = feature_pickle.depth
        self.rows = feature_pickle.rows
        self.columns = feature_pickle.columns
        return self

    def toPickle(self, pickle_path):
        """store critical data to unchanging format that can be pickled long term
        """
        feature_pickle = featpickle()
        feature_pickle.datavector = self._cache_vector
        feature_pickle.depth = self.numberOfSlices
        feature_pickle.rows = self.rows
        feature_pickle.columns = self.columns

        with open(pickle_path, 'wb') as p:
            pickle.dump(feature_pickle, p)

    def getSlice(self, idx=0, axis=0, cropextents=None, flatten=False):
        """extract a slice along the axis specified in numpy matrix form

        Args:
            idx     --  index of the slice
            axis    --  specifies axis along which to extract 
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            flatten --  flatten to 1Darray?
        """
        # perform index bounding
        if axis==1:
            idx = 0 if (idx < 0) else (self.rows-1 if idx >= self.rows else idx)
            slice = self._cache_vector.reshape((self.numberOfSlices, self.rows, self.columns))[:, idx, :]
        elif axis==2:
            idx = 0 if (idx < 0) else (self.columns-1 if idx >= self.columns else idx)
            slice = self._cache_vector.reshape((self.numberOfSlices, self.rows, self.columns))[:, :, idx]
        else:
            idx = 0 if (idx < 0) else (self.numberOfSlices-1 if idx >= self.numberOfSlices else idx)
            slice = self._cache_vector.reshape((self.numberOfSlices, self.rows, self.columns))[idx, :, :]

        # CROP
        if (cropextents is not None):
            slice = self._crop(slice, cropextents, flatten=flatten)
        else:
            # RESHAPE
            if flatten:
                slice = slice.flatten(order='C').reshape((-1, 1))

        return slice

    def vectorize(self, cropextents=None, asmatrix=False):
        """consistency method returning the vectorized voxels in the feature volume
        """
        vect = self._cache_vector

        # CROP
        if (cropextents is not None):
            vect = self._crop(vect, cropextents, flatten=(not asmatrix))
        else:
            # RESHAPE as matrix?
            if (asmatrix):
                vect = vect.reshape((self.numberOfSlices, self.rows, self.columns))

        return vect


class featpickle:
    def __init__(self):
        self.datavector = None
        self.depth = None
        self.rows = None
        self.columns = None
