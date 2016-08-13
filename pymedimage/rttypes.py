"""rttypes.py

Contains datatype definitions necessary for working with Dicom volumes, slices, and contours[masks]
"""

import numpy as np
import PIL
from PIL.ImageDraw import Draw
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


    def pixelData(self, mask=False, ROIName=None, rescale=False, vectorize=False, verbose=False):
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

        # VECTORIZE
        if (vectorize):
            # flatten ndarray in row-major order (c-style)
            pixel_data = pixel_data.flatten(order='C').reshape((-1, 1), order='C')

        return pixel_data

    def makeDenseMaskSlice(self, ROIName, vectorize=False, verbose=False):
        """use slice size and location info to construct dense binary mask based on the contour points
        defined in ROIName
        """
        # get maskslice from dict
        if (ROIName is None):
            #use first contour
            ROIName = list(self.maskslice_dict.keys())[0]
            if verbose:
                print("masking with default contour ({:s})".format(ROIName))
        if (ROIName not in self.maskslice_dict):
            if verbose:
                print('contour named: "{name:s}" not found. no mask applied'.format(name=ROIName))
            return np.ones((self.rows(), self.columns()))
        slice = self.maskslice_dict[ROIName]

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
        thisdensemask = np.array(im.getdata()).reshape((im.size[0], im.size[1]))

        # VECTORIZE
        if vectorize:
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


class maskslice():
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


class maskvolume():
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


class imvolume():
    """Data container for a dicom series instance containing a set of imslices

    contains a (sortable) list of imslices and sorting functions as well as convenience functions for
    performing further processing with the volume intensities
    """
    def __init__(self, slices, recursive=False, sortkey='instanceNumber', ascend=True, maskvolume_dict=None,
                 ROIName=None):
        self.__slicedict_instanceNumber = {}
        self.__slicedict_sliceLocation = {}
        self.ROINames = []
        self._vector = None           # cache for precomputed image vector
        self._maskvector = None         # cache for precomputed mask binary vector
        self.__cache_lastROIName = None
        self.numberOfSlices = None
        self.rows = None
        self.columns = None
        self.modality = None
        self.seriesInstanceUID = None
        self.rescaleSlope = None
        self.rescaleIntercept = None
        self.imagePositionPatient = None
        self.pixelSpacing = None
        self.sliceThickness = None

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

        # pair mask slices with slices
        self.ROINames = list(maskvolume_dict.keys())
        self.__injectMaskSlices(maskvolume_dict)

        # store static vectorized voxel intensities
        self.__vectorize_static(ROIName)


    def __fromDir(self, path, recursive=False):
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
        self.seriesInstanceUID = slices[0].seriesInstanceUID()
        self.rescaleSlope = slices[0].rescaleSlope()
        self.rescaleIntercept = slices[0].rescaleIntercept()
        self.imagePositionPatient = slices[0].imagePositionPatient()
        self.pixelSpacing = slices[0].pixelSpacing()
        self.sliceThickness = slices[0].sliceThickness()
        # add slices to dicts
        for slice in slices:
            self.__slicedict_instanceNumber[slice.instanceNumber()] = slice
            self.__slicedict_sliceLocation[slice.sliceLocation()] = slice

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

            #check output
            '''
            for sliceLocation, maskslice_dict_final in maskslice_dict.items():
                print(str(sliceLocation) + ' len: ' + str(len(maskslice_dict_final)))
            '''

            # store to imslice maskslice_dict for each slice in volume
            for sliceLocation, slice in self.__slicedict_sliceLocation.items():
                if (sliceLocation in maskslice_dict):
                    #print('storing')
                    slice.maskslice_dict = maskslice_dict[sliceLocation]
                else:
                    #print('nothing found here')
                    pass


    def _sliceList(self):
        """Function allowing extraction of a list of imslices from volume dictionary
        
        Returns:
            list<imslices>[self.numberOfSlices]
        """
        if (len(self.__slicedict_instanceNumber) == len(self.__slicedict_sliceLocation)):
            return list(self.__slicedict_instanceNumber.values())
        else:
            print('ERROR: dictionaries do not match')
            raise Exception

    def getSlice(self, ID, asdataset=False, mask=False, ROIName=None, rescale=False, vectorize=False):
        """takes ID as sliceLocation[float] or InstanceNumber[int] and returns a numpy ndarray or\
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
        # get the appropriate dict
        slicedict = self.__select_slicedict(ID)

        # check for existence
        if (not ID in slicedict):
            print('ID: "{:s}" not found in volume'.format(str(ID)))
            raise KeyError

        if (asdataset):
            return slicedict[ID]
        else:
            return slicedict[ID].pixelData(mask=mask, ROIName=ROIName, rescale=rescale, vectorize=vectorize)


    def sortedSliceList(self, sortkey='sliceLocation', ascend=True):
        """returns a sorted version of the volume's slice list where the key is the slice instanceNumber

        Args:
            sortkey      -- method that takes as input an imslice object and returns a sortkey
            ascend       -- sort in ascending order?

        Returns
            list<imslice>[self.numberOfSlices]
        """
        return sorted(self._sliceList(), key=methodcaller(sortkey), reverse=(not ascend))

    def __vectorize_static(self, ROIName, onlymask=False):
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
            if (not onlymask):
                vect_list.append(slice.pixelData(vectorize=True))
            if (slice.maskslice_dict is not None and len(slice.maskslice_dict) > 0):
                if (ROIName in slice.maskslice_dict):
                    mask_list.append(slice.makeDenseMaskSlice(ROIName, vectorize=True))
                else:
                    mask_list.append(np.zeros((slice.rows()*slice.columns(),1)))
        if (not onlymask):
            self._vector = np.vstack(vect_list)
        if (len(mask_list) > 0):
            self._maskvector = np.vstack(mask_list)
            self.__cache_lastROIName = ROIName

    def vectorize(self, mask=False, rescale=False, verbose=False):
        """Apply mask and rescale options to vectorized voxel intensities from slices in the volume

        Shape will be (numberOfSlices*rows*colums, 1)

        Optional Args:
            mask       -- return the element-wise product of pixel_array and binary mask
            rescale    -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()

        Returns:
            np.ndarray of shape (numberOfSlices*rows*colums, 1)
        """
        vect = self._vector

        # RESCALE
        if (rescale):
            f = float(self.rescaleSlope())
            i = float(self.rescaleIntercept())
            vect = np.add(np.multiply(vect, f), i)
            if verbose:
                print('data rescaled using: ({f:0.3f} * data[i]) + {i:0.3f}'.format(f=f, i=i) )

        # MASK
        if (mask):
            if (not self.mask is None):
                vect = np.multiply(vect, self._maskvector)
                if verbose:
                    print('image mask applied')
            else:
                if verbose:
                    print('No mask has been defined. returning unmasked data')

        return vect

    def getMaskSlice(self, ID, asdataset=False, ROIName=None, vectorize=False, verbose=False):
        """get the binary mask associated with slice[ID] and ROIName

        Args:
            ID        -- can be the sliceLocation or instanceNumber
        
        Optional Args:
            asdataset -- returnt the maskslice object
            ROIName   -- string referencing an ROI. if None, the first available ROI will be used
            vectorize -- return the maskslice as a flattened vector?
        """
        # get the appropriate dict
        slicedict = self.__select_slicedict(ID)
        # check for existence
        if (not ID in slicedict):
            print('ID: "{:s}" not found in volume'.format(str(ID)))
            raise KeyError

        thisslice = slicedict[ID]
        if (asdataset):
            if (ROIName not in thisslice.maskslice_dict):
                if verbose:
                    print('contour named: "{name:s}" not found. no mask applied'.format(name=ROIName))
                thismask = None
            else:
                thismask = thisslice.maskslice_dict[ROIName]
        else:
            thismask = thisslice.makeDenseMaskSlice(ROIName)

        # vectorize
        if (vectorize):
            thismask = thismask.flatten(order='C').reshape((-1, 1), order='C')

        return thismask


    def __select_slicedict(self, ID):
        """checks slicedicts for the one that will provide meaningful results for key=ID"""
        if (ID in self.__slicedict_instanceNumber):
            # check SOPInstanceUID dict for existence
            slicedict = self.__slicedict_instanceNumber
        elif (ID in self.__slicedict_sliceLocation):
            # check instanceNumber dict for existence
            slicedict = self.__slicedict_sliceLocation
        else:
            print('invalid type: "{:s}". ID must be a "float" or "int"'.format(str(type(ID))))
            raise TypeError
        return slicedict


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
                if (not ROIName == self.__cache_lastROIName):
                    print('Performing costly mask vectorization for image volume')
                    self.__vectorize_static(ROIName=ROIName, onlymask=True)
                return np.asscalar(self._vector[pos] * self._maskvector[pos])
            else:
                return np.asscalar(self._vector[pos])


class featvolume():
    def __init__(self, input, fromarray=False):
        """pass execution based on what is used to initialize"""
        self._vector = None
        self.rows = None
        self.columns = None
        self.numberOfSlices = None
        if (fromarray):
            self.fromArray(input)
        else:
            self.zeros(input)

    def fromArray(self, array):
        """initialize with values in array)"""
        self._vector = array.flatten(order='C').reshape((-1, 1))
        self.numberOfSlices = array.shape[0]
        self.rows = array.shape[1]
        self.columns = array.shape[2]
        return self

    def zeros(self, shape):
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
        return self

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
            slice = self._vector.reshape((self.numberOfSlices, self.rows, self.columns))[:, idx, :]
        elif axis==2:
            idx = 0 if (idx < 0) else (self.columns-1 if idx >= self.columns else idx)
            slice = self._vector.reshape((self.numberOfSlices, self.rows, self.columns))[:, :, idx]
        else:
            idx = 0 if (idx < 0) else (self.numberOfSlices-1 if idx >= self.numberOfSlices else idx)
            slice = self._vector.reshape((self.numberOfSlices, self.rows, self.columns))[idx, :, :]

        if vectorize:
            slice = slice.flatten()
        return slice

    def vectorize(self):
        """consistency method returning the vectorized voxels in the feature volume
        """
        return self._vector
