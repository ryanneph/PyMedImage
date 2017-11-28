"""rttypes.py

Datatypes for general dicom processing including masking, rescaling, and fusion
"""

import sys
import os
import logging
import warnings
import math
import numpy as np
import dicom  # pydicom
import pickle
import scipy.io  # savemat -> save to .mat
import h5py
import struct
import copy
import warnings
from PIL import Image, ImageDraw
from scipy.ndimage import interpolation
from pymedimage import dcmio, misc
from pymedimage.misc import ensure_extension

# initialize module logger
logger = logging.getLogger(__name__)


class FrameOfReference:
    """Defines a dicom frame of reference to which BaseVolumes can be conformed for fusion of pre-registered
    image data
    """
    def __init__(self, start, spacing, size, UID=None):
        """Define a dicom frame of reference

        Args:
            start    -- (x,y,z) describing the start of the FOR (mm)
            spacing  -- (x,y,z) describing the spacing of voxels in each direction (mm)
            size     -- (x,y,z) describing the number of voxels in each direction (integer)
            UID      -- dicom FrameOfReferenceUID can be supplied to support caching in BaseVolume

        Standard Anatomical Directions Apply:
            x -> increasing from patient right to left
            y -> increasing from patient anterior to posterior
            z -> increasing from patient inferior to superior
        """
        self.start = start
        self.spacing = spacing
        self.size = size
        self.UID = UID

    def __repr__(self):
        return '{!s}:\n'.format(self.__class__) + \
               '  start   <mm> (x,y,z): ({:0.3f}, {:0.3f}, {:0.3f})\n'.format(*self.start) + \
               '  spacing <mm> (x,y,z): ({:0.3f}, {:0.3f}, {:0.3f})\n'.format(*self.spacing) + \
               '  size    <mm> (x,y,z): ({:d}, {:d}, {:d})\n'.format(*self.size)

    def __eq__(self, compare):
        if (self.start   == compare.start and
            self.spacing == compare.spacing and
            self.size    == compare.size):
            return True
        else: return False

    def changeSpacing(self, new_spacing):
        """change frameofreference resolution while maintaining same bounding box
        Changes occur in place, self is returned
            Args:
                new_spacing (3-tuple<float>): spacing expressed as (X, Y, Z)
        """
        old_spacing = self.spacing
        old_size = self.size
        self.spacing = new_spacing
        self.size = tuple((np.array(old_size) * np.array(old_spacing) / np.array(self.spacing)).astype(int).tolist())
        return self

    def end(self):
        """Calculates the (x,y,z) coordinates of the end of the frame of reference (mm)
        """
        # compute ends
        end = []
        for i in range(3):
            end.insert(i, self.spacing[i] * self.size[i] + self.start[i])

        return tuple(end)

    def volume(self):
        """Calculates the volume of the frame of reference (mm^3)
        """
        length = []
        end = self.end()
        vol = 1
        for i in range(3):
            length.insert(i, end[i] - self.start[i])
            vol *= length[i]

        return vol

    def getIndices(self, position):
        """Takes a position (x, y, z) and returns the indices at that location for this FrameOfReference

        Args:
            position  -- 3-tuple of position coordinates (mm) in the format: (x, y, z)
        """
        indices = []
        for i in range(3):
            indices.insert(i, math.floor(int(round((position[i] - self.start[i]) / self.spacing[i] ))))

        return tuple(indices)


class ROI:
    """Defines a labeled RTStruct ROI for use in masking and visualization of Radiotherapy contours
    """
    def __init__(self, roicontour=None, structuresetroi=None):
        self.roinumber = None
        self.refforuid = None
        self.frameofreference = None
        self.roiname = None
        self.coordslices = []
        # Cached variables
        self.__cache_densemask = None   # storage for BaseVolume when consecutive calls to
                                        # makeDenseMask are made
                                        # with the same frameofreference object

        if roicontour and structuresetroi:
            self._fromDicomDataset(roicontour, structuresetroi)

    def __repr__(self):
        return '{!s}:\n'.format(self.__class__) + \
               '  roiname: {!s}\n'.format(self.roiname) + \
               '  {!s}\n'.format(self.frameofreference)

    @staticmethod
    def _loadRtstructDicom(rtstruct_path):
        """load rtstruct dicom data from a direct path or containing directory"""
        if (not os.path.exists(rtstruct_path)):
            logger.debug('invalid path provided: "{:s}"'.format(rtstruct_path))
            raise FileNotFoundError

        # check if path is file or dir
        if (os.path.isdir(rtstruct_path)):
            # search recursively for a valid rtstruct file
            ds_list = dcmio.read_dicom_dir(rtstruct_path, recursive=True)
            if (ds_list is None or len(ds_list) == 0):
                logger.debug('no rtstruct datasets found at "{:s}"'.format(rtstruct_path))
                raise Exception
            ds = ds_list[0]
        elif (os.path.isfile(rtstruct_path)):
            ds = dcmio.read_dicom(rtstruct_path)
        return ds

    def _fromDicomDataset(self, roicontour, structuresetroi):
        """takes FrameOfReference object and roicontour/structuresetroi dicom dataset objects and stores
        sorted contour data

        Args:
            roicontour         -- dicom dataset containing contour point coords for all slices
            structuresetroi    -- dicom dataset containing additional information about contour
        """
        self.roinumber = int(structuresetroi.ROINumber)
        self.refforuid = str(structuresetroi.ReferencedFrameOfReferenceUID)
        self.roiname = str(structuresetroi.ROIName)

        # Populate list of coordslices, each containing a list of ordered coordinate points
        contoursequence = roicontour.ContourSequence
        if (len(contoursequence) <= 0):
            logger.debug('no coordinates found in roi: {:s}'.format(self.roiname))
        else:
            logger.debug('loading roi: {:s} with {:d} slices'.format(self.roiname, len(roicontour.ContourSequence)))
            for coordslice in roicontour.ContourSequence:
                points_list = []
                for x, y, z in misc.grouper(3, coordslice.ContourData):
                    points_list.append( (x, y, z) )
                self.coordslices.append(points_list)

            # sort by slice position in ascending order (inferior -> superior)
            self.coordslices.sort(key=lambda coordslice: coordslice[0][2], reverse=False)

            # create frameofreference based on the extents of the roi and apparent spacing
            self.frameofreference = self.getROIExtents()

    @classmethod
    def collectionFromFile(cls, rtstruct_path):
        """loads an rtstruct specified by path and returns a dict of ROI objects

        Args:
            rtstruct_path    -- path to rtstruct.dcm file

        Returns:
            dict<key='contour name', val=ROI>
        """
        ds = cls._loadRtstructDicom(rtstruct_path)

        # parse rtstruct file and instantiate maskvolume for each contour located
        # add each maskvolume to dict with key set to contour name and number?
        if (ds is not None):
            # get structuresetROI sequence
            StructureSetROI_list = ds.StructureSetROISequence
            nContours = len(StructureSetROI_list)
            if (nContours <= 0):
                logger.exception('no contours were found')

            # Add structuresetROI to dict
            StructureSetROI_dict = {StructureSetROI.ROINumber: StructureSetROI
                                    for StructureSetROI
                                    in StructureSetROI_list }

            # get dict containing a contour dataset for each StructureSetROI with a paired key=ROINumber
            ROIContour_dict = {ROIContour.ReferencedROINumber: ROIContour
                               for ROIContour
                               in ds.ROIContourSequence }

            # construct a dict of ROI objects where contour name is key
            roi_dict = {}
            for ROINumber, structuresetroi in StructureSetROI_dict.items():
                roi_dict[structuresetroi.ROIName] = (cls(roicontour=ROIContour_dict[ROINumber],
                                                         structuresetroi=structuresetroi))
            # prune empty ROIs from dict
            for roiname, roi in dict(roi_dict).items():
                if (roi.coordslices is None or len(roi.coordslices) <= 0):
                    logger.debug('pruning empty ROI: {:s} from loaded ROIs'.format(roiname))
                    del roi_dict[roiname]

            logger.debug('loaded {:d} ROIs succesfully'.format(len(roi_dict)))
            return roi_dict
        else:
            logger.exception('no dataset was found')

    @staticmethod
    def getROINames(rtstruct_path):
        ds = ROI._loadRtstructDicom(rtstruct_path)

        if (ds is not None):
            # get structuresetROI sequence
            StructureSetROI_list = ds.StructureSetROISequence
            nContours = len(StructureSetROI_list)
            if (nContours <= 0):
                logger.exception('no contours were found')

            roi_names = []
            for structuresetroi in StructureSetROI_list:
                roi_names.append(structuresetroi.ROIName)

            return roi_names
        else:
            logger.exception('no dataset was found')

    def makeDenseMaskSlice(self, position, frameofreference=None):
        """Takes a FrameOfReference and constructs a dense binary mask for the ROI (1 inside ROI, 0 outside)
        as a numpy 2dArray

        Args:
            position           -- position of the desired slice (mm) within the frameofreference along z-axis
            frameofreference   -- FrameOfReference that defines the position of ROI and size of dense volume

        Returns:
            numpy 2dArray
        """
        # get FrameOfReference params
        if (frameofreference is None):
            if (self.frameofreference is not None):
                frameofreference = self.frameofreference
            else:
                logger.exception('no frame of reference provided')
                raise Exception
        xstart, ystart, zstart = frameofreference.start
        xspace, yspace, zspace = frameofreference.spacing
        cols, rows, depth = frameofreference.size

        # get nearest coordslice
        minerror = 5000
        coordslice = None
        ### REVISIT THE CORRECT SETTING OF TOLERANCE TODO
        tolerance = self.frameofreference.spacing[2]*4 - 1e-9  # if upsampling too much then throw error
        for slice in self.coordslices:
            # for each list of coordinate tuples - check the slice for distance from position
            error = abs(position - slice[0][2])
            if error <= minerror:
                # if minerror != 5000:
                #     logger.debug('position:{:0.3f} | slicepos:{:0.3f}'.format(position, slice[0][2]))
                #     logger.debug('improved with error {:f}'.format(error))
                minerror = error
                coordslice = slice
                # logger.debug('updating slice')
            else:
                # we've already passed the nearest slice, break
                break

        # check if our result is actually valid or we just hit the end of the array
        if coordslice and minerror >= tolerance:
            logger.debug('No slice found within {:f} mm of position {:f}'.format(tolerance, position))
            # print(minerror, tolerance)
            # print(position)
            # print(zstart, zspace*depth)
            # for slice in self.coordslices:
            #     if abs(slice[0][2]-position) < 100:
            #         print(slice[0][2])
            return np.zeros((rows, cols))
            # raise Exception('Attempt to upsample ROI to densearray beyond 5x')
        logger.debug('slice found at {:f} for position query at {:f}'.format(coordslice[0][2], position))

        # get coordinate values
        index_coords = []
        for x, y, z in coordslice:
            # shift x and y and scale appropriately
            x_idx = int(round((x-xstart)/xspace))
            y_idx = int(round((y-ystart)/yspace))
            index_coords.append( (x_idx, y_idx) )

        # use PIL to draw the polygon as a dense image (PIL uses shape: (width, height))
        im = Image.new('1', (cols, rows), color=0)
        imdraw = ImageDraw.Draw(im)
        imdraw.polygon(index_coords, fill=1, outline=None)
        del imdraw

        # convert from PIL image to np.ndarray and threshold to binary
        return np.array(im.getdata()).reshape((rows, cols))

    def makeDenseMask(self, frameofreference=None):
        """Takes a FrameOfReference and constructs a dense binary mask for the ROI (1 inside ROI, 0 outside)
        as a BaseVolume

        Args:
            frameofreference   -- FrameOfReference that defines the position of ROI and size of dense volume

        Returns:
            BaseVolume
        """
        # get FrameOfReference params
        if (frameofreference is None):
            if (self.frameofreference is not None):
                frameofreference = self.frameofreference
            else:
                logger.exception('no frame of reference provided')
                raise Exception

        # check cache for similarity between previously and currently supplied frameofreference objects
        if (self.__cache_densemask is not None
                and frameofreference == self.__cache_densemask.frameofreference):
            # cached mask frameofreference is similar to current, return cached densemask volume
            # logger.debug('using cached dense mask volume')
            return self.__cache_densemask
        else:
            xstart, ystart, zstart = frameofreference.start
            xspace, yspace, zspace = frameofreference.spacing
            cols, rows, depth = frameofreference.size

            # generate binary mask for each slice in frameofreference
            maskslicearray_list = []
            # logger.debug('making dense mask volume from z coordinates: {:f} to {:f}'.format(
            #              zstart, (zspace * (depth+1) + zstart)))
            for i in range(depth):
                position = zstart + i * zspace
                # get a slice at every position within the current frameofreference
                densemaskslice = self.makeDenseMaskSlice(position, frameofreference)
                maskslicearray_list.append(densemaskslice.reshape((1, *densemaskslice.shape)))

            # construct BaseVolume from dense slice arrays
            densemask = BaseVolume.fromArray(np.concatenate(maskslicearray_list, axis=0), frameofreference)
            self.__cache_densemask = densemask
            return densemask

    def getROIExtents(self):
        """Creates a tightly bound frame of reference around the ROI which allows visualization in a cropped
        frame
        """
        # guess at spacing and assign arbitrarily where necessary
        # get list of points first
        point_list = []
        for slice in self.coordslices:
            for point3d in slice:
                point_list.append(point3d)

        # set actually z spacing estimated from separation of coordslice point lists
        min_z_space = 9999
        prev_z = point_list[0][2]
        for point3d in point_list[1:]:
            z = point3d[2]
            this_z_space = abs(z-prev_z)
            if (this_z_space > 0 and this_z_space < min_z_space):
                min_z_space = this_z_space
            prev_z = z

        if (min_z_space <= 0 or min_z_space > 10):
            # unreasonable result found, arbitrarily set
            new_z_space = 1
            logger.debug('unreasonable z_spacing found: {:0.3f}, setting to {:0.3f}'.format(
                min_z_space, new_z_space))
            min_z_space = new_z_space
        else:
            logger.debug('estimated z_spacing: {:0.3f}'.format(min_z_space))

        # arbitrarily set spacing
        spacing = (1, 1, min_z_space)

        # get start and end of roi volume extents
        global_limits = {'xmax': -5000,
                         'ymax': -5000,
                         'zmax': -5000,
                         'xmin': 5000,
                         'ymin': 5000,
                         'zmin': 5000 }
        for slice in self.coordslices:
            # convert coords list to ndarray
            coords = np.array(slice)
            (xmin, ymin, zmin) = tuple(coords.min(axis=0, keepdims=False))
            (xmax, ymax, zmax) = tuple(coords.max(axis=0, keepdims=False))

            # update limits
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

        # build FrameOfReference
        start = (global_limits['xmin'],
                 global_limits['ymin'],
                 global_limits['zmin'] )
        size = (int((global_limits['xmax'] - global_limits['xmin']) / spacing[0]),
                int((global_limits['ymax'] - global_limits['ymin']) / spacing[1]),
                int((global_limits['zmax'] - global_limits['zmin']) / spacing[2]) )

        logger.debug('ROIExtents:\n'
                     '    start:   {:s}\n'
                     '    spacing: {:s}\n'
                     '    size:    {:s}'.format(str(start), str(spacing), str(size)))
        frameofreference = FrameOfReference(start, spacing, size, UID=None)
        return frameofreference

    def toPickle(self, path):
        """convenience function for storing ROI to pickle file"""
        warnings.warn('ROI.toPickle() will be deprecated soon in favor of other serialization methods.', DeprecationWarning)
        _dirname = os.path.dirname(path)
        if (_dirname and _dirname is not ''):
            os.makedirs(_dirname, exist_ok=True)
        with open(path, 'wb') as p:
            pickle.dump(self, p)

    @staticmethod
    def fromPickle(path):
        """convenience function for restoring ROI from pickle file"""
        warnings.warn('ROI.fromPickle() will be deprecated soon in favor of other serialization methods.', DeprecationWarning)
        with open(path, 'rb') as p:
            return pickle.load(p)

    def toHDF5(self, path):
        """serialize object to file in h5 format"""
        path = ensure_extension(path, '.h5')
        with h5py.File(path, 'w') as f:
            # store attributes
            f.attrs['roinumber'] = self.roinumber
            f.attrs['roiname'] = self.roiname
            f.attrs['refforuid'] = self.refforuid
            f.attrs['FrameOfReference.start'] = self.frameofreference.start
            f.attrs['FrameOfReference.spacing'] = self.frameofreference.spacing
            f.attrs['FrameOfReference.size'] = self.frameofreference.size
            f.attrs['fileversion'] = '1.0'

            # store datasets
            g = f.create_group('coordslices')
            g.attrs['Nslices'] = len(self.coordslices)
            for i, slice in enumerate(self.coordslices):
                arr = np.array(slice)
                g.create_dataset('{:d}'.format(i), data=arr)

    @classmethod
    def fromHDF5(cls, path):
        """reconstruct object from serialized data in h5 format"""
        self = cls()
        with h5py.File(path, 'r') as f:
            self.roinumber = int(f.attrs['roinumber'])
            self.roiname = str(f.attrs['roiname'])
            self.refforuid = str(f.attrs['refforuid'])
            self.frameofreference = FrameOfReference(
                tuple(f.attrs['FrameOfReference.start']),
                tuple(f.attrs['FrameOfReference.spacing']),
                tuple(f.attrs['FrameOfReference.size'])
            )
            self.coordslices = []
            for k in sorted(f['coordslices'].keys()):
                points = []
                data = f['coordslices'][k]
                npdata = np.empty(data.shape, dtype=data.dtype)
                data.read_direct(npdata)
                for i in range(data.shape[0]):
                    points.append(tuple(npdata[i, :]))
                self.coordslices.append(points)
        return self



class BaseVolume:
    """Defines basic storage for volumetric voxel intensities within a dicom FrameOfReference
    """
    def __init__(self):
        """Entrypoint to class, initializes members
        """
        self.array = None
        self.frameofreference = None
        self.modality = None
        self.feature_label = None

    def __repr__(self):
        return '{!s}:\n'.format(self.__class__) + \
               '  modality: {!s}\n'.format(self.modality) + \
               '  feature_label: {!s}\n'.format(self.feature_label) + \
               '  {!s}\n'.format(self.frameofreference)

    # CONSTRUCTOR METHODS
    #  @staticmethod
    #  def _getAttrMap():
    #      return {
    #          'arraydata': 'array',
    #          'size': 'FrameOfReference.size',
    #          'start': 'FrameOfReference.start',
    #          'spacing': 'FrameOfReference.spacing',
    #          'for_uid': 'FrameOfReference.uid',
    #          'modality': 'modality',
    #          'feature_label': 'feature_label',
    #          'order': 'ZYX'
    #      }

    #  @classmethod
    #  def _fromDataDict(cls, datadict):
    #      """constructor helper that standardized construction from different filetypes using a common mapping
    #      """
    #      self = cls() # instantiate
    #      invertattrs = (datadict.pop('order', default='ZYX') == 'ZYX')
    #      attrmap = BaseVolume._getAttrMap()
    #      for k, v in datadict.items():
    #          self.__setattr__(attrmap[k], v)

    def _getDataDict(self):
        xstr = misc.xstr  # shorter call-name for use in function
        return {'arraydata':     self.array,
                'size':          self.frameofreference.size[::-1],
                'start':         self.frameofreference.start[::-1],
                'spacing':       self.frameofreference.spacing[::-1],
                'for_uid':       xstr(self.frameofreference.UID),
                'modality':      xstr(self.modality),
                'feature_label': xstr(self.feature_label),
                'order':         'ZYX'
                }


    @classmethod
    def fromArray(cls, array, frameofreference):
        """Constructor: from a numpy array and FrameOfReference object

        Args:
            array             -- numpy array
            frameofreference  -- FrameOfReference object
        """
        # ensure array matches size in frameofreference
        self = cls()
        self.array = array.reshape(frameofreference.size[::-1])
        self.frameofreference = frameofreference

        return self

    @classmethod
    def fromDir(cls, path, recursive=False):
        """constructor: takes path to directory containing dicom files and builds a sorted array

        Args:
            recursive -- find dicom files in all subdirectories?
        """
        # get the datasets from files
        dataset_list = dcmio.read_dicom_dir(path, recursive=recursive)

        # pass dataset list to constructor
        self = cls.fromDatasetList(dataset_list)

        return self

    @classmethod
    def fromBinary(cls, path, frameofreference):
        """constructor: takes path to binary file (neylon .raw)
        data is organized as binary float array in row-major order

        Args:
            path (str): path to .raw file in binary format
            frameofreference (FOR): most importantly defines mapping from 1d to 3d array
        """
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in ['.raw', '.bin']:
            raise Exception('data is not formatted properly. must be one of [.raw, .bin]')

        if not isinstance(frameofreference, FrameOfReference):
            if not isinstance(frameofreference, tuple):
                raise TypeError('frameofreference must be a valid FrameOfReference or tuple of dimensions')
            frameofreference = FrameOfReference(start=(0,0,0), spacing=(1,1,1), size=frameofreference)

        with open(path, mode='rb') as f:
            flat = f.read()
        _shape = frameofreference.size[::-1]
        _expected_n = np.product(_shape)
        _n = int(os.path.getsize(path)/struct.calcsize('f'))
        if _n != _expected_n:
            raise Exception('filesize ({:f}) doesn\'t match expected ({:f}) size'.format(
                os.path.getsize((path)), struct.calcsize('f')*_expected_n
            ))
        s = struct.unpack('f'*_n, flat)
        vol = np.array(s).reshape(_shape)
        #  vol[vol>1e10] = 0
        #  vol[vol<-1e10] = 0
        return cls.fromArray(vol, frameofreference)


    @classmethod
    def fromDatasetList(cls, dataset_list):
        """constructor: takes a list of dicom slice datasets and builds a BaseVolume array
        Args:
            slices
        """
        self = cls()
        if (dataset_list is None):
            raise ValueError('no valid dataset_list provided')

        # check that all elements are valid slices, if not remove and continue
        nRemoved = 0
        for i, slice in enumerate(dataset_list):
            if (not isinstance(slice, dicom.dataset.Dataset)):
                logger.debug('invalid type ({t:s}) at idx {i:d}. removing.'.format(
                    t=str(type(slice)),
                    i=i ) )
                dataset_list.remove(slice)
                nRemoved += 1
            elif (len(slice.dir('ImagePositionPatient')) == 0):
                logger.debug('invalid .dcm image at idx {:d}. removing.'.format(i))
                dataset_list.remove(slice)
                nRemoved += 1
        if (nRemoved > 0):
            logger.info('# slices removed with invalid types: {:d}'.format(nRemoved))

        # sort datasets by increasing slicePosition (inferior -> superior)
        dataset_list.sort(key=lambda dataset: dataset.ImagePositionPatient[2], reverse=False)

        # build object properties
        start = dataset_list[0].ImagePositionPatient
        spacing = (*dataset_list[0].PixelSpacing, dataset_list[0].SliceThickness)
        try:
            # some modalities don't provide NumberOfSlices attribute
            size = (dataset_list[0].Columns, dataset_list[0].Rows, dataset_list[0].NumberOfSlices)
        except:
            # use length of list instead
            size = (dataset_list[0].Columns, dataset_list[0].Rows, len(dataset_list))

        UID = dataset_list[0].FrameOfReferenceUID
        self.frameofreference = FrameOfReference(start, spacing, size, UID)

        # standardize modality labels
        mod = dataset_list[0].Modality
        if (mod == 'PT'):
            mod = 'PET'
        self.modality = mod

        # construct 3dArray
        array_list = []
        for dataset in dataset_list:
            array = dataset.pixel_array
            factor = dataset.RescaleSlope
            offset = dataset.RescaleIntercept
            array = np.add(np.multiply(array, factor), offset)
            array = array.reshape((1, array.shape[0], array.shape[1]))
            array_list.append(array)

        # stack arrays
        self.array = np.concatenate(array_list, axis=0)
        #  self.array = self.array.astype(int)
        return self

    @classmethod
    def fromPickle(cls, path):
        """initialize BaseVolume from unchanging format so features can be stored and recalled long term
        """
        warnings.warn('{!s}.fromPickle() will be deprecated soon in favor of other serialization methods.'.format(cls.__name__), DeprecationWarning)
        if (not os.path.exists(path)):
            logger.info('file at path: {:s} doesn\'t exists'.format(path))
        with open(path, 'rb') as p:
            # added to fix broken module refs in old pickles
            sys.modules['utils'] = sys.modules[__name__]
            sys.modules['utils.rttypes'] = sys.modules[__name__]
            basevolumeserial = pickle.load(p)
            del sys.modules['utils.rttypes']
            del sys.modules['utils']

        # import data to this object
        try:
            self = cls()
            self.array = basevolumeserial.dataarray
            self.frameofreference = FrameOfReference(basevolumeserial.startposition,
                                                     basevolumeserial.spacing,
                                                     basevolumeserial.size)
            self.modality = basevolumeserial.modality
            self.feature_label = basevolumeserial.feature_label
        except:
            raise SerialOutdatedError()
        return self

    def toPickle(self, path):
        """store critical data to unchanging format that can be pickled long term
        """
        warnings.warn('{!s}.toPickle() will be deprecated soon in favor of other serialization methods.'.format(self.__class__), DeprecationWarning)
        basevolumeserial = BaseVolumeSerial()
        basevolumeserial.startposition = self.frameofreference.start
        basevolumeserial.spacing = self.frameofreference.spacing
        basevolumeserial.size = self.frameofreference.size
        basevolumeserial.dataarray = self.array
        basevolumeserial.modality = self.modality
        basevolumeserial.feature_label = self.feature_label

        path = ensure_extension(path, '.pickle')
        _dirname = os.path.dirname(path)
        if (_dirname and _dirname is not ''):
            os.makedirs(_dirname, exist_ok=True)
        with open(path, 'wb') as p:
            pickle.dump(basevolumeserial, p)

    @classmethod
    def fromMatlab(cls, path):
        """restore BaseVolume from .mat file that was created using BaseVolume.toMatlab()
        """
        extract_str = misc.numpy_safe_string_from_array
        data = scipy.io.loadmat(path, appendmat=True)
        #  for key, obj in data.items():
        #      print('{!s}({!s}: {!s}'.format(key, type(obj), obj))
        converted_data = {
            'arraydata': data['arraydata'],
            'size': tuple(data['size'][0,:])[::-1],
            'start': tuple(data['start'][0,:])[::-1],
            'spacing': tuple(data['spacing'][0,:])[::-1],
            'for_uid': extract_str(data['for_uid']),
            'modality': extract_str(data['modality']),
            'feature_label': extract_str(data['feature_label']),
            'scale': data['scale'].item(),
            'offset': data['offset'].item(),
            'order': extract_str(data['order'])
        }

        # construct new volume
        self = cls()
        self.array = converted_data['arraydata']
        self.frameofreference = FrameOfReference(converted_data['start'],
                                                 converted_data['spacing'],
                                                 converted_data['size'],
                                                 converted_data['for_uid'])
        self.modality = converted_data['modality']
        self.feature_label = converted_data['feature_label']

        return self


    def toMatlab(self, path, compress=False):
        """store critical data to .mat file compatible with matlab loading
        This is essentially .toPickle() with compat. for matlab reading

        Optional Args:
            compress (bool): compress dataarray at the cost of write speed
        """
        # first represent as dictionary for savemat()
        data = self._getDataDict()
        data['order'] = 'ZYX'
        path = ensure_extension(path, '.mat')

        # write to .mat
        scipy.io.savemat(path, data, appendmat=False, format='5', long_field_names=False,
                         do_compression=compress, oned_as='row')

    def toHDF5(self, path, compress=False):
        """store object to hdf5 file with image data stored as dataset and metadata as attributes"""
        data = self._getDataDict()
        arraydata = data.pop('arraydata')
        path = ensure_extension(path, '.h5')
        with h5py.File(path, 'w') as f:
            for k, v in data.items():
                f.attrs.__setitem__(k, v)
            f.create_dataset('arraydata', data=arraydata)
            f.attrs['fileversion'] = '1.0'

    @classmethod
    def fromHDF5(cls, path):
        """restore objects from hdf5 file with image data stored as dataset and metadata as attributes"""
        extract_str = misc.numpy_safe_string_from_array
        # construct new volume
        self = cls()
        with h5py.File(path, 'r') as f:
            ad = f['arraydata']
            self.array = np.empty(ad.shape)
            ad.read_direct(self.array)
            self.frameofreference = FrameOfReference(
                tuple(f.attrs['start'])[::-1],
                tuple(f.attrs['spacing'])[::-1],
                tuple(f.attrs['size'])[::-1],
                extract_str(f.attrs['for_uid'])
            )
            self.modality = f.attrs['modality']
            self.feature_label = f.attrs['feature_label']
        return self


    # PUBLIC METHODS
    def conformTo(self, frameofreference):
        """Resamples the current BaseVolume to the supplied FrameOfReference

        Args:
            frameofreference   -- FrameOfReference object to resample the Basevolume to

        Returns:
            BaseVolume
        """
        # conform volume to alternate FrameOfReference
        if (frameofreference is None):
            logger.exception('no FrameOfReference provided')
            raise ValueError
        elif (ROI.__name__ in (str(type(frameofreference)))):
            frameofreference = frameofreference.frameofreference
        elif (FrameOfReference.__name__ not in str(type(frameofreference))):  # This is an ugly way of type-checking but cant get isinstance to see both as the same
            logger.exception(('supplied frameofreference of type: "{:s}" must be of the type: "FrameOfReference"'.format(
                str(type(frameofreference)))))
            raise TypeError

        if self.frameofreference == frameofreference:
            return self

        # first match self resolution to requested resolution
        zoomarray, zoomFOR = self._resample(frameofreference.spacing)

        # crop to active volume of requested FrameOfReference in frameofreference
        xstart_idx, ystart_idx, zstart_idx = zoomFOR.getIndices(frameofreference.start)
        # xend_idx, yend_idx, zend_idx = zoomFOR.getIndices(frameofreference.end())
        # force new size to match requested FOR size
        xend_idx, yend_idx, zend_idx = tuple((np.array((xstart_idx, ystart_idx, zstart_idx)) + np.array(frameofreference.size)).tolist())
        try:
            cropped = zoomarray[zstart_idx:zend_idx, ystart_idx:yend_idx, xstart_idx:xend_idx]
            zoomFOR.start = frameofreference.start
            zoomFOR.size = cropped.shape[::-1]
        except:
            logger.exception('request to conform to frame outside of volume\'s frame of reference failed')
            raise Exception()

        # reconstruct volume from resampled array
        resampled_volume = MaskableVolume.fromArray(cropped, zoomFOR)
        resampled_volume.modality = self.modality
        resampled_volume.feature_label = self.feature_label
        return resampled_volume

    def _resample(self, new_voxelsize, order=3):
        if new_voxelsize == self.frameofreference.spacing:
            # no need to resample
            return (self.array, self.frameofreference)

        # voxelsize spec is in order (X,Y,Z) but array is kept in order (Z, Y, X)
        zoom_factors = np.true_divide(self.frameofreference.spacing, new_voxelsize)[::-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zoomarray = interpolation.zoom(self.array, zoom_factors, order=order, mode='nearest')
        zoomFOR = FrameOfReference(self.frameofreference.start, new_voxelsize, zoomarray.shape[::-1])
        return (zoomarray, zoomFOR)

    def resample(self, new_voxelsize, order=3):
        """resamples volume to new voxelsize

        Args:
            new_voxelsize: 3 tuple of voxel size in mm in the order (X, Y, Z)

        """
        zoomarray, zoomFOR = self._resample(new_voxelsize, order)
        new_vol = MaskableVolume.fromArray(zoomarray, zoomFOR)
        new_vol.modality = self.modality
        new_vol.feature_label = self.feature_label
        return new_vol

    def getSlice(self, idx, axis=0, flatten=False):
        """Extracts 2dArray of idx along the axis.
        Args:
            idx       -- idx identifying the slice along axis

        Optional Args:
            axis      -- specifies axis along which to extract
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            flatten   -- return a 1darray in depth-stacked row-major order
        """
        cols, rows, depth = self.frameofreference.size

        # perform index bounding
        if (axis==0):
            if (idx < 0 or idx >= depth):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(depth-1))
                raise IndexError
            thisslice = self.array[idx, :, :]
        elif (axis==1):
            if (idx < 0 or idx >= rows):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(rows-1))
                raise IndexError
            thisslice = self.array[:, idx, :]
        elif (axis==2):
            if (idx < 0 or idx >= cols):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(cols-1))
                raise IndexError
            thisslice = self.array[:, :, idx]
        else:
            logger.exception('invalid axis supplied. must be between 0 -> 2')
            raise ValueError

        # RESHAPE
        if (flatten):
            thisslice = thisslice.flatten(order='C').reshape((-1, 1))

        return thisslice

    def vectorize(self):
        """flatten self.array in stacked-depth row-major order
        """
        return self.array.flatten(order='C').reshape((-1, 1))

    def get_val(self, z, y, x):
        """take xyz indices and return the value in array at that location
        """
        frameofreference = self.frameofreference
        # get volume size
        (cols, rows, depth) = frameofreference.size

        # perform index bounding
        if (x < 0 or x >= cols):
            logger.exception('x index ({:d}) out of bounds. must be between 0 -> {:d}'.format(x, cols-1))
            raise IndexError
        if (y < 0 or y >= rows):
            logger.exception('y index ({:d}) out of bounds. must be between 0 -> {:d}'.format(y, rows-1))
            raise IndexError
        if (z < 0 or z >= depth):
            logger.exception('z index ({:d}) out of bounds. must be between 0 -> {:d}'.format(z, depth-1))
            raise IndexError

        return self.array[z, y, x]

    def set_val(self, z, y, x, value):
        """take xyz indices and value and reassing the value in array at that location
        """
        frameofreference = self.frameofreference
        # get volume size
        (cols, rows, depth) = frameofreference.size

        # perform index bounding
        if (x < 0 or x >= cols):
            logger.exception('x index ({:d}) out of bounds. must be between 0 -> {:d}'.format(x, cols-1))
            raise IndexError
        if (y < 0 or y >= rows):
            logger.exception('y index ({:d}) out of bounds. must be between 0 -> {:d}'.format(y, rows-1))
            raise IndexError
        if (z < 0 or z >= depth):
            logger.exception('z index ({:d}) out of bounds. must be between 0 -> {:d}'.format(z, depth-1))
            raise IndexError

        # reassign value
        self.array[z, y, x] = value


class MaskableVolume(BaseVolume):
    """Subclass of BaseVolume that adds support for ROI masking of the data array
    """
    def __init__(self):
        """Entry point to class"""
        # call to base class initializer
        super().__init__()

    def conformTo(self, frameofreference):
        """Resamples the current MaskableVolume to the supplied FrameOfReference and returns a new Volume

        Args:
            frameofreference   -- FrameOfReference object to resample the MaskableVolume to

        Returns:
            MaskableVolume
        """
        base = super().conformTo(frameofreference)
        maskable = MaskableVolume().fromBaseVolume(base)
        return maskable

    # CONSTRUCTOR METHODS
    def deepCopy(self):
        """makes deep copy of self and returns the copy"""
        copy_vol = MaskableVolume()
        copy_vol.array = copy.deepcopy(self.array)
        copy_vol.frameofreference = copy.deepcopy(self.frameofreference)
        copy_vol.modality = self.modality
        copy_vol.feature_label = self.feature_label
        return copy_vol

    def fromBaseVolume(self, base):
        """promotion constructor that converts baseVolume to MaskableVolume, retaining member variables

        Args:
            base -- BaseVolume object

        Returns:
            MaskableVolume
        """
        # copy attributes
        self.array = base.array
        self.frameofreference = copy.deepcopy(base.frameofreference)
        self.modality = base.modality
        self.feature_label = base.feature_label
        return self

    # PUBLIC METHODS
    def getSlice(self, idx, axis=0, flatten=False, roi=None):
        """Extracts 2dArray of idx along the axis.
        Args:
            idx     -- idx identifying the slice along axis

        Optional Args:
            axis    --  specifies axis along which to extract
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            flatten -- return a 1darray in depth-stacked row-major order
            roi     -- ROI object that can be supplied to mask the output of getSlice
         """
        # call to base class
        slicearray = super().getSlice(idx, axis, flatten)

        # get equivalent slice from densemaskarray
        if (roi is not None):
            maskslicearray = roi.makeDenseMask(self.frameofreference).getSlice(idx, axis, flatten)
            # apply mask
            slicearray = np.multiply(slicearray, maskslicearray)

        return slicearray

    def vectorize(self, roi=None):
        """flatten self.array in stacked-depth row-major order

        Args:
            roi  -- ROI object that can be supplied to mask the output of getSlice
        """
        array = self.array.flatten(order='C').reshape((-1, 1))

        # get equivalent array from densemaskarray
        if (roi is not None):
            maskarray = roi.makeDenseMask(self.frameofreference).vectorize()
            # apply mask
            array = np.multiply(array, maskarray)

        return array

    def applyMask(self, roi):
        """Applies roi mask to entire array and returns masked copy of class

        Args:
            roi -- ROI object that supplies the mask definition
        """
        volume_copy = self.deepCopy()
        masked_array = self.vectorize(roi).reshape(self.frameofreference.size[::-1])
        volume_copy.array = masked_array
        return volume_copy


class SerialOutdatedError(Exception):
    def __init__(self):
        super().__init__('a missing value was requested from a BaseVolumeSerial object')


class BaseVolumeSerial:
    """Defines common object that can store feature data for long term I/O
    """
    def __init__(self):
        self.dataarray     = None  # numpy ndarray
        self.startposition = None  # (x, y, z)<float>
        self.spacing       = None  # (x, y, z)<float>
        self.size          = None  # (x, y, z)<integer>
        self.modality      = None  # string
        self.feature_label = None  # string
