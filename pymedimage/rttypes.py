"""rttypes.py

Datatypes for general dicom processing including masking, rescaling, and fusion
"""

import os
import logging
import numpy as np
import dicom  # pydicom
import pickle
from PIL import Image, ImageDraw
from scipy.ndimage import interpolation
from . import dcmio, misc

# initialize module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class RescaleParams:
    """Defines the scale and offset necessary for linear rescaling operation of dicom data
    """
    def __init__(self, scale=1, offset=0):
        """initialize with params"""
        self.scale = scale
        self.offset = offset


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
            indices.insert(i, int(round((position[i] - self.start[i]) / self.spacing[i] )))

        return tuple(indices)


class ROI:
    """Defines a labeled RTStruct ROI for use in masking and visualization of Radiotherapy contours
    """
    def __init__(self, frameofreference, roicontour, structuresetroi, verbose=False):
        """takes FrameOfReference object and roicontour/structuresetroi dicom dataset objects and stores
        sorted contour data

        Args:
            frameofreference   -- FrameOfReference object providing details necessary for dense mask creation
            roicontour         -- dicom dataset containing contour point coords for all slices
            structuresetroi    -- dicom dataset containing additional information about contour
        """
        self.roinumber = structuresetroi.ROINumber
        self.refforuid = structuresetroi.ReferencedFrameOfReferenceUID
        self.roiname = structuresetroi.ROIName
        # Cached variables
        self.__cache_densemask = None   # storage for BaseVolume when consecutive calls to
                                        # makeDenseMask are made
                                        # with the same frameofreference object

        # Populate list of coordslices, each containing a list of ordered coordinate points
        contoursequence = roicontour.ContourSequence
        if (len(contoursequence) <= 0):
            if (verbose):
                logger.info('no coordinates found in roi: {:s}'.format(self.roiname))
            self.coordslices = None
        else:
            self.coordslices = []
            if (verbose):
                logger.info('loading roi: {:s} with {:d} slices'.format(self.roiname, len(roicontour.ContourSequence)))
            for coordslice in roicontour.ContourSequence:
                points_list = []
                for x, y, z in misc.grouper(3, coordslice.ContourData):
                    points_list.append( (x, y, z) )
                self.coordslices.append(points_list)

            # sort by slice position in ascending order (inferior -> superior)
            self.coordslices.sort(key=lambda coordslice: coordslice[0][2], reverse=False)

            # if no refframe specified, create one based on the extents of the roi and apparent spacing
            if (frameofreference is not None):
                self.frameofreference = frameofreference
            else:
                self.frameofreference = self.getROIExtents(verbose)


    def makeDenseMaskSlice(self, position, frameofreference=None, verbose=False):
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
                logger.info('no frame of reference provided')
                raise Exception
        xstart, ystart, zstart = frameofreference.start
        xspace, yspace, zspace = frameofreference.spacing
        cols, rows, depth = frameofreference.size

        # get nearest coordslice
        minerror = 5000
        coordslice = None
        tolerance = zspace
        for slice in self.coordslices:
            # for each list of coordinate tuples - check the slice for distance from position
            error = abs(position - slice[0][2])
            if error < minerror:
                if (verbose):
                    if minerror != 5000:
                        logger.info(position, slice[0][2])
                        logger.info('improved with error {:f}'.format(error))
                minerror = error
            if (error <= minerror):
                coordslice = slice
                if (verbose):
                    logger.info('updating slice')
            else:
                # we've already passed the nearest slice, break
                break

        # check if our result is actually valid or we just hit the end of the array
        if minerror >= tolerance:
            if (verbose):
                logger.info('No slice found within {:f} mm of position {:f}'.format(tolerance, position))
            return np.ones((rows, cols))
        if (verbose):
            logger.info('slice found at {:f} for position query at {:f}'.format(coordslice[0][2], position))

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

    def makeDenseMask(self, frameofreference=None, verbose=False):
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
                logger.info('no frame of reference provided')
                raise Exception

        # check cache for similarity between previously and currently supplied frameofreference objects
        if (self.__cache_densemask is not None
                and frameofreference == self.__cache_densemask.frameofreference):
            # cached mask frameofreference is similar to current, return cached densemask volume
            if (verbose):
                logger.info('using cached dense mask volume')
            return self.__cache_densemask
        else:
            xstart, ystart, zstart = frameofreference.start
            xspace, yspace, zspace = frameofreference.spacing
            cols, rows, depth = frameofreference.size

            # generate binary mask for each slice in frameofreference
            maskslicearray_list = []
            if (verbose):
                print('making dense mask volume from z coordinates: {:f} to {:f}'.format(
                    zstart, (zspace * (depth+1) + zstart)))
            for i in range(depth):
                position = zstart + i * zspace
                # get a slice at every position within the current frameofreference
                densemaskslice = self.makeDenseMaskSlice(position, frameofreference, verbose=verbose)
                maskslicearray_list.append(densemaskslice.reshape((1, *densemaskslice.shape)))

            # construct BaseVolume from dense slice arrays
            densemask = BaseVolume().fromArray(np.concatenate(maskslicearray_list, axis=0), frameofreference)
            self.__cache_densemask = densemask
            return densemask

    def getROIExtents(self, verbose=False):
        """Creates a tightly bound frame of reference around the ROI which allows visualization in a cropped
        frame
        """
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
        # assume sane spacing
        spacing = (1, 1, 1)

        start = (global_limits['xmin'],
                 global_limits['ymin'],
                 global_limits['zmin'] )
        size = (int((global_limits['xmax'] - global_limits['xmin']) / spacing[0]),
                int((global_limits['ymax'] - global_limits['ymin']) / spacing[1]),
                int((global_limits['zmax'] - global_limits['zmin']) / spacing[2]) )

        if (verbose):
            print('ROIExtents:\n'
                  '    start:   {:s}\n'
                  '    spacing: {:s}\n'
                  '    size:    {:s}'.format(str(start), str(spacing), str(size)))
        frameofreference = FrameOfReference(start, spacing, size, UID=None)
        return frameofreference


class BaseVolume:
    """Defines basic storage for volumetric voxel intensities within a dicom FrameOfReference
    """
    def __init__(self):
        """Entrypoint to class, initializes members
        """
        self.array = None
        self.frameofreference = None
        self.rescaleparams = None
        self.modality = None

    # CONSTRUCTOR METHODS
    def fromArray(self, array, frameofreference):
        """Constructor: from a numpy array and FrameOfReference object

        Args:
            array             -- numpy array
            frameofreference  -- FrameOfReference object
        """
        self.array = array
        self.frameofreference = frameofreference

        return self

    def fromDir(self, path, recursive=False):
        """constructor: takes path to directory containing dicom files and builds a sorted array

        Args:
            recursive -- find dicom files in all subdirectories?
        """
        # get the datasets from files
        dataset_list = dcmio.read_dicom_dir(path, recursive=recursive)

        # pass dataset list to constructor
        self.fromDatasetList(dataset_list)

        return self

    def fromDatasetList(self, dataset_list):
        """constructor: takes a list of dicom slice datasets and builds a BaseVolume array
        Args:
            slices
        """
        if (dataset_list is None):
            logger.info('no valid dataset_list provided')
            raise ValueError

        # check that all elements are valid slices, if not remove and continue
        nRemoved = 0
        for i, slice in enumerate(dataset_list):
            if (not isinstance(slice, dicom.dataset.Dataset)):
                logger.info('invalid type ({t:s}) at idx {i:d}. removing.'.format(
                    t=str(type(slice)),
                    i=i ) )
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
        self.modality = dataset_list[0].Modality

        # construct 3dArray
        array_list = []
        for dataset in dataset_list:
            array = dataset.pixel_array
            array = array.reshape((1, array.shape[1], array.shape[0]))
            array_list.append(array)

        # stack arrays
        self.array = np.concatenate(array_list, axis=0)

        return self

    def fromPickle(self, pickle_path):
        """initialize BaseVolume from unchanging format so features can be stored and recalled long term
        """
        if (not os.path.exists(pickle_path)):
            logger.info('file at path: {:s} doesn\'t exists'.format(pickle_path))
        with open(pickle_path, 'rb') as p:
            basevolumepickle = pickle.load(p)

        # import data to this object
        self.array = basevolumepickle.dataarray
        self.frameofreference = FrameOfReference(basevolumepickle.startposition,
                                                 basevolumepickle.spacing,
                                                 basevolumepickle.size)
        return self

    def toPickle(self, pickle_path):
        """store critical data to unchanging format that can be pickled long term
        """
        basevolumepickle = BaseVolumePickle()
        basevolumepickle.startposition = self.frameofreference.start
        basevolumepickle.spacing = self.frameofreference.spacing
        basevolumepickle.size = self.frameofreference.size
        basevolumepickle.dataarray = self.array

        with open(pickle_path, 'wb') as p:
            pickle.dump(basevolumepickle, p)

    # PUBLIC METHODS
    def conformTo(self, frameofreference, verbose=False):
        """Resamples the current BaseVolume to the supplied FrameOfReference

        Args:
            frameofreference   -- FrameOfReference object to resample the Basevolume to

        Returns:
            BaseVolume
        """
        # conform volume to alternate FrameOfReference
        if (frameofreference is None):
            logger.info('no FrameOfReference provided')
            raise ValueError
        elif (FrameOfReference.__name__ not in str(type(frameofreference))):  # This is an ugly way of type-checking but cant get isinstance to see both as the same
            print('supplied frameofreference of type: "{:s}" must be of the type: "FrameOfReference"'.format(
                str(type(frameofreference))))
            raise TypeError

        # crop to active volume of requested FrameOfReference in frameofreference
        xstart_idx, ystart_idx, zstart_idx = self.frameofreference.getIndices(frameofreference.start)
        xend_idx, yend_idx, zend_idx = self.frameofreference.getIndices(frameofreference.end())
        cropped = self.array[zstart_idx:zend_idx, ystart_idx:yend_idx, xstart_idx:xend_idx]
        if (verbose):
            """
            print('original FOR= start:{:s}, spacing:{:s}, size:{:s}'.format(
                    str(self.frameofreference.start),
                    str(self.frameofreference.spacing),
                    str(self.frameofreference.size)))
            print('new FOR= start:{:s}, spacing:{:s}, size:{:s}'.format(
                    str(frameofreference.start),
                    str(frameofreference.spacing),
                    str(frameofreference.size)))
            """
            logger.info('uncropped shape (z,y,x): {:s}'.format(str(self.array.shape)))
            logger.info('cropped shape(z,y,x): {:s}'.format(str(cropped.shape)))
            logger.info('frameofreference shape (z,y,x): ({:d}, {:d}, {:d})'.format(frameofreference.size[2],
                                                               frameofreference.size[1],
                                                               frameofreference.size[0]))

        zoomfactors = []
        for i in range(3):
            zoomfactors.insert(i, frameofreference.size[2-i] / cropped.shape[i])
        zoomfactors = tuple(zoomfactors)
        if (verbose):
            logger.info('zoom factors (z, y, x): ({:0.3f}, {:0.3f}, {:0.3f})'.format(*zoomfactors))
        resampled_array = interpolation.zoom(cropped, zoomfactors, order=3, mode='constant', cval=0)

        # reconstruct volume from resampled array
        resampled_volume = BaseVolume().fromArray(resampled_array, frameofreference)
        return resampled_volume

    def getSlice(self, idx, axis=0, rescale=False, flatten=False):
        """Extracts 2dArray of idx along the axis.
        Args:
            idx       -- idx identifying the slice along axis

        Optional Args:
            axis      -- specifies axis along which to extract
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            rescale   -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
                            If True, use self.rescaleparams, otherwise provide a RescaleParams object
            flatten   -- return a 1darray in depth-stacked row-major order
        """
        cols, rows, depth = self.frameofreference.size

        # perform index bounding
        if (axis==0):
            if (idx < 0 or idx >= depth):
                logger.info('index out of bounds. must be between 0 -> {:d}'.format(depth))
                raise IndexError
            thisslice = self.array[idx, :, :]
        elif (axis==1):
            if (idx < 0 or idx >= rows):
                logger.info('index out of bounds. must be between 0 -> {:d}'.format(rows))
                raise IndexError
            thisslice = self.array[:, idx, :]
        elif (axis==2):
            if (idx < 0 or idx >= cols):
                logger.info('index out of bounds. must be between 0 -> {:d}'.format(cols))
                raise IndexError
            thisslice = self.array[:, :, idx]
        else:
            logger.info('invalid axis supplied. must be between 0 -> 2')
            raise ValueError

        # RESCALE
        if (rescale is True):
            if (self.rescaleparams is not None):
                thisslice = self.rescaledArray(thisslice, rescale)
            else:
                logger.info('No RescaleParams assigned to self.rescaleparams')
                raise Exception

        # RESHAPE
        if (flatten):
            thisslice = thisslice.flatten(order='C').reshape((-1, 1))

        return thisslice

    def rescaledArray(self, array, rescaleparams):
        factor = float(rescaleparams.factor)
        offset = float(rescaleparams.offset)
        return np.add(np.multiply(array, factor), offset)

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
        if (x < 0 or x >= rows):
            logger.info('x index out of bounds. must be between 0 -> {:d}'.format(cols-1))
            raise IndexError
        if (y < 0 or y >= cols):
            logger.info('y index out of bounds. must be between 0 -> {:d}'.format(rows-1))
            raise IndexError
        if (z < 0 or z >= depth):
            logger.info('z index out of bounds. must be between 0 -> {:d}'.format(depth-1))
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
            logger.info('x index out of bounds. must be between 0 -> {:d}'.format(cols-1))
            raise IndexError
        if (y < 0 or y >= rows):
            logger.info('y index out of bounds. must be between 0 -> {:d}'.format(rows-1))
            raise IndexError
        if (z < 0 or z >= depth):
            logger.info('z index out of bounds. must be between 0 -> {:d}'.format(depth-1))
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

    def conformTo(self, frameofreference, verbose=False):
        """Resamples the current MaskableVolume to the supplied FrameOfReference

        Args:
            frameofreference   -- FrameOfReference object to resample the MaskableVolume to

        Returns:
            MaskableVolume
        """
        base = super().conformTo(frameofreference, verbose)
        maskable = MaskableVolume().fromArray(base.array, base.frameofreference)
        return maskable

    def getSlice(self, idx, axis=0, rescale=False, flatten=False, roi=None, verbose=False):
        """Extracts 2dArray of idx along the axis.
        Args:
            idx     -- idx identifying the slice along axis

        Optional Args:
            axis    --  specifies axis along which to extract
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            rescale -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
                        If True, use self.rescaleparams, otherwise provide a RescaleParams object
            flatten -- return a 1darray in depth-stacked row-major order
            roi     -- ROI object that can be supplied to mask the output of getSlice
         """
        # call to base class
        slicearray = super().getSlice(idx, axis, rescale, flatten)

        # get equivalent slice from densemaskarray
        if (roi is not None):
            maskslicearray = roi.makeDenseMask(self.frameofreference, verbose=verbose).getSlice(idx, axis, rescale, flatten)
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


class BaseVolumePickle:
    """Defines common object that can store feature data for long term I/O
    """
    def __init__(self):
        self.dataarray = None       # numpy ndarray
        self.startposition = None   # (x, y, z)<float>
        self.spacing = None         # (x, y, z)<float>
        self.size = None            # (x, y, z)<integer>
