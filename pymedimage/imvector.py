"""
imvector.py

class definition for working with flattened image vector
"""

import numpy as np
import dicom #pydicom

class imvector:
    """Encapsulates numpy ndarray and adds attributes

    depth is the original dim0 of the flattened ndarray in self.array
    rows is the original dim1 of the flattened ndarray in self.array
    columns is the original dim2 of the flattened ndarray in self.array
    """
    def __init__(self, dataset=None):
        self.mask = None
        if type(dataset) is dicom.dataset:
            self.INIT_dataset(dataset)
        elif type(dataset) is np.ndarray:
            self.array = dataset.flatten()
            self.depth = dataset.shape[0]
            self.rows = dataset.shape[1]
            self.columns = dataset.shape[2]
            self.sliceidx = []
        else:
            self.array = None
            self.depth = 0
            self.rows = 0
            self.columns = 0
            self.sliceidx = []


    def INIT_dataset(self, dataset):
        self.array = dataset.pixel_array.flatten()
        self.depth = 1
        self.rows = dataset.pixel_array.shape[0]
        self.columns = dataset.pixel_array.shape[1]
        self.sliceidx = []
        if ('InstanceNumber' in dataset.dir()):
            self.sliceidx.append(dataset.InstanceNumber)


    def SizeMatch(self, dataset):
        """check that the rows and columns of the dataset matches the current imvector size"""
        if self.depth <= 0:
            # nothing here yet, cant match so we pass it
            return True
        else:
            #check sizes
            return (self.rows == dataset.pixel_array.shape[0]
                    and self.columns == dataset.pixel_array.shape[1])


    def append(self, dataset):
        """take a pydicom dataset and appends it to this imvector.

        updates size attributes accordingly
        """
        if dataset is not None and self.SizeMatch(dataset):
            if self.array is not None:
                #append flattened pixel data to self.array
                self.array = np.concatenate((self.array, dataset.pixel_array.flatten()), axis=0) 
                self.depth += 1
                if ('InstanceNumber' in dataset.dir()):
                    self.sliceidx.append(dataset.InstanceNumber)
            else:
                #nothing here yet, just iniitialize with this dataset
                self.INIT_dataset(dataset)


    def get_val(self, z, y, x):
        """convenience function for returning image intensity at location

        Uses depth-row major ordering:
        depth: axial slices inf->sup
        rows: coronal slices anterior->posterior
        cols: sagittal slices: pt.right->pt.left
        """
        r = self.rows
        c = self.columns
        d = self.depth
        if (z<0 or y<0 or x<0) or (z>=d or y>=r or x>=c):
            return 0
        else:
            pos = r*c*z + c*y + x
            return self.array[pos]


    def set_val(self, z, y, x, value):
        """convenience function for reassigning image intensity at location

        Uses depth-row major ordering:
        depth: axial slices inf->sup
        rows: coronal slices anterior->posterior
        cols: sagittal slices: pt.right->pt.left
        """
        r = self.rows
        c = self.columns
        d = self.depth
        if not (z<0 or y<0 or x<0) and not (z>=d or y>=r or x>=c):
            pos = r*c*z + c*y + x
            self.array[pos] = value


    def get_slice(self, idx=0, axis=0, flatten=False):
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
            slice = self.array.reshape((self.depth, self.rows, self.columns))[:, idx, :]
        elif axis==2:
            idx = 0 if (idx < 0) else (self.columns-1 if idx >= self.columns else idx)
            slice = self.array.reshape((self.depth, self.rows, self.columns))[:, :, idx]
        else:
            idx = 0 if (idx < 0) else (self.depth-1 if idx >= self.depth else idx)
            slice = self.array.reshape((self.depth, self.rows, self.columns))[idx, :, :]

        if flatten:
            slice = slice.flatten()
        return slice


    def get_wl(self, window, level=0):
        """rescales image intensities to fit within window and centered at level

        Args:
            window  --  tuple (,) of values that define the min and max intensities of the window
                        such that all values that are below min are set to min and all above max are set to max
                    --  integer that defines the width of the window. level must be set to something other than 0.
            level   --  integer that specifies the center of the new data representation if window is an int.
                        if window is tuple, it is unchecked
        """
        # crop and swap window order if necessary
        if not isinstance(window, int) or window<1 or not isinstance(window, tuple):
            # bad input
            print('window must be a positive int or a 2-tuple. exiting')
            return None
        if len(window) == 1:
            # use window as the full width of the window, centered on level
            pass

        if len(window) > 1:
            if len(window) > 2:
                window = window[:2]
        if window[0] > window[1]:
            window = window.reverse()
            


