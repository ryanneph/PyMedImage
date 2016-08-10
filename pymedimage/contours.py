"""Contours.py

library for importing and manipulating rtstruct type dicom contours
"""

from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


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

    def denseMask(self, shape):
        """converts contour_points list to a dense 1Darray binary mask in shape of imvector images

        Args:
            shape   --  numpy shape object
        Returns:
            1Darray binary mask with same shape as imvector
        """
        pass

