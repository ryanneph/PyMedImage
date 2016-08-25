"""scripting.py

A collection of functions/methods that carry us from one step to another in the study
"""

import os
import dicom
from .rttypes import BaseVolume, MaskableVolume, ROI
from .logging import print_indent, g_indents
from . import features

def loadImages(images_path, modalities):
    """takes a list of modality strings and loads dicoms into an imvolume dataset from images_path

    Args:
        images_path --  Full path to patient specific directory containing various modality dicom images
            each modality imageset is contained in a directory within images_path where the modality string
            in modalities must match the directory name. This subdir is recursively searched for all dicoms
        modalities  --  list of modality strings that are used to identify subdirectories from which dicoms
            are loaded
    Returns:
        dictionary of {modality: imvolume} that contains loaded image data for each modality supported
    """
    # check if path specified exists
    if (not os.path.exists(images_path)):
        print('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # load imvector and store to dictionary for each modality
        if (modalities is None or len(modalities)==0):
            print('No modalities supplied. skipping')
            return None
        else:
            volumes = {}
            l1_indent = g_indents[1]
            l2_indent = g_indents[2]
            for mod in modalities:
                print_indent('Importing {mod:s} images'.format(mod=mod.upper()), l1_indent)
                dicom_path = os.path.join(images_path, '{mod:s}'.format(mod=mod))

                if (os.path.exists(dicom_path)):
                    # recursively walk modality path for dicom images, and build a dataset from it
                    volumes[mod] = MaskableVolume().fromDir(dicom_path, recursive=True)
                    volume = volumes[mod]
                    if (volume is not None):
                        size = volume.frameofreference.size
                        print_indent('stacked {len:d} datasets of shape: ({d:d}, {r:d}, {c:d})'.format(
                                      len=size[2],
                                      d=1,
                                      r=size[1],
                                      c=size[0]
                                    ), l2_indent)
                else:
                    print_indent('path to {mod:s} dicoms doesn\'t exist. skipping\n'
                                 '(path: {path:s}'.format(mod=mod, path=dicom_path), l2_indent)
                print()
            return volumes

def loadROIs(rtstruct_path, verbose=False):
    """loads an rtstruct specified by path and returns a dict of ROI objects

    Args:
        rtstruct_path    -- path to rtstruct.dcm file

    Returns:
        dict<key='contour name', val=ROI>
    """
    if (rtstruct_path is not None and os.path.exists(rtstruct_path)):
        # parse rtstruct file and instantiate maskvolume for each contour located
        # add each maskvolume to dict with key set to contour name and number?
        ds = dicom.read_file(rtstruct_path)
        if (ds is not None):
            # get structuresetROI sequence
            StructureSetROI_list = ds.StructureSetROISequence
            nContours = len(StructureSetROI_list)
            if (nContours <= 0):
                if (verbose):
                    print('no contours were found')
                return None

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
                roi_dict[structuresetroi.ROIName] = (ROI(frameofreference=None,
                                                         roicontour=ROIContour_dict[ROINumber],
                                                         structuresetroi=structuresetroi,
                                                         verbose=verbose))
            # prune empty ROIs from dict
            for roiname, roi in dict(roi_dict).items():
                if (roi.coordslices is None or len(roi.coordslices) <= 0):
                    if (verbose):
                        print('pruning empty ROI: {:s} from loaded ROIs'.format(roiname))
                    del roi_dict[roiname]

            return roi_dict
        else:
            print('no dataset was found')
            return None

    else:
        print('path invalid: "{:s}"'.format(str(rtstruct_path)))
        return None

def loadEntropy(entropy_pickle_path, image_volumes, roi=None, radius=4,
                savePickle=True, verbose=False):
    """Checks if entropy vector has already been pickled at path specified and
    loads the files if so, or computes entropy for each modality and pickles for later access.
    Returns tuple of entropy imvectors (CT_entropy, PET_entropy)

    Args:
        entropy_pickle_path --  should be the full path to the patient specific "precomputed" dir.
            pickle file names are searched for occurence of pet, ct, and entropy and will be loaded if a
            modality string and "entropy" are both present.
        image_volumes       --  dictionary of {modality, BaseVolume} that contains loaded image data for
            each modality supported
    """
    # check if path specified exists
    if (not os.path.exists(entropy_pickle_path)):
        print('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # extract modalities from image_volumes
        if (image_volumes is None or len(image_volumes)==0):
            print('No image data was provided. Skipping')
            return None
        modalities = image_volumes.keys()

        # get list of files in immediate path (not recursive)
        files = [
            f
            for f in os.listdir(entropy_pickle_path)
            if os.path.isfile(os.path.join(entropy_pickle_path, f))
            and ('entropy' in f.lower())
            and ('.pickle' == os.path.splitext(f)[1])
        ]

        # load first file that matches the search and move to next modality
        entropy_volumes = {}
        for mod in modalities:
            l1_indent = g_indents[1]
            l2_indent = g_indents[2]
            print_indent('Loading {mod:s} entropy:'.format(mod=mod.upper()), l1_indent)
            # initialize to None
            entropy_volumes[mod] = None
            # find first pickle that matches modality string or compute entropy fresh for that modality
            if (roi is not None):
                # match with modality and ROIName
                match = next((f for f in files
                              if (mod in f.lower()
                                  and roi.roiname.lower() in f.lower())), None)  # gets first match and stops
            else:
                # match with modality
                match = next((f for f in files if (mod in f.lower()) ), None)  # gets first match and stops
            if (match is not None):
                # found pickled entropy vector, load it and add to dict
                print_indent('Pickled entropy vector found ({mod:s}). Loading.'.format(mod=mod), l2_indent)
                try:
                    path = os.path.join(entropy_pickle_path, match)
                    entropy_volumes[mod] = BaseVolume().fromPickle(path)
                except:
                    print_indent('there was a problem loading the file: {path:s}'.format(path=path),
                                 l2_indent)
                    entropy_volumes[mod] = None
                else:
                    print_indent('Pickled {mod:s} entropy vector loaded successfully.'.format(
                        mod=mod.upper()), l2_indent)
            else:
                # if no file is matched for that modality, calculate instead if image dicom files are
                #   present for that modality
                # no match, compute entropy
                print_indent('No pickled entropy vector found ({mod:s})'.format(mod=mod), l2_indent)
                # check for presence of image vector in modality
                image_volume = image_volumes[mod]
                if image_volume is not None:
                    print_indent('Computing entropy now...'.format(mod=mod), l2_indent)
                    entropy_volumes[mod] = features.image_entropy(image_volume, roi=roi,
                                                                  radius=radius, verbose=verbose)
                    if entropy_volumes[mod] is None:
                        print_indent('Failed to compute entropy for {mod:s} images.'.format(
                            mod=mod.upper()), l2_indent)
                    else:
                        print_indent('Entropy computed successfully', l2_indent)
                        # pickle for later recall
                        if (roi is not None):
                            # append ROIName to pickle path
                            pickle_dump_path = os.path.join(entropy_pickle_path,
                              '{mod:s}_mask_{roiname:s}_entropy.pickle'.format(mod=mod, roiname=roi.roiname))
                        else:
                            # dont append roiname to pickle path
                            pickle_dump_path = os.path.join(entropy_pickle_path,
                                                            '{mod:s}_entropy.pickle'.format(mod=mod))
                        try:
                            entropy_volumes[mod].toPickle(pickle_dump_path)
                        except:
                            print_indent('error pickling: {:s}'.format(pickle_dump_path), l2_indent)
                        else:
                            print_indent('entropy pickled successfully to:\n{:s}'.format(pickle_dump_path),
                                         l2_indent)
                else:
                    print_indent('No {mod:s} image vector was supplied.'
                                 ' Could not compute entropy.'.format(mod=mod.upper()), l2_indent)
            print()

        # return dict of modality specific entropy imvectors with keys defined by keys for image_volumes arg.
        return entropy_volumes
