"""scripting.py

A collection of functions/methods that carry us from one step to another in the study
"""

import os
import logging
import pickle
from utils.rttypes import MaskableVolume, ROI
from utils.misc import indent, g_indents, findFiles
from utils import features, dcmio, cluster

# initialize module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

l1_indent = g_indents[1]
l2_indent = g_indents[2]

def loadImages(images_path, modalities):
    """takes a list of modality strings and loads dicoms as a MaskableVolume instance from images_path

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
        logger.info('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # load imvector and store to dictionary for each modality
        # if modality is missing, dont add to dictionary
        if (modalities is None or len(modalities)==0):
            logger.info('No modalities supplied. skipping')
            return None
        else:
            volumes = {}
            for mod in modalities:
                logger.info(indent('Importing {mod:s} images'.format(mod=mod.upper()), l1_indent))
                dicom_path = os.path.join(images_path, '{mod:s}'.format(mod=mod))

                if (os.path.exists(dicom_path)):
                    # recursively walk modality path for dicom images, and build a dataset from it
                    try:
                        volumes[mod] = MaskableVolume().fromDir(dicom_path, recursive=True)
                    except:
                        logger.info('failed to create Volume for modality: {:s}'.format(mod))
                    else:
                        size = volumes[mod].frameofreference.size
                        logger.info(indent('stacked {len:d} datasets of shape: ({d:d}, {r:d}, {c:d})'.format(
                                      len=size[2],
                                      d=1,
                                      r=size[1],
                                      c=size[0]
                                    ), l2_indent))
                else:
                    logger.info(indent('path to {mod:s} dicoms doesn\'t exist. skipping\n'
                                 '(path: {path:s}'.format(mod=mod, path=dicom_path), l2_indent))
                logger.info('')
            return volumes

def loadROIs(rtstruct_path):
    """loads an rtstruct specified by path and returns a dict of ROI objects

    Args:
        rtstruct_path    -- path to rtstruct.dcm file

    Returns:
        dict<key='contour name', val=ROI>
    """
    if (not os.path.exists(rtstruct_path)):
        logger.info(indent('invalid path provided: "{:s}"'.format(rtstruct_path), l2_indent))
        raise ValueError

    logger.info(indent('Importing ROIs', l1_indent))

    # search recursively for a valid rtstruct file
    ds_list = dcmio.read_dicom_dir(rtstruct_path, recursive=True)
    if (ds_list is None or len(ds_list) == 0):
        logger.info('no rtstruct datasets found at "{:s}"'.format(rtstruct_path))
        raise Exception

    # parse rtstruct file and instantiate maskvolume for each contour located
    # add each maskvolume to dict with key set to contour name and number?
    ds = ds_list[0]
    if (ds is not None):
        # get structuresetROI sequence
        StructureSetROI_list = ds.StructureSetROISequence
        nContours = len(StructureSetROI_list)
        if (nContours <= 0):
            logger.debug(indent('no contours were found', l2_indent))
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
                                                     structuresetroi=structuresetroi))
        # prune empty ROIs from dict
        for roiname, roi in dict(roi_dict).items():
            if (roi.coordslices is None or len(roi.coordslices) <= 0):
                logger.debug(indent('pruning empty ROI: {:s} from loaded ROIs'.format(roiname), l2_indent))
                del roi_dict[roiname]

        logger.info(indent('loaded {:d} ROIs succesfully'.format(len(roi_dict)), l2_indent))
        return roi_dict
    else:
        logger.info(indent('no dataset was found', l2_indent))
        return None

def loadEntropy(entropy_pickle_path, image_volumes, radius, roi=None, savepickle=True, recalculate=False):
    """Checks if entropy vector has already been pickled at path specified and
    loads the files if so, or computes entropy for each modality and pickles for later access.

    Args:
        entropy_pickle_path --  should be the full path to the patient specific "precomputed" dir.
            pickle file names are searched for occurence of pet, ct, and entropy and will be loaded if a
            modality string and "entropy" are both present.
        image_volumes       --  dictionary of {modality, BaseVolume} that contains loaded image data for
            each modality supported
    Returns:
        dict<key=mod, value=MaskableVolume>
    """
    # check if path specified exists
    if (not os.path.exists(entropy_pickle_path)):
        logger.info('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # extract modalities from image_volumes
        if (image_volumes is None or len(image_volumes)==0):
            logger.info('No image data was provided. Skipping')
            return None
        modalities = list(image_volumes.keys())

        # load first file that matches the search and move to next modality
        entropy_volumes = {}
        for mod in modalities:
            l1_indent = g_indents[1]
            l2_indent = g_indents[2]
            logger.info(indent('Loading {mod:s} entropy:'.format(mod=mod.upper()), l1_indent))
            # initialize to None
            entropy_volumes[mod] = None

            # get files that match settings
            keywords = ['entropy',
                        'rad{:d}'.format(radius)]
            if (roi is not None):
                keywords.append(roi.roiname)
            keywords.append(mod)
            matches = findFiles(entropy_pickle_path, type='.pickle', keywordlist=keywords)

            if (matches is not None):
                match = matches[0]
            else:
                match = None

            if (not recalculate and match is not None):
                # found pickled entropy vector, load it and add to dict - no need to calculate entropy
                logger.info(indent('Pickled entropy vector found ({mod:s}). Loading.'.format(mod=mod), l2_indent))
                try:
                    path = os.path.join(entropy_pickle_path, match)
                    entropy_volumes[mod] = MaskableVolume().fromPickle(path)
                except:
                    logger.info(indent('there was a problem loading the file: {path:s}'.format(path=path),
                                 l2_indent))
                    entropy_volumes[mod] = None
                else:
                    logger.info(indent('Pickled {mod:s} entropy vector loaded successfully.'.format(
                        mod=mod.upper()), l2_indent))
            else:
                # Calculate entropy this time
                if (match is not None):
                    # force calculation of entropy
                    logger.info(indent('Recalculating entropy as requested', l2_indent))
                else:
                    # if no file is matched for that modality, calculate instead if image dicom files are
                    # present for that modality
                    logger.info(indent('No pickled entropy vector found ({mod:s})'.format(mod=mod), l2_indent))

                logger.info(indent('Computing entropy now...'.format(mod=mod), l2_indent))
                entropy_volumes[mod] = features.image_entropy(image_volumes[mod], roi=roi,
                                                              radius=radius)
                if entropy_volumes[mod] is None:
                    logger.info(indent('Failed to compute entropy for {mod:s} images.'.format(
                        mod=mod.upper()), l2_indent))
                else:
                    logger.info(indent('Entropy computed successfully', l2_indent))
                    # pickle for later recall
                    if (roi is not None):
                        # append ROIName to pickle path
                        pickle_dump_path = os.path.join(entropy_pickle_path,
                            'entropy_{mod:s}_roi_{roiname:s}_rad{rad:d}.pickle'.format(mod=mod,
                                                                                    roiname=roi.roiname,
                                                                                    rad=radius))
                    else:
                        # dont append roiname to pickle path
                        pickle_dump_path = os.path.join(entropy_pickle_path,
                                'entropy_{mod:s}_rad{rad:d}.pickle'.format(mod=mod, rad=radius))
                    try:
                        entropy_volumes[mod].toPickle(pickle_dump_path)
                    except:
                        logger.info(indent('error pickling: {:s}'.format(pickle_dump_path), l2_indent))
                    else:
                        logger.info(indent('entropy pickled successfully to: {:s}'.format(pickle_dump_path),
                                     l2_indent))
            logger.info('')

        # return dict of modality specific entropy imvectors with keys defined by keys for image_volumes arg.
        return entropy_volumes

def loadClusters(clusters_pickle_path, feature_volumes_list, nclusters, radius, roi=None, savepickle=True,
        recalculate=False):
    """Creates feature matrix and calculates clusters then stores the result in new pickle at path specified
    for later recall during hierarchical clustering.

    Args:
        clusters_pickle_path    -- path to search for pickle file and to store new result to
        feature_volumes_list    -- list of BaseVolumes to be used as feature vectors in clustering
        nclusters               -- desired number of clusters computed by kmeans
    Optional Args:
        savepickle              -- should we save the result to pickle?
        recalculate             -- should we recalculate anyway?
    """
    # check if path specified exists
    if (not os.path.exists(clusters_pickle_path)):
        logger.info('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # extract modalities
        if (feature_volumes_list is None or len(feature_volumes_list)==0):
            logger.info('No image data was provided. Skipping')
            return None
        modalities = set([vol.modality.lower()
                          for vol in feature_volumes_list
                          if (vol.modality is not None)])
        # replace pt with pet
        if ('pt' in modalities):
            modalities.remove('pt')
            modalities.add('pet')
        # reorder modalities for predictable naming
        orderedmodalities = []
        order_pref = ['ct', 'pet']
        # add any extra modalities to the end
        for mod in modalities:
            if (mod not in order_pref):
                order_pref.append(mod)
        # get modalities in preferred order
        for mod in order_pref:
            if (mod in modalities):
                orderedmodalities.append(mod)
        # turn modalities into string
        mod_string = '_'.join(orderedmodalities)

        # get files that match settings
        keywords = ['clusters',
                    'rad{:d}'.format(radius),
                    'ncl{:d}'.format(nclusters)]
        if (roi is not None):
            keywords.append(roi.roiname)
        keywords = keywords + list(modalities)
        matches = findFiles(clusters_pickle_path, type='.pickle', keywordlist=keywords)
        if (matches is not None):
            match = matches[0]
        else:
            match = None


        # proceed with loading or recalculating clusters
        if (not recalculate and match is not None):
            # found pickled clusters, load it and add to dict - no need to calculate
            logger.info('Pickled clusters volume found. Loading from path: {:s}'.format(
                               os.path.join(clusters_pickle_path, match)))
            try:
                path = os.path.join(clusters_pickle_path, match)
                clusters = MaskableVolume().fromPickle(path)
            except:
                logger.info('there was a problem loading the file: {path:s}'.format(path=path))
                clusters = None
            else:
                logger.info('Pickled clusters volume loaded successfully.')
        else:
            # Calculate this time
            if (match is not None):
                # force calculation
                logger.info('Recalculating clusters as requested')
            else:
                # if no file is matched, calculate instead
                logger.info('No pickled clusters volume found')

            # get pruned feature matrix
            pruned_feature_matrix, clusters_frameofreference, feature_matrix = cluster.create_feature_matrix(
                                                                                        feature_volumes_list,
                                                                                        roi=roi)
            # calculate:
            clustering_result = cluster.cluster_kmeans(pruned_feature_matrix, nclusters)

            if clustering_result is None:
                logger.info('Failed to compute clusters.')
            else:
                logger.info('Clusters computed successfully')

                # expand sparse cluster assignment vector to dense MaskableVolume
                clusters = cluster.expand_pruned_vector(clustering_result, roi, clusters_frameofreference,
                                                        fill_value=-1)

                # pickle for later recall
                if (roi is not None):
                    # append ROIName to pickle path
                    pickle_dump_path = os.path.join(clusters_pickle_path,
                        'clusters_{mods:s}_roi_{roiname:s}_rad{rad:d}_ncl{ncl:d}.pickle'.format(
                            mods=mod_string,
                            roiname=roi.roiname,
                            rad=radius,
                            ncl=nclusters))
                else:
                    # dont append roiname to pickle path
                    pickle_dump_path = os.path.join(clusters_pickle_path,
                            'clusters_{mods:s}_rad{rad:d}_ncl{ncl:d}.pickle'.format(
                                mods=mod_string, rad=radius, ncl=nclusters))
                try:
                    clusters.toPickle(pickle_dump_path)
                except:
                    logger.info('error pickling: {:s}'.format(pickle_dump_path))
                else:
                    logger.info('clusters pickled successfully to: {:s}'.format(pickle_dump_path))


                # store feature matrix in pickle as numpy ndarray
                featpickle_dump_path = os.path.join(clusters_pickle_path,
                        'features_{mod:s}_roi_{roiname:s}_rad{rad:d}.pickle'.format(
                            mod=mod_string,
                            roiname=roi.roiname,
                            rad=radius))
                try:
                    with open(featpickle_dump_path, mode='wb') as f:
                        pickle.dump(feature_matrix, f)  # dense form
                except:
                    logger.info('error pickling: {:s}'.format(featpickle_dump_path))
                else:
                    logger.info('features successfully pickled to: {:s}'.format(featpickle_dump_path))

                logger.info('')

        # return MaskableVolume containing cluster assignments
        return clusters
