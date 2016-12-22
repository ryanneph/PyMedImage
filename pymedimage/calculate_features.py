"""Standardized methods for mutltiprocess capable feature calculation and storage
"""
import os
import time
import logging
from multiprocessing import Pool
from .rttypes import MaskableVolume
from .notifications import pushNotification

# initialize module logger
logger = logging.getLogger(__name__)

def checkCalculated(doi, local_feature_def):
    # check if already calculated
    p_doi_features = doi.getFeaturesPath()
    if os.path.exists(p_doi_features):
        matches = local_feature_def.findFiles(p_doi_features)
        if matches: return True
    return False

def loadPrecalculated(doi, local_feature_def):
    # check if already calculated
    p_doi_features = doi.getFeaturesPath()
    if os.path.exists(p_doi_features):
        matches = local_feature_def.findFiles(p_doi_features)
        if matches and len(matches)>0:
            # print(', '.join(matches))
            return None ##QUICKFIX - corrupt pickle file errors on nextline with HYPOFRAC dataset
            return MaskableVolume().fromPickle(matches[0])
    return None

def pickleFeature(doi, local_feature_def, result_array):
    p_doi_features = doi.getFeaturesPath()
    p_feat_pickle = os.path.join(p_doi_features, local_feature_def.generateFilename())
    os.makedirs(p_doi_features, exist_ok=True)
    result_array.toPickle(p_feat_pickle)
    logger.debug('Feature: "{:s}" was stored to: {!s}'.format(local_feature_def.label, p_feat_pickle))

def calculateFeature(doi, local_feature_def):
    """single doi, single feature calculation sub-unit that can be multithreaded and called by a pool
    of workers

    Args:
        doi (str): string identifier unique to each patient/doi
        local_feature_def (LocalFeatureDefinition): information for feature calculation

        pickle (bool): if true, store result to pickle file, if false simply return 2-tuple of (result_code, result_array)
    Returns:
        int: status code
    """
    recalculated = False
    if checkCalculated(doi, local_feature_def):
        if (local_feature_def.recalculate):
            recalculated = True
        else:
            logger.debug('Feature already calculated. skipping')
            loaded_feature_vol = loadPrecalculated(doi, local_feature_def)
            return (10, loaded_feature_vol)

    # load dicom data
    ct_vol = doi.getImageVolume()
    roi = doi.getROI()
    if (not ct_vol or not roi):
        logger.debug('missing ct or roi. skipping.')
        return (1, None)

    # compute feature
    logger.debug('calculating "{!s}" for doi: {!s}'.format(local_feature_def.label, doi))
    feature_vol = local_feature_def.calculation_function(ct_vol, roi, **local_feature_def.args)

    # return status
    if (recalculated):
        return (11, feature_vol)
    else:
        return (0, feature_vol)

def calculateCompositeFeature(doi, composite_feature_def, pickleintermediate=False):
    recalculated = False
    if checkCalculated(doi, composite_feature_def):
        if (composite_feature_def.recalculate):
            recalculated = True
        else:
            return (10, None)

    vol_list = []
    for lfeatdef in composite_feature_def.featdefs:
        # lfeatdef.recalculate = True  # force recalculation
        result_code, feature_vol = calculateFeature(doi, lfeatdef)
        if result_code not in [0, 11, 10]:
            return 2, None
        if feature_vol:
            vol_list.append(feature_vol)
            if pickleintermediate: pickleFeature(doi, lfeatdef, feature_vol)

    if len(vol_list) <= 0:
        return 3, None
    composite_vol = composite_feature_def.composition_function(vol_list)

    # return status
    if (recalculated):
        return (11, composite_vol)
    else:
        return (0, composite_vol)

def worker_calculateFeature(args_tuple):
    (doi, local_feature_def) = args_tuple
    time_start = time.time()
    try:
        cls = local_feature_def.__class__.__name__
        if ('LocalFeatureDefinition' in cls):
            result_code, feature_vol = calculateFeature(doi, local_feature_def)
            if feature_vol:
                pickleFeature(doi, local_feature_def, feature_vol)
        elif ('LocalFeatureCompositionDefinition' in cls):
            result_code, composite_result = calculateCompositeFeature(doi, local_feature_def, pickleintermediate=False)
            if composite_result:
                pickleFeature(doi, local_feature_def, composite_result)

        if (result_code == 0):
            result_string = 'success'
        elif (result_code == 1):
            result_string = 'missing data'
        elif (result_code == 3):
            result_string = 'composite error'
        elif (result_code == 10):
            result_string = 'skipped'
        elif (result_code == 11):
            result_string = 'recalc'
        else:
            # unknown result
            result_code = -1
            result_string = 'unknown'

    except Exception as e:
        result_code = 2
        result_string = 'exception'
        logger.error('{!s}'.format(e))
        raise

    time_end = time.time()
    job_time_string = time.strftime('%H:%M:%S', time.gmtime(time_end-time_start))

    return (result_code, result_string, job_time_string, doi, local_feature_def)


def multiprocessCalculateFeatures(doi_list, feature_def_list, processes=16, logskipped=True):
    """multithreaded manager for calculating local image features for a large number of dois"""

    # build argmap for worker pool
    argmap = []
    for doi in doi_list:
        for local_feature_def in feature_def_list:
            argmap.append((doi, local_feature_def))

    try:
        # start multiprocessing and print output summary for each job
        time_start = time.time()
        total_jobs = len(argmap)
        jnum = 0
        error_count = 0
        # limit number of concurrent processes as there is only so much GPU memory available at one time<Plug>(neosnippet_expand)
        # with 8 proc: max mem usage of ~4-4.5GB of 12.204GB total global mem
        with Pool(processes=processes) as p:
            logger.info('Feature Calculation:')
            logger.info('-----------------------------------------------------------------------------------------')
            logger.info('BEGINNING PROCESSING (at {!s})'.format(time.strftime('%Y-%b-%d %H:%M:%S')))
            logger.info('')
            logger.info('RESULTS:  (total #jobs: {:d})'.format(total_jobs))
            logger.info('-----------------------------------------------------------------------------------------')

            for worker_results in p.imap(worker_calculateFeature, argmap, chunksize=1):
                jnum += 1
                (result_code, result_string, job_time_string, doi, local_feature_def) = worker_results
                if (result_code!=10 or (result_code==10 and logskipped)):
                    log_string = 'job#{jnum:_>5d} [{string:12s}:{code:2d}]: {doi!s:9s}  {label!s:25s}  {args!s:45s}  {time!s}'.format(
                        jnum    = jnum,
                        string  = result_string,
                        code    = result_code,
                        doi     = doi,
                        label   = local_feature_def.label,
                        args    = local_feature_def.getArgsString(),
                        time    = job_time_string
                    )
                    logger.info(log_string)

                if (result_code == -1 or (result_code > 0 and result_code < 10)):
                    error_count += 1
                    logger.error('{:05d}.  {!s}'.format(error_count, log_string))

        time_finish = time.time()
        logger.info('-----------------------------------------------------------------------------------------')
        logger.info('(success: {:d} | error: {:d}) of {:d} jobs'.format(total_jobs-error_count, error_count,
                                                                        total_jobs))
        logger.info('total time: {:s}'.format(time.strftime('%H:%M:%S', time.gmtime(time_finish-time_start))))
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info('')

    except Exception as e:
        pushNotification('FAILURE - Multiprocess_Features', '{!s}'.format(repr(e)))
        raise e

    pushNotification('SUCCESS - Multiprocess_Features', 'Finished processing {:d} jobs \
                                 with {:d} errors'.format(total_jobs, error_count))
