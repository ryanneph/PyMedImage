"""cluster.py

implementation of clustering algorithms and helpers for working with rttypes"""
import logging
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
from .misc import indent, g_indents

# initialize module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def create_feature_matrix(features, roi=None):
    """takes a list of feature BaseVolumes and combines them into a numpy ndarray of N rows and D features

    where N is the number of samples in each feature vector (voxels in the image) and D is the number of
    feature vectors stored in the "features" list.

    Args:
        features    --  python list of BaseVolumes that are aligned
    Returns:
        NDArray     --  numpy ndarray with N rows and D columns where N is the number of voxels in the
                        aligned images (in depth-row major order) and D is the number of feature vectors
                        in the list (len(features))
    """
    if len(features) <= 0:
        logger.info(indent('no features supplied. skipping', g_indents[1]))
        return None
    else:
        if (roi is not None):
            # use roi.frameofreference as common shape
            frameofreference = roi.frameofreference
        else:
            # find highest resolution volume and use as FrameOfReference
            highest_res_volume = features[0]
            highest_res = np.product(highest_res_volume)
            for volume in features[1:]:
                res = np.product(volume.frameofreference.size)
                if (res > highest_res):
                    highest_res_volume = volume
                    highest_res = res
            # assign highest res FOR as common shape
            frameofreference = highest_res_volume.frameofreference

        # take the first feature vectors shape to be the reference
        ref_shape = frameofreference.size[::-1]  # reverses tuple from (x,y,z) to (z,y,x)
        logger.info('Common Shape (z,y,x): ({:d}, {:d}, {:d})'.format(*ref_shape))

        # create list of commonly shaped feature vectors
        conformed_feature_list = []
        for i, feature in enumerate(features):
            # check for invalid feature
            if (feature is None):
                logger.info('empty (None) feature provided at index {:d}, removing and continuing'.format(i))
                features.remove(feature)
                continue

            # conform feature volumes and add to list
            conformed_feature = feature.conformTo(frameofreference)

            if (conformed_feature.array.shape != ref_shape):
                logger.info(indent('shape mismatch. ref={ref:s} != feature[{num:d}]={shape:s}.'
                ' removing and continuing'.format(ref=str(ref_shape),
                                                  num=i,
                                                  shape=str(conformed_feature.array.shape))
                                                  , g_indents[1]))
                continue
            else:
                # concatenate, need to make feat.array a 2d vector
                conformed_feature_list.append(conformed_feature.vectorize(roi))

        # combine accepted features into array of shape (nSamples, nFeatures)
        feature_array = np.concatenate(conformed_feature_list, axis=1)

        logger.info(indent('combined {n:d} features into array of shape: {shape:s}'.format(
            n=feature_array.shape[1],
            shape=str(feature_array.shape))
            , g_indents[1]))
        return (feature_array, frameofreference)


def cluster_kmeans(feature_matrix, nclusters=10, eps=1e-4):
    """take input feature array of N rows and D columns and perform standard kmeans clustering using \
            sklearn kmeans library

    Args:
        feature_matrix -- numpy array of N rows and D columns where N is the number of voxels in the
                            volume and D is the number of features.

    Optional Args:
        nclusters      -- number of clusters
        eps            -- epsilon convergence criteria
    Returns:
        imvector of cluster assignments from 0 to k-1 aligned to the BaseVolumes of feature_matrix
    """
    # check inputs
    if not isinstance(feature_matrix, np.ndarray):
        logger.info(indent('a proper numpy ndarray was not provided. skipping.', g_indents[1]))
        logger.info(indent(str(type(feature_matrix)) + str(type(np.ndarray)), g_indents[1]))
        return None
    if (nclusters<=1):
        logger.exception(indent('k must be >1', g_indents[1]))
        raise ValueError

    # Preprocessing - normalization
    normalizer = StandardScaler()
    normalized_feature_matrix = normalizer.fit_transform(feature_matrix)

    # create estimator obj
    km = KMeans(n_clusters=nclusters,
                max_iter=300,
                n_init=10,
                init='k-means++',
                precompute_distances=True,
                tol=eps,
                n_jobs=-3
                )
    km.fit(normalized_feature_matrix)
    logger.info(indent('#iters: {:d}'.format(km.n_iter_), g_indents[1]))
    logger.info(indent('score: {score:0.4f}'.format(score=km.score(normalized_feature_matrix)), g_indents[1]))
    return km.predict(normalized_feature_matrix)

def cluster_hierarchical(feature_matrix, nclusters=3, affinity='euclidean', linkage='ward'):
    """take input feature array of N rows and D columns and perform agglomerative hierarchical clustering \
            using the standard sklearn agglomerative clustring library

    Args:
        feature_matrix -- numpy array of N rows and D columns where N is the number of voxels in the
                            volume and D is the number of features.

    Optional Args:
        nclusters      -- number of clusters to find
        affinity       -- metric used to compute linkage ['euclidean', 'l1', 'l2', 'manhattan']
        linkage        -- criterion to use for cluster merging ['ward', 'complete', 'average']
    """
    # check inputs
    if not isinstance(feature_matrix, np.ndarray):
        logger.exception(indent('a proper numpy ndarray was not provided. {:s} != {:s}'.format(
            str(type(feature_matrix)),
            str(type(np.ndarray))
        ), g_indents[1]))
        raise TypeError

    # sanitize string inputs
    linkage = linkage.lower()
    affinity = affinity.lower()

    # Preprocessing - normalization
    normalizer = StandardScaler()
    normalized_feature_matrix = normalizer.fit_transform(feature_matrix)

    # determine valid parameters
    valid_linkage = ['ward', 'complete', 'maximum', 'average']
    if (linkage not in valid_linkage):
        logger.exception('linkage must be one of {:s}'.format(str(valid_linkage)))
        raise ValueError(str)
    if (linkage is 'maximum'):
        linkage = 'complete'

    valid_affinity = ['l1', 'l2', 'manhattan', 'cosine', 'euclidean']
    if (affinity not in valid_affinity):
        logger.exception('affinity must be one of {:s}'.format(str(valid_affinity)))
        raise ValueError(str)

    if (linkage is 'ward'):
        # must use euclidean distance
        affinity = 'euclidean'

    conn_matrix = None

    # create estimator obj
    agg = AgglomerativeClustering(n_clusters=nclusters,
                                  connectivity=conn_matrix,
                                  affinity=affinity,
                                  compute_full_tree=True,
                                  linkage=linkage,
                                  pooling_func=np.mean
                                  )

    # perform fit and estimation
    prediction = agg.fit_predict(normalized_feature_matrix)
    logger.info(indent('#leaves: {:d}'.format(agg.n_leaves_), g_indents[1]))
    logger.info(indent('#components: {:d}'.format(agg.n_components_), g_indents[1]))
    return prediction
