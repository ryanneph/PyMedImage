"""implementation of standard kmeans clustering"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils.logging import print_indent, g_indents

def create_feature_vector(features, roi=None):
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
        print_indent('no features supplied. skipping', g_indents[1])
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
        print('Common Shape (z,y,x): ({:d}, {:d}, {:d})'.format(*ref_shape))

        # create list of commonly shaped feature vectors
        conformed_feature_list = []
        for i, feature in enumerate(features):
            # check for invalid feature
            if (feature is None):
                print('empty (None) feature provided at index {:d}, removing and continuing'.format(i))
                features.remove(feature)
                continue

            # conform feature volumes and add to list
            conformed_feature = feature.conformTo(frameofreference)

            if (conformed_feature.array.shape != ref_shape):
                print_indent('shape mismatch. ref={ref:s} != feature[{num:d}]={shape:s}.'
                ' removing and continuing'.format(ref=str(ref_shape),
                                                  num=i,
                                                  shape=str(conformed_feature.array.shape))
                                                  , g_indents[1])
                continue
            else:
                # concatenate, need to make feat.array a 2d vector
                conformed_feature_list.append(conformed_feature.vectorize(roi))

        # combine accepted features into array of shape (nSamples, nFeatures)
        feature_array = np.concatenate(conformed_feature_list, axis=1)

        print_indent('combined {n:d} features into array of shape: {shape:s}'.format(
            n=feature_array.shape[1],
            shape = str(feature_array.shape))
            , g_indents[1])
        return feature_array


def cluster(input, k=10, eps=1e-4):
    """take input feature array of N rows and D columns and perform standard kmeans clustering using \
            sklearn kmeans library

    Args:
        input   --  numpy array of N rows and D columns where N is the number of voxels in the volume and
                    D is the number of features.
        k       --  number of clusters
        eps     --  epsilon convergence criteria
    Returns:
        imvector of cluster assignments from 0 to k-1 aligned to the BaseVolumes of input
    """
    # check inputs
    if not isinstance(input, np.ndarray):
        print_indent('a proper numpy ndarray was not provided. skipping.', g_indents[1])
        print_indent(str(type(input)) + str(type(np.ndarray)), g_indents[1])
        return None
    if (k<=1):
        k=12
        print_indent('k must be >1, reassigned to {k:d}'.format(k=k), g_indents[1])

    # Preprocessing - normalization
    normalizer = StandardScaler()
    normalized_input = normalizer.fit_transform(input)

    # create estimator obj
    km = KMeans(n_clusters=k,
                max_iter=300,
                n_init=10,
                init='k-means++',
                precompute_distances=True,
                tol=eps,
                n_jobs=-3
                )
    km.fit(normalized_input)
    print_indent('#iters: {:d}'.format(km.n_iter_), g_indents[1])
    print_indent('score: {score:0.4f}'.format(score=km.score(normalized_input)), g_indents[1])
    return km.predict(normalized_input)

