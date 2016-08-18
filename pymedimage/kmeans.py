"""implementation of standard kmeans clustering"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils.logging import print_indent, g_indents

def create_feature_vector(features, cropextents=None):
    """takes a list of feature imvectors and combines them into a numpy ndarray of N rows and D features

    where N is the number of samples in each feature vector (voxels in the image) and D is the number of
    feature vectors stored in the "features" list.
    
    Args:
        features    --  python list of imvectors that are aligned
    Returns:
        NDArray     --  numpy ndarray with N rows and D columns where N is the number of voxels in the
                        aligned images (in depth-row major order) and D is the number of feature vectors
                        in the list (len(features))
    """
    if len(features) <= 0:
        print_indent('no features supplied. skipping',g_indents[1])
        return None
    else:
        #registration
        #TODO - REGISTRATION

        # take the first feature vectors shape to be the reference
        ref_shape = features[0].vectorize(cropextents=cropextents).shape
        print(ref_shape)

        # initialize array with first feature
        feature_array = features[0].vectorize(cropextents=cropextents)

        if len(features) >= 1:
            for i, feat in enumerate(features[1:]):
                if (feat is None or feat.vectorize(cropextents=cropextents).shape != ref_shape):
                    print_indent('shape mismatch. ref={ref:s} != feature[{num:d}]={shape:s}'.format(
                        ref=str(ref_shape),
                        num=i,
                        shape=str(feat.vectorize(cropextents=cropextents).shape))
                        , g_indents[1])
                    continue
                else:
                    # concatenate, need to make feat.array a 2d vector
                    feature_array = np.concatenate((feature_array, feat.vectorize(cropextents=cropextents)), axis=1)

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
        imvector of cluster assignments from 0 to k-1 aligned to the imvectors of input
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

