'''
- AIM: create a row-matched catalog with columns corresponding to the
       FC assigned to each galaxy per random_seed run.
- Also create:
    - a column containing the modal FC assignment
    - a column containing the probability of assignment to that FC

Approach:
---------
K-means labels are arbitrary between runs, so clusters are aligned
using centroid positions in PCA space before storing FC labels.
'''

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd


def fcs_across_seeds(feature_data, features, k, n_seeds=50):
    """
    Run K-means clustering across multiple random seeds and track
    feature class (FC) assignments for each object.

    Parameters
    ----------
    feature_data : dataframe
        Sample used for clustering.
    features : list
        List of structural parameters used for k-means clustering.
    k : int
        Number of clusters.
    n_seeds : int, optional
        Number of K-means runs with different random states.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing FC assignments from each run
        (``fc_run{i}``), the modal FC assignment (``fc_mode``),
        and the fraction of runs assigned to the modal FC
        (``probability``).
    """

    #isolate features
    X = feature_data[features]

    #PCA coordinates used for centroid matching
    pca_coords = feature_data[['Comp1', 'Comp2']].values

    #initialize output dataframe
    df = pd.DataFrame(index=feature_data.index)

    # ---------------------------------------------------------
    # Reference clustering
    # ---------------------------------------------------------

    ref_labels = feature_data['Feature Cluster']

    #reference centroids in PCA space
    ref_centroids = np.array([pca_coords[ref_labels == fc].mean(axis=0) for fc in range(k)])

    # store reference labels
    df['fc_run0'] = ref_labels

    # ---------------------------------------------------------
    # Remaining random-seed runs
    # ---------------------------------------------------------

    for i, rs in enumerate(range(1, n_seeds), start=1):

        km = KMeans(n_clusters=k, n_init=1, random_state=rs)

        labels = km.fit_predict(X)

        #centroids in PCA space for this run
        new_centroids = np.array([pca_coords[labels == fc].mean(axis=0) for fc in range(k)])

        #distance matrix between new and reference centroids
        distances = cdist(new_centroids, ref_centroids)

        #map each new cluster to nearest reference cluster
        mapping = np.argmin(distances, axis=1)

        #relabel clusters
        aligned_labels = np.array([mapping[label] for label in labels])

        #store aligned labels
        df[f'fc_run{i}'] = aligned_labels

    # ---------------------------------------------------------
    # Modal FC assignment
    # ---------------------------------------------------------

    fc_modes = df.mode(axis=1)[0]

    # probability of belonging to modal FC
    matches = df.eq(fc_modes, axis=0)

    df['probability'] = matches.mean(axis=1)

    # store modal FC
    df['fc_mode'] = fc_modes

    return df