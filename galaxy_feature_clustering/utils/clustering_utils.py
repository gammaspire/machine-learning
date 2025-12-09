from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


##################################
# 2D PCA CONVERSION FOR PLOTTING #
##################################

#convert to 2 PCA components for 2D plotting
#placeholder until (assuming I choose to) UMAP is set up
def pca_2d(feature_data, features, plot=False):
    '''
    Perform PCA dimensionality reduction on the feature data
    Output is the updated pandas dataframe with Comp1, Comp2 columns
    '''
    feature_data = feature_data.copy()   #needed, again, to suppress some pandas Copy warning
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(feature_data[features])   #ignores Feature Cluster column!
    feature_data[['Comp1', 'Comp2']] = X_pca
    
    #PLOT THE COMPONENTS
    if plot:
        from plotting_utils import plot_pca_components
        plot_pca_components(feature_data, features, pca, cmap_name='tab20')
    
    return feature_data


###################################
# 2D UMAP CONVERSION FOR PLOTTING #
###################################

def umap_2d(feature_data, features):
    """
    Reduce the dataset to 2D using UMAP.
    
    Returns
    -------
    feature_data : pandas.DataFrame
        Same dataframe, but with new 'Comp1' and 'Comp2' columns appended.
    """
    from umap import UMAP   #pip install umap-learn
    import numpy as np

    #extract the "matrix"
    X = feature_data[features].values

    #remove NaNs — UMAP cannot embed them
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]

    #UMAP model (default params are fine for visualization, I guess.)
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)

    #compute the 2D embedding (the 'low-dimensional' representation of the multi-dimensional manifold of parameters)
    embedding = reducer.fit_transform(X_clean)

    #allocate the final columns
    #fill rows with NaN for points that were dropped
    feature_data['Comp1'] = np.nan
    feature_data['Comp2'] = np.nan
    feature_data.loc[mask, 'Comp1'] = embedding[:, 0]  #first column of embedding df
    feature_data.loc[mask, 'Comp2'] = embedding[:, 1]  #second column of embedding df

    return feature_data


################################
# K-MEANS CLUSTERING FUNCTIONS #
################################
    
def run1_kmeans(feature_data, features, k=3, print_=False):
    '''
    Perform k-means clustering, output updated df with Feature Cluster column corresponding
        to which cluster each galaxy row belongs.
    '''
    
    feature_data = feature_data.copy()   #some failsafe line to suppress the pandas warning
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    #this silly bracket addition is needed when making a new df col and not editing an existing one..?
    feature_data.loc[:, 'Feature Cluster'] = kmeans.fit_predict(feature_data[features])
    
    return feature_data


def find_optimal_k(feature_data, features, min_k=2, max_k=10, plot=True):
    '''
    Use silhouette scoring to find the 'optimal k' for the given dataset.
    '''
    silhouettes = []
    K = np.arange(min_k, max_k+1)
    
    for k in K:
        faux_df = run1_kmeans(feature_data, features, k)
        silhouettes.append(silhouette_score(faux_df[features], faux_df['Feature Cluster']))
    
    if plot:
        from plotting_utils import plot_silhouette
        plot_silhouette(K, silhouettes)
    
    best_k = K[np.argmax(silhouettes)] #np.argmax returns index of maximum in array or list
    return best_k


################################
# HDBSCAN CLUSTERING FUNCTIONS #
################################

def run1_hdbscan(feature_data, features, min_cluster_size=10, min_samples=None):
    """
    Perform HDBSCAN clustering on the feature data.
    Adds a new 'Feature Cluster' column containing cluster labels.
    Noise points have label -1.
    """
    feature_data = feature_data.copy()

    X = feature_data[features]

    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = hdbscan.fit_predict(X)

    feature_data.loc[:, 'Feature Cluster'] = labels

    return feature_data


def find_optimal_hdbparams(feature_data, features, min_mincluster=3, max_mincluster=30,
                                min_samples=2):
    """
    Find HDBSCAN min_cluster_size hyperparameter by maximizing stability (cluster persistence).
    Performs an elbow-like search on total stability.
    """
    import numpy as np
    
    X = feature_data[features].values

    #create list to store the results
    results = []

    #loop over every hyperparameter pair to sample the space 
    for mcs in range(min_mincluster, max_mincluster + 1, 5):
            
        #run HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples).fit(X)

        #skip degenerate cases (where all data are "noise" --> -1)
        if clusterer.labels_.max() < 0:
            continue

        #sum of cluster stabilities (persistence)
        #high persistence --> stable, robust cluster
        #low persistence --> spurious, flimsy cluster
        total_stability = clusterer.cluster_persistence_.sum()

        results.append((mcs, min_samples, total_stability))  #cluster size, min sample, stability tuple

    #convert the array of stabilities to array
    results = np.array(results, dtype=object)

    #pull ALL of the stabilities
    stabilities = np.array([r[2] for r in results], dtype=float)

    #calculate the differences between consecutive stabilities
    diffs = np.diff(stabilities)
    
    #the index where the diff is smallest -- that is, where there is minimal 
    #improvement between consecutive stabilities
    elbow_idx = np.argmin(diffs)

    best_params = results[elbow_idx][:2]
    best_stability = stabilities[elbow_idx]

    print("############################")
    print(f"Best (elbow) min_cluster_size = {best_params[0]}")
    print(f"Best min_samples = {best_params[1]}")
    print(f"Total cluster stability = {best_stability:.3f}")
    print("############################")

    return best_params