import numpy as np
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

def run1_hdbscan(feature_data, features, min_cluster_size=10, min_samples=None, 
                 metric='euclidean', cluster_selection_method='leaf'):
    """
    Perform HDBSCAN clustering on the feature data.
    Adds a new 'Feature Cluster' column containing cluster labels.
    Noise points have label -1.
    """
    feature_data = feature_data.copy()

    X = feature_data[features]
    
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
                     cluster_selection_method=cluster_selection_method)
    labels = hdbscan.fit_predict(X)

    feature_data.loc[:, 'Feature Cluster'] = labels

    return feature_data


def find_optimal_hdbparams(feature_data, 
                           features,
                           min_cluster_sizes=[10,15,20,25,30],
                           min_samples_list=[2,5,10,15,20,'same'],
                           metrics=['euclidean', 'manhattan', 'canberra', 'braycurtis'],
                           cluster_methods=['eom','leaf']):
    """
    Full-grid HDBSCAN hyperparameter optimization using cluster stability
    (persistence). Returns the hyperparameter tuple at the elbow where 
    marginal improvement in stability is minimal.
    """
    
    X = feature_data[features].values
    results = []
    
    #a rather unwieldy approach to sampling the entire space. cope. :-)
    for mcs in min_cluster_sizes:
        
        for ms in min_samples_list:
            
            if ms == 'same':
                ms_effective = mcs
            else:
                ms_effective = ms
            
            for metric in metrics:
                for csm in cluster_methods:
                    
                    clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms_effective, metric=metric, 
                                        cluster_selection_method=csm).fit(X)
                    
                    #skip degenerate solutions (noise = -1)
                    if clusterer.labels_.max() < 0:
                        continue
                    
                    #sum of cluster stabilities (persistence)
                    #high persistence --> stable, robust cluster
                    #low persistence --> spurious, flimsy cluster
                    total_stability = clusterer.cluster_persistence_.sum()
                    
                    results.append((mcs, ms_effective, metric, csm, total_stability))
    
    #convert to np array for easy slicing
    results = np.array(results, dtype=object)
    stabilities = np.array([r[4] for r in results], dtype=float)

    #calculate the differences between consecutive stabilities
    diffs = np.diff(stabilities)
    
    #the index where the diff is smallest -- that is, where there is minimal 
    #improvement between consecutive stabilities
    elbow_idx = np.argmin(diffs)

    best = results[elbow_idx]
    
    print("############################")
    print(" Best parameters at elbow:")
    print(f"   min_cluster_size         = {best[0]}")
    print(f"   min_samples              = {best[1]}")
    print(f"   metric                   = {best[2]}")
    print(f"   cluster_selection_method = {best[3]}")
    print(f"   total stability          = {best[4]:.3f}")
    print("############################")
    
    return best[0], best[1], best[2], best[3]