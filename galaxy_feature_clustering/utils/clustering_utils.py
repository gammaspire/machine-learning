from sklearn.cluster import KMeans, DBSCAN
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
    #pip install umap-learn
    from umap import UMAP

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


###############################
# DBSCAN CLUSTERING FUNCTIONS #
###############################

def run1_dbscan(feature_data, features, eps=0.5, min_samples=10):
    """
    Perform DBSCAN clustering on the feature data.
    Adds a new 'Feature Cluster' column containing cluster labels.
    Noise points have label -1.
    """
    feature_data = feature_data.copy()

    X = feature_data[features]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    feature_data.loc[:, 'Feature Cluster'] = labels

    return feature_data


def find_optimal_dbparams(feature_data, features, mineps=0.2, maxeps=1.2, min_minsamples=5, max_minsamples=50):
    '''
    Use silhouette scoring to find the optimal DBSCAN eps and min_samples parameters for the given dataset.
    Default min and max eps expect that the input feature data are standardized!
    '''
    import numpy as np
    
    silhouettes = []
    eps_ = np.linspace(mineps, maxeps, 15)
    min_samples_ = np.arange(min_minsamples, max_minsamples, 5)

    #initiate the "best score" and set of "best params"
    best_score = -1
    best_params = None

    X = feature_data[features].values

    for ep in eps_:
        for min_samp in min_samples_:

            db = DBSCAN(eps=ep, min_samples=min_samp).fit(X)
            labels = db.labels_

            #remove noise...remember that DBSCAN marks noise with a -1
            mask = (labels != -1)
            if mask.sum() < 10:
                continue  #too few points to evaluate

            n_clusters = len(set(labels[mask]))
            if n_clusters <= 1:
                continue  #silhouette will be undefined since there is <= 1 cluster!

            score = silhouette_score(X[mask], labels[mask])
            silhouettes.append((ep, min_samp, score, n_clusters))

            #the higher the score, the "better" the parameters!
            if score > best_score:
                best_score = score
                best_params = (ep, min_samp)

    print('#'*20)
    print(f'Best eps = {best_params[0]}, best min_sample = {best_params[1]}')
    print(f'Best score = {best_score}')
    print('#'*20)
    
    return best_params[0], best_params[1]