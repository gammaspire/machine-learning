from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

def ari_across_seeds(X, k, n_runs=20, random_states=None):
    """
    Compute pairwise ARI for k-means clusterings with different random seeds.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix used for clustering.
    k : int
        Number of clusters.
    n_runs : int
        Number of k-means runs.
    random_states : list or None
        Optional list of random seeds.
    
    Returns
    -------
    ari_values : list
        Pairwise ARI values between all clustering runs.
    """
    
    #if user does not pre-define a list of random_states, create a list from 0 to n_runs
    if random_states is None:
        random_states = range(n_runs)
    
    #initialize labels list
    labels = []
    
    #for every random_state in the random_states list, run k-means clustering with n_clusters=k
        #why n_init=1?
        #THIS PART IS CRUCIAL!
        #for my actual k-means clusters, I choose n_init=10 (meaning the clustering is run 10 times and the algorithm chooses the "best" solution
        #for ARI, I want to measure the "intrinsic stability" of the clusters. that is, I want to isolate the effects of random_state on the stability on feature groups. 
        #THAT IS, I want to see how much the k-means result varies whenever it runs with a new random_state. If I set n_init=10, that can wash out the variability because the algorithm is trying to find the best solution every time.
            #THAT IS (x2), if I run k-means once with the same k, how does a variable random_state affect the output?
            #if very little effect, ARI~1. if no better than random labellings, ARI~0. 
        
        
    for rs in random_states:
        km = KMeans(n_clusters=k, n_init=1, random_state=rs)
        labels.append(km.fit_predict(X))
    
    ari_values = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            ari_values.append(adjusted_rand_score(labels[i], labels[j]))
    
    return np.array(ari_values)