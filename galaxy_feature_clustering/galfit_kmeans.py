'''
OBJECTIVE: 
* apply k-means feature clustering (unsupervised machine learning) to VFS GALFIT output parameters for optical grz, unWISE W1-4.

NEED:
* vf_v2_galfit_{band}.fits
    * band = g, r, z, W1-fixBA, W2, W3-fixBA W4
    * should ALL be row-matched!

PROCEDURE:
* open FITS files using astropy.table.Table, convert to pandas dataframe
* apply these quality check flags to every set of data:
    * ('CXC' > 0, 'CYC' > 0) --> if nonzero, then the row will have no data
    * (nser_{band} <= 6.) --> a Sersic index > 6 is unphysical
    * (~'CNumerical_Error') --> if True, then GALFIT hiccupped for this galaxy and 
        generated an unreliable model
* convert all effective radii to arcseconds!
* standardize all data using sklearn.StandardScaler
* determine the optimal k value using silhouette score (optionally generate elbow-method 
    and silhouette plots)
    * quantitative metric of feature cluster quality
    * higher score --> more robust clustering
* use optimal k-value to generate "cluster membership array"
    * row-matched to input data; contains integer value for every galaxy indicating 
        the cluster to which they belong
* if desired, plot in either 2D PCA space or in the X & Y chosen in galfit_parameters.py
'''

import sys
sys.path.insert(0,'scripts')
from dataprocessing_utils import px_to_arcsec, arcsec_to_kpc

from galfit_parameters import Params
params = Params()

import numpy as np
from scipy.stats import zscore
from astropy.table import Table
import pandas as pd
from rich import print

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_feature_names():
    #define Re, Sersic index feature columns
    re_cols = [f'CRE_{band}' for band in params.BANDS]
    nser_cols = [f'CN_{band}' for band in params.BANDS]
    
    #combine
    features = re_cols + nser_cols
    
    #and return
    return features


def get_vcosmic_column():
    env = Table.read('data/vf_v2_environment.fits')
    return env['Vcosmic']

def get_photometric_colors():
    phot = Table.read('data/vf_v2_legact_ephot.fits', include_names=['','','','']


def make_galfit_table():
    '''
    Read GALFIT grz, W1-4 output tables
    Convert from astropy Table to pandas df
    Output --> dataframe with grz, W1-4 Re, nser, CXC, CNumerical_Error columns
    '''
    #create empty astropy table
    data_table = Table()
    
    for band in params.BANDS:
        t = Table.read(f'data/vf_v2_galfit_{band}.fits')
        for colname in params.COLUMNS:
            data_table[f'{colname}_{band}'] = t[colname]
    
    #append the Vcosmic column
    data_table['Vcosmic'] = get_vcosmic_column()
    
    return data_table.to_pandas()


def trim_galfit_table(full_df):
    '''
    Apply trimming flags -- non-data, Re, nser, numerical error
    '''
    
    #unfortunately have to trim any row which has at least one of the below problems.
    
    ngal_before = len(full_df)
    
    #center x-position, Nser, numerical error columns
    xyc_cols = [f'CXC_{band}' for band in params.BANDS]
    nser_cols = [f'CN_{band}' for band in params.BANDS]
    numerr_cols = [f'CNumerical_Error_{band}' for band in params.BANDS]
    
    #drop row if any central x pixel coordinate cell is zero
    full_df = full_df.loc[~(full_df[xyc_cols]==0).any(axis=1)]
    
    #drop rows with any nser > 6.
    full_df = full_df.loc[~(full_df[nser_cols]>6).any(axis=1)]
    
    #drop rows with any convolved numerical error
    full_df = full_df.loc[~(full_df[numerr_cols]).any(axis=1)]
    
    #print number of removed galaxies
    message=f'Removed {ngal_before - len(full_df)}/{ngal_before} galaxies with GALFIT quality flags.'
    print('#'*len(message))
    print(message)
    print('#'*len(message))
    
    #return the 'cleansed' dataframe
    return full_df


def iqr_clipping(df, features, k_clip=1.5):
    '''
    AIM: perform interquartile range clipping. 
    k_clip=1.5 has a Gaussian equivalent of roughly 2.7-sigma
    k_clip=2.5 has a Gaussian equivalent of roughly 3-sigma
    '''
    df = df.copy()
    
    Q1 = df[features].quantile(0.25)   #find 25% quartile of data distributions
    Q3 = df[features].quantile(0.75)   #find 75% quartile of data distributions
    
    #find the range of data values between these two bounds
    IQR = Q3 - Q1

    #generate mask --> data must not be beyond some multiple of the IQR width
    #the Q1-... and Q3+... just expand the endpoints of IQR outward such that the "new" IQR
    #is k-times the original IQR size.
    outlier_mask = ((df[features] < (Q1 - k_clip * IQR)) | (df[features] > (Q3 + k_clip * IQR))).any(axis=1)
    
    print(f"IQR clipping (k={k_clip}): Removing an additional {outlier_mask.sum()}/{len(outlier_mask)} outliers ({outlier_mask.mean():.1%})")
    print(f"Remaining galaxies: {len(df) - outlier_mask.sum()}")
    
    return df[~outlier_mask]


def standardize_data(df, features):
    '''
    Standardize data features such that each column has a mean of 0 and a standard deviation of 1.
    Uses get_feature_names() by default -- has CN and CRE for grz, W1-4
    output: edited pandas dataframe with input columns standardized
    '''
    
    #initiate the scaler    
    scaler = StandardScaler()
    
    #apply the scaler to transform the features
    df[features] = scaler.fit_transform(df[features])
    
    #ANNNND return (the full df)
    return df


#convert to 2 PCA components for 2D plotting
#placeholder until (assuming I choose to) UMAP is set up
def pca_2d(feature_data, features, plot=False):
    '''
    Perform PCA dimensionality reduction on the feature data
    Output is the updated pandas dataframe with PCA1, PCA2 columns
    '''
    feature_data = feature_data.copy()   #needed, again, to suppress some pandas Copy warning
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(feature_data[features])   #ignores Feature Cluster column!
    feature_data[['PCA1', 'PCA2']] = X_pca
    
    #PLOT THE COMPONENTS
    if plot:
        from plotting_utils import plot_pca_components
        plot_pca_components(feature_data, features, pca)
    
    return feature_data

    
def run1_kmeans(feature_data, features, k=3, print_=False):
    '''
    Perform k-means clustering, output updated df with Feature Cluster column corresponding
        to which cluster each galaxy row belongs.
    '''
    
    feature_data = feature_data.copy()   #some failsafe line to suppress the pandas warning
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    #this silly bracket addition is needed when making a new df col and not editing an existing one..?
    feature_data.loc[:, 'Feature Cluster'] = kmeans.fit_predict(feature_data[features])
    
    if print_:
        #print a summary of the feature medians in every cluster!
        
        '''
        features=get_feature_names()
        df=make_galfit_table()
        trim=trim_galfit_table(df.copy())
        sig=iqr_clipping(trim,features)
        
        sig['Feature Cluster'] = feature_data['Feature Cluster']
        cluster_summary = sig[features+['Feature Cluster']].groupby('Feature Cluster').median().round(3)
        print("\nFEATURE CLUSTER (STANDARDIZED) MEDIAN PROPERTIES:")
        print(cluster_summary)
        '''
        
        cluster_summary = feature_data[features+['Feature Cluster']].groupby('Feature Cluster').median().round(3)
        print("\nFEATURE CLUSTER (STANDARDIZED) MEDIAN PROPERTIES:")
        print(cluster_summary)
    
    return feature_data


def find_optimal_k(feature_data, features, min_k=2, max_k=10, plot=True):
    '''
    Use silhouette scoring to find the 'optimal k' for the given dataset.
    '''
    silhouettes = []
    K = np.arange(min_k, max_k+1)
    
    for k in K:
        faux_df = run1_kmeans(feature_data[features].copy(), k)
        silhouettes.append(silhouette_score(faux_df[features], faux_df['Feature Cluster']))
    
    if plot:
        from plotting_utils import plot_silhouette
        plot_silhouette(K, silhouettes)
    
    best_k = K[np.argmax(silhouettes)] #np.argmax returns index of maximum in array or list
    return best_k


####################################
# RUN IT ALL RUN IT ALL RUN IT ALL #
####################################
def galfit_kmeans():

    print('NOTE: be sure to edit galfit_parameters.py so parameters are to your liking!')
    
    #pull the full list of features which will be clustered
    features = get_feature_names()
    
    #generate the dataframe
    df_full = make_galfit_table()
    
    #trim the table. remove the errors and unphysical data
    df_trimmed = trim_galfit_table(df_full)
    
    #convert pixels to arcseconds, then arcseconds to kpc
    for band in params.BANDS:
        re_col = f'CRE_{band}'
        re_arcsec = px_to_arcsec(band, df_trimmed[re_col], params=params)
        re_kpc = arcsec_to_kpc(re_arcsec, df_trimmed['Vcosmic'])        
        
        df_trimmed[re_col] = px_to_arcsec(band, df_trimmed[re_col], params=params) 
    
    #remove pesky outliers that lie beyond 3-sigma of their respective features' means
    df_clipped = iqr_clipping(df_trimmed, features, k_clip=params.IQRCLIP)
    
    #scale the feature data.
    df_scaled = standardize_data(df_clipped, features)
    
    K = params.K
    
    #if user did not pre-select a K value, extract optimal K using the silhouette method
    if K is None:
        K = find_optimal_k(df_scaled, features, min_k=2, max_k=10, plot=params.PLOT_SILHOUETTES)
    
    #perform k-means clustering on the full set of features, using the K defined above.
    feature_data = run1_kmeans(df_scaled, features, k=K, print_=True)
    
    #if user indicated a preference for a corner plot in galfit_parameters.py, oblige them
    #must precede PCA if any, so that these features are not included in the analysis
    if params.PLOT_CORNER:
        from plotting_utils import plot_corner
        plot_corner(feature_data, features)
    
    #if the user should like a 2D plot of the clusters...
    if params.PLOT_CLUSTERS:
        from plotting_utils import plot_kmeans_clusters
        
        #reduce dimensionality to 2 if desired...
        #otherwise will default to the X and Y columns defined in galfit_parameters.py
        if params.PCA_FOR_PLOTTING:
            
            #create the PCA1, PCA2 columns. IGNORES 'Feature Cluster' column!
            #will also output a plot of the PCA vector components
            feature_data = pca_2d(feature_data, features, plot=params.PLOT_PCA_COMPONENTS)
            
        #plort.
        plot_kmeans_clusters(feature_data, x=params.X, y=params.Y, PCA=params.PCA_FOR_PLOTTING)
    
    #return the data for further analysis, if needed.
    return feature_data