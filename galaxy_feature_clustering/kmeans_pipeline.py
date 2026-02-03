'''
OBJECTIVE: 
* apply k-means feature clustering (unsupervised machine learning) to VFS GALFIT output parameters for optical grz, unWISE W1-4.

NEED:
* vf_v2_galfit_{band}.fits
    * band = g, r, z, W1-fixBA, W2, W3-fixBA W4
    * should ALL be row-matched!

SCAFFOLD PROCEDURE:
* open FITS files using astropy.table.Table, convert to pandas dataframe
* apply these quality check flags to every set of data:
    * ('CXC' > 0, 'CYC' > 0) --> if nonzero, then the row will have no data
    * (nser_{band} <= 6.) --> a Sersic index > 6 is unphysical
    * (~'CNumerical_Error') --> if True, then GALFIT hiccupped for this galaxy and 
        generated an unreliable model
* convert all effective radii to kpc!
* standardize all data using sklearn.StandardScaler
* determine the optimal k value using silhouette score (optionally generate silhouette plot)
    * quantitative metric of feature cluster quality
    * higher score --> more robust clustering
* use optimal k-value to generate "cluster membership array"
    * row-matched to input data; contains integer value for every galaxy indicating 
        the cluster to which they belong
* if desired, plot in either 2D PCA space or in the X & Y chosen in galfit_parameters.py
'''

import os
import sys

sys.path.insert(0,'utils')

from data_utils import *
from table_utils import *
from conversion_utils import *

from feature_utils import get_feature_names
from clustering_utils import find_optimal_k, run1_kmeans, pca_2d

from init_parameters import Params
params = Params()

import numpy as np
np.seterr(all='ignore')  #ignore those pesky log10() errors

import pandas as pd
from rich import print


####################################
# RUN IT ALL RUN IT ALL RUN IT ALL #
####################################
def run_kmeans(colors=False, save_table=True):
    '''
    *If colors=True, the kmeans features will include the following magnitude colors:
        *NUV - r
        *W1 - W3
    *If save_table=True, unscaled median feature data and their uncertainties will be saved as a .csv
        *save loc will be the same as the location of galfit_kmeans.py
    *Note that these magnitudes originate from extinction-corrected photometric fluxes
        courtesy of SGA2020
    '''

    print('NOTE: be sure to edit galfit_parameters.py so parameters are to your liking!')
    
    #pull the full list of features which will be clustered
    features = get_feature_names(params=params, colors=colors)
        
    print(f'USING THESE FEATURES: {features}')
      
    if params.LOADTABLE:
        print(f'Reading feature data from {params.KMEANS_DF_PATH}...')
        df_scaled = pd.read_csv(params.KMEANS_DF_PATH, na_values=['', ' '])

    else:
        #generate the dataframe
        df_full = make_galfit_table(params, colors=colors)  
        
        #trim the table. remove the errors and unphysical data.
        df_trimmed = trim_galfit_table(df_full, params)

        #convert effective radii (px) to effective radii (kpc)
        df_trimmed = get_kpc_columns(df_trimmed, params)
        
        if params.IQRCLIP is not None:
            #remove pesky outliers that lie beyond 3-sigma of their respective features' means
            df_clipped = iqr_clipping(df_trimmed, features, k_clip=params.IQRCLIP)
        else:
            df_clipped = df_trimmed.copy()
        
        #scale the feature data.
        df_scaled = standardize_data(df_clipped, features)
    
    #write the table if SAVETABLE=True
    if params.SAVETABLE:
        print(f'A copy of unscaled/scaled galaxy features was written to {params.KMEANS_DF_PATH}!')
        df_scaled.to_csv(params.KMEANS_DF_PATH, index=False)
    
    #define k cluster variable
    K = params.K

    #if user did not pre-select a K value, extract optimal K using the silhouette method
    if K is None:
        K = find_optimal_k(df_scaled, features, min_k=2, max_k=10, plot=params.PLOT_SILHOUETTES)
        
    #perform k-means clustering on the full set of features, using the K defined above.
    feature_data = run1_kmeans(df_scaled, features, k=K)
      
    if params.PLOT_RAINCLOUDS:
        from plotting_utils import feature_rainclouds
        
        #plot rainclouds.
        feature_rainclouds(feature_data, feature_list=params.FEATURE_LIST)
        
    #create a separate pandas dataframe comprising the median, uncertainty summary
    summary_rows = create_median_table(feature_data, features)
    cluster_summary = pd.DataFrame(summary_rows)
        
    if save_table:
        loc = os.path.join(os.getcwd(), 'data/kmeans_median_features.csv')
        print(f"\n A summary of feature cluster median properties saved to {loc}:")
        cluster_summary.to_csv(loc, index=False)

    if params.PLOT_MEDIANS:
        from plotting_utils import plot_group_features
        
        #plot medians.
        plot_group_features(cluster_summary, layout_dict=params.LAYOUT_DICT)

    #if user indicated a preference for a corner plot in galfit_parameters.py, oblige them
    #must precede PCA if any, so that these features are not included in the analysis
    if params.PLOT_CORNER:
        from plotting_utils import plot_corner
        plot_corner(feature_data, features=None)
    
    #if the user should like a 2D plot of the clusters...
    if params.PLOT_CLUSTERS:
        from plotting_utils import plot_clusters
        
        #reduce dimensionality to 2 if desired...
        #otherwise will default to the X and Y columns defined in galfit_parameters.py
        if params.PCA_FOR_PLOTTING:
            
            #create the PCA1, PCA2 columns. IGNORES 'Feature Cluster' column!
            #will also output a plot of the PCA vector components
            feature_data = pca_2d(feature_data, features, plot=params.PLOT_PCA_COMPONENTS)
            
        #plort.
        plot_clusters(feature_data, x=params.X, y=params.Y, PCA=params.PCA_FOR_PLOTTING)
    
    #self-explanatory. uninvolved. demure.
    if params.PLOT_ENV_FRACTION:       
        from plotting_utils import plot_env_fraction, plot_env_stacked_hist
        plot_env_fraction(feature_data, main_only=False, envfrac=True, envcomp=False)
        plot_env_fraction(feature_data, main_only=False, envfrac=False, envcomp=True)
        
        #this plot is a histogram companion to envcomp=True
        plot_env_stacked_hist(feature_data, main_only=False)
        
    #also self-explanatory. collected. uninhibited.
    if params.PLOT_SFRMSTAR:
        from plotting_utils import plot_sfrmstar
        plot_sfrmstar(feature_data)
    
    #return the data for further analysis, if needed.
    return feature_data