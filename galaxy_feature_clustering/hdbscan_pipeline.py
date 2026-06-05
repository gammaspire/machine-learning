'''
OBJECTIVE: 
* apply HDBSCAN Feature Classing (unsupervised machine learning) to VFS GALFIT output parameters for optical grz, unWISE W1-4.

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
* determine the optimal eps and min_sample value using silhouette score (optionally generate
    silhouette plot)
    * quantitative metric of Feature Class quality
    * higher score --> more robust clustering
* use optimal eps and min_sample parameters to generate "cluster membership array"
    * row-matched to input data; contains integer value for every galaxy indicating 
        the cluster to which they belong
* if desired, plot in the X & Y chosen in galfit_parameters.py
'''

import os
import sys

sys.path.insert(0,'utils')

from data_utils import *
from table_utils import *
from conversion_utils import *

from feature_utils import get_feature_names
from clustering_utils import find_optimal_hdbparams, run1_hdbscan, umap_2d

from init_parameters import Params
params = Params()

import numpy as np
np.seterr(all='ignore')  #ignore those pesky log10() errors

import pandas as pd
from rich import print


####################################
# RUN IT ALL RUN IT ALL RUN IT ALL #
####################################
def run_hdbscan(colors=False,save_table=True):
    '''
    *If save_table=True, unscaled median feature data and their uncertainties will be saved as a .csv
        *save loc will be the same as the location of galfit_kmeans.py
    *Note that these magnitudes originate from extinction-corrected photometric fluxes
        courtesy of SGA2020
    '''

    print('NOTE: be sure to edit galfit_parameters.py so parameters are to your liking!')
    
    #pull the full list of features which will be clustered
    features = get_feature_names(params=params)
    
    if params.INCLUDE_MSTAR:
        features += ['logmstar', 'CN_W1-fixBA']
    
    print(f'USING THESE FEATURES: {features}')
    
    params.colors=colors
    
    if params.LOADTABLE:
        print(f'Reading feature data from {params.HDBSCAN_DF_PATH}...')
        df_scaled = pd.read_csv(params.HDBSCAN_DF_PATH)
    
    else:
        #generate the dataframe
        df_full = make_galfit_table(params)       

        #trim the table. remove the errors and unphysical data
        df_trimmed = trim_galfit_table(df_full, params)

        #convert effective radii (px) to effective radii (kpc)
        df_trimmed = get_kpc_columns(df_trimmed, params) 

        #replace inf, -inf with NaN
        df_trimmed = df_trimmed.replace([np.inf, -np.inf], np.nan)

        #replace any blanks with np.nan
        df_trimmed = df_trimmed.replace(r'^\s*$', np.nan, regex=True)
        
        #drop rows with ANY NaNs in the feature columns
        df_trimmed = df_trimmed.dropna(subset=features+['NUV_r']+['W1_W3'])

        #scale the feature data.
        df_scaled = standardize_data(df_trimmed, features)
    
    #write the table if SAVETABLE=True
    if params.SAVETABLE:
        print(f'A copy of the scaled galaxy features was written to {params.HDBSCAN_DF_PATH}!')
        df_scaled.to_csv(params.HDBSCAN_DF_PATH, index=False)

    #read the HDBSCAN parameters from init_parameters.py, if defined.
    MIN_CLUSTER_SIZE = params.MIN_CLUSTER_SIZE
    MIN_SAMPLES = params.MIN_SAMPLES
    METRIC = params.METRIC
    SELECTION_METHOD = params.SELECTION_METHOD
        
    #if user set this parameter to True, extract optimal values using a modified elbow method
    if params.OPTIMIZE_HDB_PARAMS:
        print('Calculating optimal HDBSCAN parameters...')
        MIN_CLUSTER_SIZE, MIN_SAMPLES, METRIC, SELECTION_METHOD = find_optimal_hdbparams(df_scaled, features)
    
    #perform HDBSCAN on the full set of features, using the parameters defined above.
    feature_data = run1_hdbscan(df_scaled, features, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                                metric=METRIC, cluster_selection_method=SELECTION_METHOD)
    
    #for HDBSCAN, will need to remove [-1] noise galaxies when looking at physical properties. 
    #UMAP is an exception, since it relies on the full data distribution
    #NOISE != CLUSTER
    clean_data = feature_data[feature_data['Feature Class'] != -1].copy()  
        
    #create a separate pandas dataframe comprising the median, uncertainty summary
    #use clean data here
    summary_rows = create_median_table(clean_data, features)
    cluster_summary = pd.DataFrame(summary_rows)
        
    if save_table:        
        loc = os.path.join(os.getcwd(), 'data/hdbscan_median_features.csv')
        print(f"\n A summary of Feature Class median properties saved to {loc}:")
        cluster_summary.to_csv(loc, index=False)
    
    #if params.PLOT_MEDIANS:
    #from plotting_utils import plot_group_features

    #plot medians.
    #plot_group_features(cluster_summary, layout_dict=params.LAYOUT_DICT)

    #if user indicated a preference for a corner plot in galfit_parameters.py, oblige them
    #must precede UMAP, so that the UMAP components are not included in the analysis
    if params.PLOT_CORNER:
        from plotting_utils import plot_corner
        
        #use clean_data here
        plot_corner(clean_data, features=None)
    
    #if the user should like a 2D plot of the clusters...
    #use ALL OF THE DATA here
    #if params.PLOT_CLUSTERS:
    from plotting_utils import plot_clusters

    #reduce dimensionality to 2 if desired...
    #otherwise will default to the X and Y columns defined in galfit_parameters.py
    if params.UMAP_FOR_PLOTTING:

        #create the Comp1, Comp2 columns. IGNORES 'Feature Class' column (i.e., noise not a problem)
        feature_data_umap = umap_2d(feature_data, features)

    #plort.
    plot_clusters(feature_data_umap,
                  x=params.X, y=params.Y, 
                  PCA=params.PCA_FOR_PLOTTING, UMAP=params.UMAP_FOR_PLOTTING)

    #self-explanatory. uninvolved. demure.
    if params.PLOT_ENV_FRACTION:
        from plotting_utils import plot_env_fraction
        plot_env_fraction(clean_data, main_only=False, envcomp=True) #no noise data, pls.
        plot_env_fraction(clean_data, main_only=False, envfrac=True) #no noise data, pls.
    
    #also self-explanatory. collected. uninhibited.
    if params.PLOT_SFRMSTAR:
        from plotting_utils import plot_sfrmstar
        plot_sfrmstar(clean_data)   #no light gray points allowed.
    
    #return the data for further analysis, if needed.
    return feature_data