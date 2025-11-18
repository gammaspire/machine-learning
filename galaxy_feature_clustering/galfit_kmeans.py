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
from dataprocessing_utils import px_to_arcsec, arcsec_to_kpc, nmaggies_to_mag

from galfit_parameters import Params
params = Params()

import numpy as np
np.seterr(all='ignore')  #ignore those pesky log10() errors

from scipy.stats import zscore
from astropy.table import Table
import pandas as pd
from rich import print

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_feature_names(colors=False, flux=False):
    #define Re, Sersic index feature columns
    re_cols = [f'CRE_{band}' for band in params.BANDS]
    nser_cols = [f'CN_{band}' for band in params.BANDS]
    
    #combine
    features = re_cols + nser_cols
    
    #use averages of g&r, W1&W2 effective radii
    if 'CRE_r' and 'CRE_g' in features:
        print('Using average g and r effective radius!')
        features = [f for f in features if f not in ['CRE_r', 'CRE_g']] + ['AVG_RE_gr']
    
    if 'CRE_W1-fixBA' and 'CRE_W2' in features:
        print('Using average W1 and W2 effective radius!')
        features = [f for f in features if f not in ['CRE_W1-fixBA', 'CRE_W2']] + ['AVG_RE_W1W2']
    
    if colors:
        features += ['NUV_r','W1_W4']
    
    if flux:
        for band in params.BANDS:
            band = band.split('-')[0].upper()
            features += [f'FLUX_SB22_{band}']
            features += [f'FLUX_SB25_{band}']

    #and return
    return features


def get_vcosmic_column():
    env = Table.read('data/vf_v2_environment.fits')
    return env['Vcosmic']


def get_stellar_columns():
    cigale = Table.read('data/cigale_vf_metallicity.fits')
    
    mstar = np.log10(cigale['bayes.stellar.m_star'])
    sfr = np.log10(cigale['bayes.sfh.sfr'])
    
    return mstar, sfr


def get_photometric_colors():
    #needed photometric fluxes (in nanomaggies)
    phot = Table.read('data/vf_v2_legacy_ephot.fits')['FLUX_AP06_NUV','FLUX_AP06_R',
                                                       'FLUX_AP06_W1','FLUX_AP06_W4']
    #extinction corrections
    ext = Table.read('data/vf_v2_extinction.fits')['A(NUV)_SandF', 'A(R)_SandF',
                                                   'A(W1)_SandF', 'A(W4)_SandF']
    band = ['NUV', 'R', 'W1', 'W4']
    #convert phot fluxes to extinction-corrected AB magnitudes
    for i in range(4):   #0, 1, 2, 3...NUV, R, W1, W4
        mAB_corr = nmaggies_to_mag(phot[f'FLUX_AP06_{band[i]}'], ext[f'A({band[i]})_SandF'])        
        phot[f'mAB_{band[i]}'] = mAB_corr
    
    NUV_r = phot[f'mAB_NUV'] - phot['mAB_R']
    W1_W4 = phot[f'mAB_W1'] - phot['mAB_W4']
    
    return NUV_r, W1_W4


def get_SB_flux():
    '''
    AIM: pull flux measurements at the needed wavelengths for SB22, SB25
    '''
    phot = Table.read('data/vf_v2_legacy_ephot.fits')
    
    phot_sb = Table()
    
    for band in params.BANDS:
        band = band.split('-')[0].upper()  #split band into components delimited by '-' (e.g., 'W3-fixBA')
                                           #then take the zeroth component
                                           #THEN capitalize all alphabet characters

        phot_sb[f'FLUX_SB22_{band}'] = np.log10(phot[f'FLUX_SB22_{band}'])
        phot_sb[f'FLUX_SB25_{band}'] = np.log10(phot[f'FLUX_SB25_{band}'])
    
    phot_sb = phot.to_pandas()
    
    return phot_sb


def make_galfit_table(colors=False,flux=False):
    '''
    Read GALFIT grz, W1-4 output tables
        * If colors=True, will include NUV-r, W1-W4 colors
        * If 
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
    
    #append mstar, sfr columns
    data_table['logmstar'], data_table['logsfr'] = get_stellar_columns()
    
    #add a size ratio column...just because. (I actually need it for analysis.)
    data_table['Size Ratio'] = data_table['CRE_W3-fixBA'] / data_table['CRE_W1-fixBA']
    
    if colors:
        NUV_r, W1_W4 = get_photometric_colors()
        data_table.add_columns([NUV_r,W1_W4], names=['NUV_r','W1_W4'])
    
    data_table = data_table.to_pandas()

    if flux:
        phot_tab = get_SB_flux()
        data_table = pd.concat([data_table.copy(), phot_tab.copy()],axis=1)

    return data_table


def add_average_re(data_table):
    '''
    AIM: append average g & r, W1 & W2 effective radii if the columns exist.
    '''
    if 'CRE_r' and 'CRE_g' in data_table.columns:
        data_table['AVG_RE_gr'] = (data_table['CRE_g'] + data_table['CRE_r'])/2
    if 'CRE_W1-fixBA' and 'CRE_W2' in data_table.columns:
        data_table['AVG_RE_W1W2'] = (data_table['CRE_W1-fixBA'] + data_table['CRE_W2'])/2
    
    return data_table


def get_kpc_columns(data_table):
    '''
    AIM: convert pixels to arcseconds, then arcseconds to kpc for every effective radius column.
    '''
    for band in params.BANDS:
        re_col = f'CRE_{band}'
        re_arcsec = px_to_arcsec(band, data_table[re_col], params=params)
        re_kpc = arcsec_to_kpc(re_arcsec, data_table['Vcosmic'])                
        data_table[re_col] = re_kpc
    
    return data_table


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
        
    #if magnitude colors are in the list of features, then we have to apply
    #a quality flag here too. This amount to just dropping the NaNs
    if 'NUV_r' in full_df.columns:
        full_df = full_df.copy().dropna()
        message=f'Removed {ngal_before - len(full_df)}/{ngal_before} galaxies with GALFIT and PHOT quality flags.'
    else:
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
    output: edited pandas dataframe with input columns standardized, scaler object
    '''
    df = df.copy()
    
    #create 'new' dataframe; add '_unscaled' to the feature column names
    unscaled = df[features].add_suffix("_unscaled")

    #initiate the scaler
    scaler = StandardScaler()
    
    #apply the scaler to transform the features
    df[features] = scaler.fit_transform(df[features])

    #concatenate the transformed features with the unscaled features!
    df = pd.concat([df, unscaled], axis=1)
    
    #ANNNNNND return
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
        plot_pca_components(feature_data, features, pca, cmap_name='tab20')
    
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
    
    #print a summary of the feature medians in every cluster!
    if print_:
        #get list of unscaled feature columns
        features_unscaled = [feature+'_unscaled' for feature in features]
        
        #add Feature Cluster and Size Ratio integer columns to the mix
        #we need Size Ratio!
        all_columns = features_unscaled+['Feature Cluster']+['Size Ratio']
        
        #generate a summary of the *medians*
        cluster_summary = feature_data[all_columns].groupby('Feature Cluster').median().round(3)
        
        #annnd print.
        print("\nFEATURE CLUSTER MEDIAN PROPERTIES:")
        print(cluster_summary)
    
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


####################################
# RUN IT ALL RUN IT ALL RUN IT ALL #
####################################
def galfit_kmeans(colors=False, flux=False):
    '''
    *If colors=True, the kmeans features will include the following magnitude colors:
        *NUV - r
        *W1 - W4
    *If flux=True, kmeans features will include the following surface brightness flux measurements:
        *FLUX_SB22_{band}
        *FLUX_SB25_{band}
    *Note that these magnitudes originate from extinction-corrected photometric fluxes
        courtesy of SGA2020
    '''

    print('NOTE: be sure to edit galfit_parameters.py so parameters are to your liking!')
    
    #pull the full list of features which will be clustered
    features = get_feature_names(colors=colors,flux=flux)
    
    print(f'USING THESE FEATURES: {features}')
    
    #generate the dataframe
    df_full = make_galfit_table(colors=colors,flux=flux)       
        
    #trim the table. remove the errors and unphysical data
    df_trimmed = trim_galfit_table(df_full)
    
    df_trimmed = get_kpc_columns(df_trimmed)
    
    #calculate average g & r, W1 & W2 effective radii columns to df_trimmed (if those columns exist)
    df_trimmed = add_average_re(df_trimmed)   
    
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
        plot_corner(feature_data, features=None)
    
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