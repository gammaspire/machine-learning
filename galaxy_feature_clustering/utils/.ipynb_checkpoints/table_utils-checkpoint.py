# Main  Table Utility Functions #

'''
AIM: create the clean and sparkly dataframe of galaxy features!
'''

from astropy.table import Table
import numpy as np
import pandas as pd

from conversion_utils import get_photometric_colors

##############################
# GET COLUMN AND TABLE NAMES #
##############################

def read_phot_tables():
    #needed NUV, R, W1, W3 photometric fluxes (in nanomaggies)
    phot = Table.read('data/vf_v2_legacy_ephot.fits')['FLUX_AP06_NUV','FLUX_AP06_R',
                                                       'FLUX_AP06_W1','FLUX_AP06_W3']
    #extinction corrections
    ext = Table.read('data/vf_v2_extinction.fits')['A(NUV)_SandF', 'A(R)_SandF',
                                                   'A(W1)_SandF', 'A(W3)_SandF']
    #snr column
    snr = Table.read('data/virgowise_data.fits')['SNR_phot']
    
    return phot, ext, snr


#yes, I read the environment table twice. I like organization. cope.
def get_vcosmic_column():
    env = Table.read('data/vf_v2_environment.fits')
    return env['VFID'], env['Vcosmic']


def get_env_columns():
    '''
    *set up the environment flags...
    *very basic setup -- no environment, save pure field, is entirely decoupled from the others.
    '''
    env = Table.read('data/vf_v2_environment.fits')['cluster_member', 'rich_group_memb',
                                                    'poor_group_memb', 'filament_member', 'pure_field']
    return env.to_pandas()


def get_stellar_columns():
    
    from data_utils import get_ms_line, get_delta_logsfr
    
    cigale = Table.read('data/cigale_vf_metallicity.fits')
    mstar = np.log10(cigale['bayes.stellar.m_star'])
    sfr = np.log10(cigale['bayes.sfh.sfr'])
    
    #magphys = Table.read('data/vf-altphot.fits')
    #mstar = magphys['combined_logMstar_med']
    #sfr = magphys['combined_logSFR_med']
    
    #before ANY trimming is applied to the data,
    #determine the main sequence line fit to log(ssfr)>-11.5 galaxies
    m, b = get_ms_line(mstar,sfr)
    delta_sfr = get_delta_logsfr(mstar, sfr, m, b)
    
    
    return mstar, sfr, delta_sfr


################################
# INITIALIZE THE FEATURE TABLE #
################################

def make_galfit_table(params, colors=False):
    '''
    Read GALFIT grz, W1-4 output tables
        * If colors=True, will include NUV-r, W1-W3 colors
    Convert from astropy Table to pandas df
    Output --> dataframe with grz, W1-3 Re, nser, CXC, r-band inclination (CAR), CNumerical_Error columns
    '''
    #create empty astropy table
    data_table = Table()
    
    #put ALL bands in the dataframe.
    for band in params.BANDS:
        t = Table.read(f'data/vf_v2_galfit_{band}.fits')
        for colname in params.COLUMNS:
            data_table[f'{colname}_{band}'] = t[colname]
            
            #if r-band, pull the axis ratio!
            if band=='r':
                data_table[f'Axis Ratio'] = t['CAR']
        
    #append the VFID and Vcosmic columns
    data_table['VFID'], data_table['Vcosmic'] = get_vcosmic_column()
    
    #append mstar, sfr columns
    data_table['logmstar'], data_table['logsfr'], data_table['delta_logsfr'] = get_stellar_columns()
    
    #add a size ratio column...just because. (I actually need it for analysis. not used for clustering.'d)
    data_table['Size Ratio'] = data_table['CRE_W3-fixBA'] / data_table['CRE_W1-fixBA']
    
    #add NUV-r, W1-W3 colors
    phot, ext, snr = read_phot_tables()
    NUV_r, W1_W3 = get_photometric_colors(phot, ext)   #from conversion_utils
    data_table.add_columns([NUV_r,W1_W3,snr], names=['NUV_r','W1_W3','SNR'])
    
    data_table = data_table.to_pandas()

    #append environment columns
    envflags = get_env_columns()
    data_table = pd.concat([data_table.copy(), envflags.copy()],axis=1)

    return data_table


#######################################################
# REMOVE ROWS WITH GALFIT (AND PHOT) NUMERICAL ERRORS #
#######################################################

def trim_galfit_table(full_df, params):
    '''
    Apply trimming flags -- non-data, Re, nser, numerical error
    '''
    
    #unfortunately have to trim any row which has at least one of the below problems.
    
    ngal_before = len(full_df)
    
    #center x-position, Nser, numerical error columns
    xyc_cols = [f'CXC_{band}' for band in params.BANDS_TO_CLUSTER]
    nser_cols = [f'CN_{band}' for band in params.BANDS_TO_CLUSTER]
    numerr_cols = [f'CNumerical_Error_{band}' for band in params.BANDS_TO_CLUSTER]
    
    #drop row if any central x pixel coordinate cell is zero
    full_df = full_df.loc[~(full_df[xyc_cols]==0).any(axis=1)]
        
    #drop rows with any nser > 6.
    full_df = full_df.loc[~(full_df[nser_cols]>6).any(axis=1)]
        
    #drop rows with any convolved numerical error
    full_df = full_df.loc[~(full_df[numerr_cols]).any(axis=1)]
    
    #apply the logMstar, logSFR completeness limit flags. 
    #note: if either or both set to None in init_parameters.py, then this function will do nothing.
    full_df = completeness_limits(full_df, params.LOGMSTAR_LIM, params.LOGSFR_LIM)
    
    #apply W3 SNR limit (TEST)
    #full_df = full_df.loc[~(full_df['SNR']<10.)]
    
    #apply inclination cut (remove galaxies with B/A < 0.25)
    full_df = full_df.loc[full_df['Axis Ratio']>=0.25]
        
    #if magnitude colors are in the list of features, then we have to apply
    #a quality flag here too. This amount to just dropping the NaNs
    if 'NUV_r' in full_df.columns:
        
        #filter out non-finite numeric values (inf/-inf)
        full_df = full_df[np.isfinite(full_df.select_dtypes(include=[np.number])).all(axis=1)]
        
        message=f'Removed {ngal_before - len(full_df)}/{ngal_before} galaxies with GALFIT, completeness limit, inclination, and PHOT quality flags.'
    
    else:
        message=f'Removed {ngal_before - len(full_df)}/{ngal_before} galaxies with completeness limit, inclination, and GALFIT quality flags.'        
    
    print('#'*len(message))
    print(message)
    print('#'*len(message))
    
    #return the 'cleansed' dataframe
    return full_df


###########################################
# APPLYING SFR, MSTAR COMPLETENESS LIMITS #
# Used in the "trimming" function above #
###########################################

def completeness_limits(trimmed_df, mstar_limit=None, sfr_limit=None):
    '''
    Apply logmstar, logsfr completeness limits from the Virgowise sample paper!
    '''
    
    print(f'Applying the following completeness limits: logMstar > {mstar_limit}, logSFR > {sfr_limit}')
    
    #initialize the sfr, mstar flags as all-True flags (equivalent to multiplying by 1)
    sfr_flag = np.ones(len(trimmed_df),dtype=bool)
    mstar_flag = np.ones(len(trimmed_df),dtype=bool)
    
    #if the user actually put integers into the galfit_parameters.py file, CHANGE THE BOOLS
    if sfr_limit is not None:
        sfr_flag = (trimmed_df['logsfr'] > sfr_limit)
    if mstar_limit is not None:
        mstar_flag = (trimmed_df['logmstar'] > mstar_limit)
    
    complete_df = trimmed_df.copy()[mstar_flag & sfr_flag]
    
    return complete_df


##################################
# STANDARDIZING THE FEATURE DATA #
##################################

def standardize_data(df, features):
    '''
    Standardize data features such that each column has a mean of 0 and a standard deviation of 1.
    output: edited pandas dataframe with input columns standardized, scaler object
    '''
    from sklearn.preprocessing import StandardScaler
    
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


###################################
# REASSIGN K-MEANS CLUSTER LABELS #
###################################

def k_reassignment(feature_data):
    '''
    AIM: for k=3 clusters, there is a set color/marker palette cadence I would like to follow. This cadence is often not met despite the marker_palette() function in plotting_utils.py, since the k assignment may change with changed parameters or feature inputs. FG1 will forever be seagreen, but both may be assigned to a completely different cluster of galaxies. I want FG1 - seagreen - suppressed galaxy population, and thus this function was born.
    
    * Must be run AFTER k-means clustering!
    * Will only run on k=3 model labels
    * Rules:
        * 'suppressed galaxy population' --> smallest np.median('Size Ratio') --> FG1
        * Of the remaining two:
            * most massive population --> largest np.median('CRE_g') --> FG2
            * least massive population --> smallest np.median('CRE_g') --> FG0
    '''
    
    #define the clusters, TYPICALLY (0,1,2)
    clusters = sorted(feature_data['Feature Cluster'].unique())
    
    #only run if k=3 AND k-means clustering has already run
    if (len(clusters)!=3) or ('Feature Cluster' not in feature_data.columns):
        return
    
    #create clean copy of feature_data dataframe:
    df = feature_data.copy()
        
    #calculate per-cluster medians
    #groups df by Feature Cluster, calculates median of FG0, FG1, FG2 size ratio and cre_g
    stats = (df.groupby('Feature Cluster')[['Size Ratio', 'CRE_g']].median())
    
    #identify the index where the minimum size ratio exists -- this is the "old" k-value for what will be FG1
    fg1_old = stats['Size Ratio'].idxmin()

    #isolate the remaining Feature Groups assuming they are not part of this suppressed population
    remainder = [c for c in clusters if c != fg1_old]

    #FG2 = larger median CRE_g among remainder
    #FG0 = smaller
    #pull the two stats rows with remainder indices; sort from lowest to highest; convert to indices; convert to list.
    remainder_sorted = (stats.loc[remainder].sort_values('CRE_g', ascending=True).index.tolist())
    
    #determine the "old" k-values for what will be FG0 and FG2
    fg0_old = remainder_sorted[0]   #smaller are all FG0 galaxies
    fg2_old = remainder_sorted[1]   #larger are all FG2 galaxies

    #create mapping dictionary!
    mapping = {fg0_old: 0, fg1_old: 1, fg2_old: 2}

    #add the mapping dictionary to df...
    df['Feature Cluster'] = df['Feature Cluster'].map(mapping)

    #and lastly, return the df.
    return df


#############################################################
# CREATE TABLE SUMMARY OF MEDIAN FEATURE CLUSTER PROPERTIES #
#############################################################

def create_median_table(feature_data, features):
    '''
    AIM: save a summary of the feature medians + bootstrap uncertainties in every feature cluster
    '''
    from data_utils import get_bootstrap_confint
    
    #get list of unscaled feature columns, including the size ratios
    features_unscaled = [feature+'_unscaled' for feature in features] + ['Size Ratio'] + ['NUV_r'] + ['W1_W3']

    #initialize the rows
    summary_rows = []

    #for every cluster_id (e.g., k=0), isolate the rows which belong to that cluster_id
    for cluster_id, df_cluster in feature_data.groupby("Feature Cluster"):

        #create a dictionary. will be adding medians and such in the loop below.
        row = {"Feature Cluster": cluster_id}

        #now, for every (unscaled) feature...
        for feature in features_unscaled:

            #isolate the feature from the cluster_id data
            arr = df_cluster[feature].values

            #calculate the median and lower+upper bootstrap confidence intervals
            med = np.median(arr)
            low, high = get_bootstrap_confint(arr)

            #store median + error in the row set for that feature cluster
            row[feature] = med
            row[feature+"_err_low"] = med - low
            row[feature+"_err_high"] = high - med

        #add the row...
        summary_rows.append(row)
    
    return summary_rows