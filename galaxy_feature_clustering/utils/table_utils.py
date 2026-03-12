# Main  Table Utility Functions #

'''
AIM: create the clean and sparkly dataframe of galaxy features!
'''

from astropy.table import Table
import numpy as np
import pandas as pd

from conversion_utils import get_photometric_colors, calculate_SNR

##############################
# GET COLUMN AND TABLE NAMES #
##############################    


def read_phot_tables():
    
    #needed NUV, R, W1, W3 photometric fluxes (in nanomaggies)
    flux_cols = ['FLUX_AP06_NUV', 'FLUX_AP06_W3',
                 'FLUX_AP06_R', 'FLUX_AP06_W1',
                 'FLUX_AP06_G']
    
    #grab the W3, NUV, g, W1 ivar columns (also in nanomaggies)
    ivar_cols = ['FLUX_IVAR_AP06_NUV', 'FLUX_IVAR_AP06_W3', 'FLUX_IVAR_AP06_W1', 'FLUX_IVAR_AP06_G'] 
    
    #also grab RA, DEC columns. :-)
    radec_cols = ['RA_MOMENT', 'DEC_MOMENT']
    
    #one more column...bright star flag!
    bs_col = ['BRIGHTSTAR', 'MEDIUMSTAR']
    
    phot = Table.read('data/vf_v2_legacy_ephot.fits')[flux_cols+ivar_cols+radec_cols+bs_col]   
    
    #extinction corrections
    ext = Table.read('data/vf_v2_extinction.fits')['A(NUV)_SandF', 'A(R)_SandF',
                                                   'A(W1)_SandF', 'A(W3)_SandF']
    
    return phot, ext


#yes, I read the environment table twice. I like organization. cope.
def get_vcosmic_column():
    env = Table.read('data/vf_v2_environment.fits')
    return env['VFID'], env['Vcosmic']


#pull Hubble t-type from Hyperleda catalog (which I then saved to virgowise_data.fits)
def get_ttype_column():
    hyp = Table.read('data/virgowise_data.fits')
    return hyp['t_type']


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

    #before ANY trimming is applied to the data,
    #determine the main sequence line fit to log(ssfr)>-11.5 galaxies
    m, b = get_ms_line(mstar,sfr)
    delta_sfr = get_delta_logsfr(mstar, sfr, m, b)
    
    return mstar, sfr, delta_sfr


###################################################
# CREATE BOOL COLUMNS TO ISOLATE dSFR POPULATIONS #
#   * Main Sequence
#   * Transition
#   * Suppressed
###################################################

def dsfr_columns(df, n_pop):
    '''
    Aim: Create bool columns to isolate the dSFR populations.
    * df --> dataframe of galaxies with dSFR column
    * pop_list --> number of populations (integer; 2 or 3)
        * 3 --> main sequence, transition, suppressed (respectively)
        * 2 --> main sequence, suppressed (respectively)
    Result: row-matched boolean flags for each population type!
    '''
    
    if 'delta_logsfr' not in df.columns:
        print('Cannot add dSFR bool columns! Need "delta_logsfr" column to continue.')
        return
        
    #if only two populations given, then separate into main sequence and suppressed
    if n_pop==2:
        pop1_flag = (df['delta_logsfr']>-1.)
        pop3_flag = (df['delta_logsfr']<=-1.)
    
    #if three populations given, then separate into main sequence, transition, and suppressed
    elif n_pop==3:
        pop1_flag = (df['delta_logsfr']>-0.5)
        pop2_flag = (df['delta_logsfr']<=-0.5) & (df['delta_logsfr']>=-2.)
        pop3_flag = (df['delta_logsfr']<-2.)
        
        #add transition flag
        df['transition_pop'] = pop2_flag
    
    else:
        print('Number of populations must be 2-3. Unable to continue.')
        return
    
    #add main sequence, suppressed flags
    df['ms_pop'] = pop1_flag
    df['suppressed_pop'] = pop3_flag
    
    #return the updated dataframe
    return df


################################
# INITIALIZE THE FEATURE TABLE #
################################

def make_galfit_table(params):
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
            
            #if 'CRE' in colname:
            #    data_table[f'{colname}_{band}'] = np.log10(t[colname])
            
            #if r-band, pull the axis ratio!
            if band=='r':
                data_table[f'Axis Ratio'] = t['CAR']
        
    #append the VFID and Vcosmic columns
    data_table['VFID'], data_table['Vcosmic'] = get_vcosmic_column()
    
    #append mstar, sfr columns
    data_table['logmstar'], data_table['logsfr'], data_table['delta_logsfr'] = get_stellar_columns()
    
    #add a size ratio column...just because. (I actually need it for analysis. not used for clustering.)
    data_table['Size Ratio'] = data_table['CRE_W3-fixBA'] / data_table['CRE_W1-fixBA']
    
    #add NUV-r, W1-W3 colors
    phot, ext = read_phot_tables()
    NUV_r, W1_W3 = get_photometric_colors(phot, ext)   #from conversion_utils
    data_table.add_columns([NUV_r,W1_W3], names=['NUV_r','W1_W3'])
    
    #add RA, DEC columns
    data_table.add_columns([phot['RA_MOMENT'], phot['DEC_MOMENT']], names=['RA','DEC'])
    
    #add Hubble t-type column
    data_table['t_type'] = get_ttype_column()
    
    #add SNR columns!
    data_table['SNR_W1'] = calculate_SNR(phot['FLUX_AP06_W1'], phot['FLUX_IVAR_AP06_W1'])
    data_table['SNR_W3'] = calculate_SNR(phot['FLUX_AP06_W3'], phot['FLUX_IVAR_AP06_W3'])
    data_table['SNR_g'] = calculate_SNR(phot['FLUX_AP06_G'], phot['FLUX_IVAR_AP06_G'])
    data_table['SNR_NUV'] = calculate_SNR(phot['FLUX_AP06_NUV'], phot['FLUX_IVAR_AP06_NUV'])
    
    #add bright star flag!
    data_table['BRIGHTSTAR_FLAG'] = phot['BRIGHTSTAR']
    data_table['MEDIUMSTAR_FLAG'] = phot['MEDIUMSTAR']
    
    data_table = data_table.to_pandas()

    #append environment columns
    envflags = get_env_columns()
    data_table = pd.concat([data_table.copy(), envflags.copy()],axis=1)

    return data_table


def trim_colors(df, color_cols=['NUV_r','W1_W3'], print_=True):
    '''
    AIM: Mask the input Pandas dataframe to remove galaxy rows with illegitimate color magnitudes. 
    * Used for trim_galfit_table() and create_median_table()
    * color_cols is to be a list of strings identifying the relevant column names.
    '''
    
    #isolate the length of the dataframe before the mask
    ngal_before_cut = len(df)

    #create a mask which isolates the rows with FINITE color magnitudes
    mask = np.isfinite(df[color_cols]).all(axis=1)
    df_masked = df[mask]

    if print_:
        print(f'ALERT! Removed {ngal_before_cut - len(df_masked)} after vetting inf/-inf photometric entries.')
    
    return df_masked


def trim_ratios(df, print_=True):
    '''
    AIM: Mask the input Pandas dataframe to remove size ratios calculated with W3 SNR < 10.
    * Used in create_median_table() and plotting_utils.py -- 
    '''
    #isolate length of dataframe before mask
    ngal_before_cut=len(df)
    
    #create a mask to isolate galaxies with a W3 SNR > 10.
    mask = (df['SNR_W3']>10.)
    df_masked = df[mask]
    
    if print_:
        print(f'ALERT! Removed {ngal_before_cut - len(df_masked)} after limiting W3 SNR > 10.')
    
    return df_masked


#######################################################
# REMOVE ROWS WITH GALFIT (AND PHOT) NUMERICAL ERRORS #
#######################################################

def trim_galfit_table(full_df, params):
    '''
    Apply trimming flags -- non-data, Re, nser, numerical error; SNR; stellar mass completeness
    '''
    
    #unfortunately have to trim any row which has at least one of the below "problems."
    
    ngal_before = len(full_df)
    
    #apply W3, NUV SNR limit
    snr_limit = (full_df['SNR_W3']>=5.) | (full_df['SNR_NUV']>=5.)
    print(f'ALERT! Removed {np.sum(~snr_limit)} galaxies after applying the W3, NUV SNR limit.')
    df_snrtrim1 = full_df.loc[snr_limit]
    
    #apply W1 SNR limit
    snr_limit_w1 = (df_snrtrim1['SNR_W1']>=20.)
    print(f'ALERT! Removed {np.sum(~snr_limit_w1)} galaxies after applying the W1 SNR limit.')
    df_snrtrim = df_snrtrim1.loc[snr_limit_w1]
    
    #isolate the center x-position, Nser, numerical error columns
    nser_cols = [f'CN_{band}' for band in params.BANDS_TO_CLUSTER]
    re_cols = [f'CRE_{band}' for band in params.BANDS_TO_CLUSTER]
    xc_cols = [f'CXC_{band}' for band in params.BANDS_TO_CLUSTER]
    numerr_cols = [f'CNumerical_Error_{band}' for band in params.BANDS_TO_CLUSTER]
    
    #drop row if any nser or re model value is zero (indicates the model did not finish)
    zero_flag = (df_snrtrim[xc_cols]==0)
    df_one = df_snrtrim.loc[~(zero_flag).any(axis=1)]
    print(f'ALERT! Removing {len(df_snrtrim)-len(df_one)} galaxies with no GALFIT fit.')
        
    #drop rows with any nser > 6.
    df_two = df_one.loc[~(df_one[nser_cols]>6).any(axis=1)]
    print(f'ALERT! Removing {len(df_one) - len(df_two)} with GALFIT nser > 6 in one of the bands used for k-means clustering.')
        
    #drop rows with any convolved numerical error
    df_three = df_two.loc[~(df_two[numerr_cols]).any(axis=1)]
    print(f'ALERT! Removing {len(df_two) - len(df_three)} with a GALFIT numerical error.')
    
    #apply the bright star flag (from JM's photometry catalog)
    bs_flag = df_three['BRIGHTSTAR_FLAG'] & df_three['MEDIUMSTAR_FLAG']
    df_bs = df_three.loc[~bs_flag]
    print(f'ALERT! Removing {np.sum(bs_flag)} galaxies with a nearby bright star.')
    
    #apply the logMstar, logSFR completeness limit flags. 
    #note: if either or both set to None in init_parameters.py, then this function will do nothing.
    df_four = completeness_limits(df_bs, params.LOGMSTAR_LIM, params.LOGSFR_LIM)
    print(f'ALERT! Removing {len(df_bs) - len(df_four)} which do not pass any Mstar, SFR completeness limits specified in init_parameters.txt')
    
    #apply inclination cut (remove galaxies with B/A < 0.25)
    df_five = df_four.loc[df_four['Axis Ratio']>=0.25]
    print(f'ALERT! Removed {len(df_four) - len(df_five)} galaxies after applying the inclination cut.')
        
    #if magnitude colors are in the list of features, then we have to apply
    #a quality flag here too. This amount to just dropping the non-finite/unphysical/fake news values
    if params.colors:
        df_five = trim_colors(df_five)
    
    #and lastly...
    
    #if user specified VFIDs to exclude in init_parameters.py, this is the time to create the flag
    if params.EXCLUDE_LIST is not None:
        exclude_flag = [VFID.decode('utf-8') not in params.EXCLUDE_LIST for VFID in df_five['VFID']]
        df_five = df_five[exclude_flag]
        print(f'ALERT! Removed {np.sum(~np.asarray(exclude_flag))} galaxies after excluding VFIDs from init_parameters.py.')
    
    message=f'Removed {ngal_before - len(df_five)}/{ngal_before}  galaxies in total. This leaves {len(df_five)} galaxies. Wow.'        
    
    print('#'*len(message))
    print(message)
    print('#'*len(message))
    
    #return the 'cleansed' dataframe
    return df_five

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
    Standardize data features such that each column has a median of 0 
    and is scaled by the interquartile range (robust to outliers).
    output: edited pandas dataframe with input columns standardized
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
            
            #if features are the magnitude colors, be sure to exclude invalid values
            if feature in ['NUV_r', 'W1_W3']:
                print('Calculating medians -- removing invalid entries for NUV-r and W1-W3...')
                df_cluster = trim_colors(df_cluster.copy(), print_=False)
            
            if feature == 'Size Ratio':
                df_cluster = trim_ratios(df_cluster.copy())
            
            #isolate the feature from the cluster_id data
            arr = df_cluster[feature].values
            
            #calculate the median and lower+upper bootstrap confidence intervals
            med = np.median(arr)
            low, high = get_bootstrap_confint(arr, nboot=5000)

            #store median + error in the row set for that feature cluster
            row[feature] = med
            row[feature+"_err_low"] = med - low
            row[feature+"_err_high"] = high - med

        #add the row...
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)