#utility functions for trimming and standardizing...


import numpy as np
import pandas as pd
from astropy.table import Table
from galfit_parameters import Params
params = Params()


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


##############################################################
# PERFORM IQR CLIPPING TO ELIMINATE OUTLIERS IN FEATURE SETS #
##############################################################t

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