'''
For all of your numeric and statistic needs. :-)
'''

import numpy as np

###################################################
# MODIFIED BOOTSTRAP FUNCTION, COURTESY OF RFINN  #
###################################################

def get_bootstrap_confint(d,bootfunc=np.median,nboot=2000):
    
    from astropy.stats import bootstrap
    
    '''
    AIM: Calculate (lower, upper) bootstrap 68% confidence interval for any 
         statistic bootfunc applied to data d.
    
    ASTROPY.STATS.BOOTSTRAP
        -create nboot resamplings of the data and calculate the bootfunc of each resample.
        -will return the e.g. median for each of the nboot resamples
    '''
    bootsamp = bootstrap(d,bootfunc=bootfunc,bootnum=nboot)

    # sort the bootstrap sampled medians
    bootsamp.sort()

    # get indices corresponding to 68% confidence interval
    ilower = int(0.16*nboot)
    iupper = int(0.84*nboot)

    # return the e.g. median at the 68% confidence interval
    # need to subtract from median to get the actual errorbars
    # like err_lower = actual_median - bootsamp[ilower]
    # and err_upper = bootsamp[iupper] - actual_median
    return bootsamp[ilower],bootsamp[iupper]


##############################################################
# PERFORM IQR CLIPPING TO ELIMINATE OUTLIERS IN FEATURE SETS #
##############################################################

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


def binomial_uncertainty(N_subset, N_total):
    '''
    AIM: Calculate the binomial uncertainty for the fraction of a population that an extracted subset represents.
         For example, the uncertainty on (# field galaxies)/(# total galaxies).
    '''
    f = N_subset / N_total
    unc = np.sqrt((f * (1 - f)) / N_total)
    
    return unc