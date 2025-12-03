'''
The statistics stuff. :-)
'''

import numpy as np


#bootstrap function courtesy of Rose Finn; 
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


def binomial_uncertainty(N_subset, N_total):
    '''
    AIM: calculate the binomial uncertainty for the fraction of a population that an extracted subset represents.
         for example, the uncertainty on (# field galaxies)/(# total galaxies).
    '''
    f = N_subset / N_total
    unc = np.sqrt((f * (1 - f)) / N_total)
    
    return unc


def create_median_table(feature_data, features):
    '''
    AIM: save a summary of the feature medians + bootstrap uncertainties in every feature cluster
    '''
    
    #get list of unscaled feature columns, including the size ratios
    features_unscaled = [feature+'_unscaled' for feature in features] + ['Size Ratio']

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