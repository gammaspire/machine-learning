'''
For all of your numeric and statistic needs. :-)
'''

import numpy as np

###################################################
# MODIFIED BOOTSTRAP FUNCTION, COURTESY OF RFINN  #
###################################################

def get_bootstrap_confint(d, bootfunc=np.median, nboot=2000, seed=42):
    
    from astropy.stats import bootstrap
    
    '''
    AIM: Calculate (lower, upper) bootstrap 68% confidence intervals for any 
         statistic bootfunc applied to data d.
    
    ASTROPY.STATS.BOOTSTRAP
        -create nboot resamplings of the data and calculate the bootfunc of each resample.
        -will return the e.g. median for each of the nboot resamples
    '''
    
    if seed is not None:
        np.random.seed(seed)
    
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
    k_clip=2.5 has a Gaussian equivalent of roughly 4-sigma
    ''' 
    df = df.copy()
    
    # work in log10-space for strictly positive structural parameters
    df_log = df[features].copy()
    df_log = np.log10(df_log)
    
    Q1 = df_log.quantile(0.25)   #find 25% quartile of data distributions
    Q3 = df_log.quantile(0.75)   #find 75% quartile of data distributions
    
    #find the range of data values between these two bounds
    IQR = Q3 - Q1

    #generate mask --> data must not be beyond some multiple of the IQR width
    #the Q1-... and Q3+... just expand the endpoints of IQR outward such that the "new" IQR
    #is k-times the original IQR size.
    outlier_mask = ((df_log < (Q1 - k_clip * IQR)) | (df_log > (Q3 + k_clip * IQR))).any(axis=1)
        
    excluded = df.loc[outlier_mask]

    print(f"IQR clipping (k_clip={k_clip}): "
          f"Removing {outlier_mask.sum()}/{len(df)} galaxies "
          f"({outlier_mask.mean():.1%})")

    print("Excluded VFIDs:")
    print(excluded['VFID'].values)
    import csv

    with open('output.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(excluded['VFID'].values)
    
    return df.loc[~outlier_mask]


##########################################################
# CALCULATE BINOMIAL UNCERTAINTIES FOR ENVIRONMENT PLOT  #
##########################################################

def binomial_uncertainty(N_subset, N_total):
    '''
    AIM: Calculate the binomial uncertainty for the fraction of a population that an extracted subset represents.
         For example, the uncertainty on (# field galaxies)/(# total galaxies).
    '''
    f = N_subset / N_total
    unc = np.sqrt((f * (1 - f)) / N_total)
    
    return unc


########################################
# GET sSFR>-11.5 FLAG (FROM SALIM+2018 #
########################################

def ssfr_flag(mstar_array, sfr_array):
    '''
    AIM: return salim+2018 log(sSFR)>-11.5 flag, row-matched to input logmstar and logsfr arrays.
    '''
    
    logsSFR = sfr_array - mstar_array
    salim_flag = (logsSFR > -11.5)
    
    return salim_flag


#########################################
# GET SFRvMSTAR MAIN SEQUENCE EQUATION  #
#########################################

def get_ms_line(mstar_array, sfr_array):
    '''
    AIM: get the slope (m) and y-intercept (b) for the linear fit to SFR vs. Mstar data
    Note that I apply a log(sSFR) > -11.5 cut first, "as galaxies below this limit are those 
    whose UV and IR emission are likely dominated by sources not associated with star formation 
    (Salim+2018)" -- Conger+2025
    '''
    
    salim_flag = ssfr_flag(mstar_array, sfr_array)
    
    m, b = np.polyfit(mstar_array[salim_flag], sfr_array[salim_flag], deg=1)
        
    return m, b


########################################################
# GET SFRvMSTAR MAIN SEQUENCE PERPENDICULAR DISTANCES  #
########################################################

def get_ms_distance(mstar_array, sfr_array, m, b):
    '''
    AIM: calculate the perpendicular distance (in SFR vs. Mstar space) between array elements and the linear fit.
    
    Distance = (|A*x1 + B*y1 + C|) / (sqrt(A**2 + B**2))
    This originates from the standard form of a line: Ax + By = C (contrast with y = mx + b)
        * mapping onto y=mx+b:
            * Ax + By - C = y - mx - b
            * --> [A = -m, B = 1, C = b]
    
    OUTPUT: array of perpendicular distances from point to main sequence
    '''
    x1 = mstar
    y1 = sfr
    A = (-1)*(m)
    B = 1
    C = b
    
    distance_numerator = np.abs(A*x1 + B*y1 + C)
    distance_denominator = np.sqrt(A**2 + B**2)
    ms_distance = distance_numerator / distance_denominator
    
    return ms_distance
        
    
##########################################
# GET log(SFR) OFFSET FROM MAIN SEQUENCE #
##########################################

def get_delta_logsfr(mstar_array, sfr_array, m=0.799, b=-8.556, logsfr_floor=None):
    '''
    AIM: calculate /\log(SFR) array for the input galaxy data. Treat log(Mstar) as fixed.
    * Default m, b are from fits to the original data
    '''

    #first calculate the predicted logSFR at the given logMstar value
    logSFR_MS = m * mstar_array + b
    
    #collapse log(SFR)<-3 to -3
    if logsfr_floor is not None:
        sfr_array[sfr_array<logsfr_floor] = logsfr_floor
    
    #/\log(SFR) = logSFR_data - logSFR_MS
    delta_sfr = sfr_array - logSFR_MS
    
    return delta_sfr


#######################################################
# HELPER FUNCTION FOR plotting_utils.py/plot_pop_frac #
#######################################################

def mod_pop_fractions(df, fc=0, sup_to_pas=False, pas_to_sup=False):
    '''
    AIM: for a given fc, create modified suppressed_pop and passive_pop bool columns and calculate pop fractions.
        * due to logSFR=-3 floor, many galaxies moved from passive to suppressed regime
        * want to create uncertainties that show what happens when all passive open shape
          galaxies in an FC are suppressed, or all suppressesed open shape galaxies are passive
        * logSFRs for these galaxies are NOISY -- want to reflect this with uncertainties!
    
    ** open shape galaxies == galaxies with logSFR<-3.
    
    OUTPUT:
        * If sup_to_pas, will take all open shape galaxies in suppressed regime and move to passive
            * output is two FLOATS
                * modified passive fraction (closed passive + open passive + open suppressed)
                * modified suppressed fraction (closed suppressed)
        * If pas_to_sup, will take all open shape galaxies in passive regime and move to suppressed
            * output is two FLOATS
                * modified passive fraction (closed passive)
                * modified suppressed fraction (open passive + open suppressed + closed suppressed)
    '''
    
    if 'suppressed_pop' not in df.columns:
        print('suppressed_pop not a df column! exiting.')
        return
    
    if sup_to_pas == pas_to_sup:
        print('either choose sup_to_pas or pas_to_sup -- do not choose both or neither!')
        return
    
    #isolate galaxies in the given FC
    fc_galaxies = df[df['Feature Class']==int(fc)]
    
    #isolate logSFR. if logSFR=-3 floor already set, then cannot create modified bool columns (i.e., no open shapes!)
    sfr = fc_galaxies['logsfr']
    if np.sum(sfr < -3) == 0:
        print('logsfr values set to -3 floor! no modifications can be created. exiting.')
        return
    
    #isolate galaxies with logSFR < -3, and galaxies with logSFR >= -3
    lowsfr_galaxies = fc_galaxies[sfr < -3]
    highsfr_galaxies = fc_galaxies[sfr >= -3]
    
    #create the subpopulations of open shapes, closed shapes, all shapes.
    N_passive_closed = np.sum(highsfr_galaxies['passive_pop'])
    N_passive_open = np.sum(lowsfr_galaxies['passive_pop'])
    N_passive_all = N_passive_open + N_passive_closed
    
    N_suppressed_closed = np.sum(highsfr_galaxies['suppressed_pop'])
    N_suppressed_open = np.sum(lowsfr_galaxies['suppressed_pop'])
    N_suppressed_all = N_suppressed_open + N_suppressed_closed
    
    #and of course, the total number of galaxies in the FC
    N_all = len(fc_galaxies)
    
    #if moving all open suppressed galaxies to passive
    if sup_to_pas:
        mod_suppressed = (N_suppressed_closed) / N_all
        mod_passive = (N_passive_all + N_suppressed_open) / N_all
    
    if pas_to_sup:
        mod_suppressed = (N_suppressed_all + N_passive_open) / N_all
        mod_passive = (N_passive_closed) / N_all
    
    return mod_suppressed, mod_passive
    