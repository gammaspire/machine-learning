from galfit_parameters import Params
params = Params()


#########################################
# PULL THE NAMES OF THE GALAXY FEATURES #
#########################################

def get_feature_names(colors=False, flux=False):
    '''
    AIM: return list of all feature names for specified parameters in, surprise, galfit_parameters.py
    '''
    
    #define Re, Sersic index feature columns
    re_cols = [f'CRE_{band}' for band in params.BANDS]
    nser_cols = [f'CN_{band}' for band in params.BANDS]
    
    #combine
    features = re_cols + nser_cols
    
    #use averages of g&r, W1&W2 effective radii
    if 'CRE_r' in features and 'CRE_g' in features:
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


############################################################
# CALCULATING THE AVERAGE OF CERTAIN FEATURES (IF TOGGLED) #
############################################################

def add_average_re(data_table):
    '''
    AIM: append average g & r, W1 & W2 effective radii if the columns exist.
    '''
    if 'CRE_r' and 'CRE_g' in data_table.columns:
        data_table['AVG_RE_gr'] = (data_table['CRE_g'] + data_table['CRE_r'])/2
    if 'CRE_W1-fixBA' and 'CRE_W2' in data_table.columns:
        data_table['AVG_RE_W1W2'] = (data_table['CRE_W1-fixBA'] + data_table['CRE_W2'])/2
    
    return data_table