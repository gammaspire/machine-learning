from galfit_parameters import Params
params = Params()

################################################
# PULL THE NAMES OF THE GALFIT FEATURE COLUMNS #
################################################

def get_feature_names(colors=False):
    '''
    AIM: return list of all feature names for specified parameters in, surprise, galfit_parameters.py
    '''
    #define Re, Sersic index feature columns
    re_cols = [f'CRE_{band}' for band in params.BANDS]
    nser_cols = [f'CN_{band}' for band in params.BANDS]
    
    #combine
    features = re_cols + nser_cols
    
    #use averages of g&r, W1&W2 effective radii
    #if 'CRE_r' in features and 'CRE_g' in features:
    #    print('Using average g and r effective radius!')
    #    features = [f for f in features if f not in ['CRE_r', 'CRE_g']] + ['AVG_RE_gr']
    
    #if 'CRE_W1-fixBA' and 'CRE_W2' in features:
    #    print('Using average W1 and W2 effective radius!')
    #    features = [f for f in features if f not in ['CRE_W1-fixBA', 'CRE_W2']] + ['AVG_RE_W1W2']
    
    if colors:
        features += ['NUV_r','W1_W3']
        
    #and return
    return features


##################################################
# CREATE DICTIONARY FOR INTELLIGIBLE AXIS LABELS #
##################################################

def make_label_dictionary():
    '''
    Create label dictionary to better ensure that the plot labels with physical parameters are UNDERSTANDABLE.
    The list of possible features is written below. Add and subtract at will.
        * Remember that this list is meant to be all-encompassing! Set both arguments to True.
    '''
    possible_features = get_feature_names(colors=True)
    
    feature_dict = {'CRE_W1-fixBA': '[W1] Effective Radius',
                    'CRE_W2': '[W2] Effective Radius',
                    'CRE_W3-fixBA': '[W3] Effective Radius',
                    'CRE_W4': '[W4] Effective Radius',
                    'CRE_g': '[g] Effective Radius',
                    'CRE_r': '[r] Effective Radius',
                    'AVG_RE_gr': '[g+r] Average Effective Radius',
                    'AVG_RE_W1W2': '[W1+W2] Average Effective Radius',
                    'CN_W1': '[W1] Sersic Index',
                    'CN_W1-fixBA': '[W1] Sersic Index',
                    'CN_W2': '[W2] Sersic Index',
                    'CN_W3': '[W3] Sersic Index',
                    'CN_W3-fixBA': '[W3] Sersic Index',
                    'CN_W4': '[W4] Sersic Index',
                    'CN_g': '[g] Sersic Index',
                    'CN_r': '[r] Sersic index',
                    'NUV_r': '[NUV-r]',
                    'W1_W3': '[W1-W3]'}
    
    return feature_dict


###############################################
# HELPER FUNCTION FOR APPLYING THE DICTIONARY #
###############################################

def get_feature_label(colname, label_dict):
    """
    Strip the '_unscaled' suffix from columnnames before matching to the feature dictionary
    """
    base = colname.replace('_unscaled', '')
    return label_dict.get(base, colname)


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