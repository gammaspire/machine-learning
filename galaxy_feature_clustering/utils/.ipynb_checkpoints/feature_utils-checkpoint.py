################################################
# PULL THE NAMES OF THE GALFIT FEATURE COLUMNS #
################################################

def get_feature_names(params, colors=False):
    '''
    AIM: return list of all feature names for specified parameters in, surprise, init_parameters.py
    '''
    #define Re, Sersic index feature columns
    re_cols = [f'CRE_{band}' for band in params.BANDS_TO_CLUSTER]
    nser_cols = [f'CN_{band}' for band in params.BANDS_TO_CLUSTER]
    
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
                    'W1_W3': '[W1-W3]',
                    'delta_logsfr': '$\Delta$logSFR'}
    
    return feature_dict


#####################################################
# HELPER FUNCTION FOR APPLYING THE LABEL DICTIONARY #
#####################################################

def get_feature_label(colname, label_dict):
    """
    Strip the '_unscaled' suffix from columnnames before matching to the feature dictionary
    """
    base = colname.replace('_unscaled', '')
    return label_dict.get(base, colname)


####################################
# DICTIONARY FOR ENVIRONMENT FLAGS #
####################################

def make_env_defs(feature_data, main_only=True):
    '''
    AIM: create (environment name : bool flags) dictionary to faciliate feature group plotting!
    
    feature_data:
        pandas dataframe with environment flags!
    
    main_only:
        If True, restricts to the five primary environments.
    '''
    if 'pure_field' not in feature_data.columns:
        print('Uh oh. Make sure you have the environment flags in the input dataframe!')
        return
    
    if main_only:
        env_defs = {'Cluster':     feature_data['cluster_member'],
                    'Rich Group':  feature_data['rich_group_memb'],
                    'Poor Group':  feature_data['poor_group_memb'],
                    'Filament':    feature_data['filament_member'],
                    'Pure Field':  feature_data['pure_field']}
        return env_defs

    env_defs = {#'Pure Cluster':              (feature_data['cluster_member']) & \
                #                             (~feature_data['filament_member']),

               'All Cluster':                (feature_data['cluster_member']),

               #'Filament\n&\nCluster':       (feature_data['cluster_member']) & \
               #                              (feature_data['filament_member']) & \
               #                              (~feature_data['rich_group_memb']) & \
               #                              (~feature_data['poor_group_memb']),

               'Filament\n&\nRich Group':    (feature_data['rich_group_memb']) & \
                                             (feature_data['filament_member']),

               'Pure Rich \n Group':         (feature_data['rich_group_memb']) & \
                                             (~feature_data['filament_member']),

               #'All Filament\n(PG+RG+CLUS)': (feature_data['filament_member']),

               'Filament\n&\nPoor Group':    (feature_data['filament_member']) & \
                                             (feature_data['poor_group_memb']) & \
                                             (~feature_data['rich_group_memb']),

               'Pure Poor \n Group':         (feature_data['poor_group_memb']) & \
                                             (~feature_data['filament_member']),

               'Pure Filament':              (feature_data['filament_member']) & \
                                             (~feature_data['cluster_member']) & \
                                             (~feature_data['poor_group_memb']) & \
                                             (~feature_data['rich_group_memb']),

               'Pure Field':                 (feature_data['pure_field'])}

    return env_defs
    