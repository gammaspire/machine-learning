'''
AIM: create the dataframe of galaxy features!
'''

from astropy.table import Table
import numpy as np
import pandas as pd

from galfit_parameters import Params
params = Params()

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
    return phot, ext


#yes, I read the environment table twice. I like organization. cope.
def get_vcosmic_column():
    env = Table.read('data/vf_v2_environment.fits')
    return env['Vcosmic']


def get_env_columns():
    '''
    *set up the environment flags...
    *very basic setup -- no environment, save pure field, is entirely decoupled from the others.
    '''
    env = Table.read('data/vf_v2_environment.fits')['cluster_member', 'rich_group_memb',
                                                    'poor_group_memb', 'filament_member', 'pure_field']
    return env.to_pandas()


def get_stellar_columns():
    cigale = Table.read('data/cigale_vf_metallicity.fits')
    
    mstar = np.log10(cigale['bayes.stellar.m_star'])
    sfr = np.log10(cigale['bayes.sfh.sfr'])
    
    return mstar, sfr


################################
# INITIALIZE THE FEATURE TABLE #
################################

def make_galfit_table(colors=False):
    Read GALFIT grz, W1-4 output tables
        * If colors=True, will include NUV-r, W1-W3 colors
        * If 
    Convert from astropy Table to pandas df
    Output --> dataframe with grz, W1-3 Re, nser, CXC, CNumerical_Error columns
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
        from conversion_utils import get_photometric_colors
        phot, ext = read_phot_tables()
        NUV_r, W1_W3 = get_photometric_colors(phot, ext)
        data_table.add_columns([NUV_r,W1_W3], names=['NUV_r','W1_W3'])
    
    data_table = data_table.to_pandas()

    #append environment columns
    envflags = get_env_columns()
    data_table = pd.concat([data_table.copy(), envflags.copy()],axis=1)

    return data_table