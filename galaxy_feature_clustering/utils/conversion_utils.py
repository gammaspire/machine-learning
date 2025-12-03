import numpy as np
from galfit_parameters import Params
params = Params()

###############################
# CONVERTING ARCSEC TO PIXELS #
###############################

def px_to_arcsec(band, data):
    '''
    AIM: convert GALFIT effective radii from pixels to arcseconds
    * band must be a wavelength band (str) that is featured in PSCALE (grz, W1-4)
    * if data is float or list type, the output will be a numpy array
    '''
    import sys
    
    if not params:
        print('You need a valid params argument in order to proceed!')
        sys.exit()
    
    if band not in params.PSCALE:
        print(f'{band} not found! Please select from grz, W1-4 only.')
        return
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    return np.asarray(data) * params.PSCALE[band]

##########################################
# CONVERSION FUNCTIONS FOR ARCSEC TO KPC #
##########################################

def arcsec_to_radians(arcsec_array):
    '''
    AIM: simply, convert array of arcseconds to radians
    OUTPUT: ...data array of radians
    '''
    
    #first convert arcseconds to degrees
    degree_array = arcsec_array / 3600.   #3600 arcseconds per degree
    
    #convert degrees to radians
    radian_array = degree_array * np.pi / 180.   #180/pi degrees per radian
    
    return radian_array

def radians_to_kpc(radians_array, vcosmic_data):
    '''
    AIM: convert galaxy effective radius from radians to kpc
    * Will use Vcosmic for now from VFS. 
    * Assume H0 = 74. km/s/Mpc
    INPUT:
    * Re array (arcsec)
    * Vcosmic array (km/s)
        * MUST BE ROW-MATCHED TO RE ARRAY!
    OUTPUT:
    * Row-matched array of kpc effective radii
    '''
    if len(radians_array) != len(vcosmic_data):
        import sys
        sys.exit('Re array and Vcosmic array are not the same length! exiting.')
    
    dist_mpc = vcosmic_data/74.
    
    #convert Mpc to kpc
    dist_kpc = dist_mpc*1.e3
    
    #that dist_kpc is the adjacent component of the right triangle.
    #re_data is the angle.
    #we need the opposite component of the right triangle.
    #and that is...radians_array.
    
    #small angle approximation applies here. no need for tan.
    #tan(theta) = re_kpc / dist_kpc
    re_data_kpc = dist_kpc * radians_array
    
    return re_data_kpc

#####################################
# DIRECTLY CONVERTING ARCSEC TO KPC #
#####################################

def arcsec_to_kpc(arcsec_array, vcosmic_data):
    radians_array = arcsec_to_radians(arcsec_array)
    re_data_kpc = radians_to_kpc(radians_array, vcosmic_data)
    return re_data_kpc

#############################################################
# CONVERTING NANOMAGGIES TO EXTINCTION-CORRECTED MAGNITUDES #
#############################################################

def nmaggies_to_mag(phot_array, extinction_array):
    '''
    AIM: convert column of fluxes (nanomaggies) to extinction-corrected AB magnitudes
    INPUT: 
        *row-matched photometry array (in nanomaggies) for a single wavelength band
        *row-matched extinction array (in magnitudes) for a single wavelength band
    OUTPUT:
        *row-matched, extinction corrected AB magnitude array
    '''
        
    #first convert nanomaggies to janskys
    phot_jy = phot_array * 3.631e-6   #conversion factor

    #convert to AB magnitudes
    mAB = (-2.5) * np.log10(phot_jy / 3631.)
    
    #now...apply extinction correction
    mAB_corr = mAB - extinction_array
    
    return mAB_corr


#####################################################
# CONVERTING PIXELS TO KPC FOR EFFECTIVE RADII DATA #
#####################################################
def get_kpc_columns(data_table):
    '''
    AIM: convert pixels to arcseconds, then arcseconds to kpc for every effective radius column.
    '''
    for band in params.BANDS:
        re_col = f'CRE_{band}'
        re_arcsec = px_to_arcsec(band, data_table[re_col])
        
        if 'Vcosmic' not in data_table.columns:
            print('Need Vcosmic column in order to proceed! Expect errors imminently...')
            return
        
        re_kpc = arcsec_to_kpc(re_arcsec, data_table['Vcosmic'])                
        data_table[re_col] = re_kpc
    
    return data_table


#####################################################
# CONVERTING FLUXES TO PHOTOMETRIC MAGNITUDE COLORS #
#####################################################

def get_photometric_colors(phot, ext): 
    from conversion_utils import nmaggies_to_mag
    
    band = ['NUV', 'R', 'W1', 'W4'] 
    
    #convert phot fluxes to extinction-corrected AB magnitudes 
    for i in range(4): #0, 1, 2, 3...NUV, R, W1, W4
        mAB_corr = nmaggies_to_mag(phot[f'FLUX_AP06_{band[i]}'], ext[f'A({band[i]})_SandF']) 
        phot[f'mAB_{band[i]}'] = mAB_corr 
        
    NUV_r = phot[f'mAB_NUV'] - phot['mAB_R'] 
    W1_W4 = phot[f'mAB_W1'] - phot['mAB_W4'] 

    return NUV_r, W1_W4