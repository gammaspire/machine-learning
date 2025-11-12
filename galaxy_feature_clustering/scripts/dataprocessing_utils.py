import numpy as np

###############################
# CONVERTING ARCSEC TO PIXELS #
###############################

def px_to_arcsec(band, data, params=None):
    '''
    AIM: convert GALFIT effective radii from pixels to arcseconds
    * band must be a wavelength band (str) that is featured in PSCALE (grz, W1-4)
    * if data is float or list type, the output will be a numpy array
    * params
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