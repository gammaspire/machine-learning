##############################
##############################
##############################


#################################
#    FEATURE TABLE SAVE/LOAD    #
#################################
'''
if the feature df is already available, set the DF_PATH and LOADTABLE=True
TABLE REQUIREMENTS:
    * trimmed to remove unreliable fits, nser>6, etc.
    * effective radii converted from pixels to kpc
    * IQR-clipped rows
    * standardized values for every feature (possibly including color, depending on user input) 
    * the _unscaled variants of every feature
    * correct column names!
* if the table is in data/ of the root directory, only need to specify 'data/filename.csv'
* set SAVETABLE=True to save the feature_data table after creation! will default to DF_PATH
* NOTE I: IF SAVETABLE=TRUE, YOU MUST SET LOADTABLE=FALSE!
* NOTE II: there are two df paths because kmeans may use IQR-clipping. This fundamentally changes the density space of the data distribution, and thus non-trivially affects HDBSCAN.
'''
LOADTABLE=False
SAVETABLE=True
KMEANS_DF_PATH='data/kmeans_feature_data.csv'
HDBSCAN_DF_PATH='data/hdb_feature_data.csv'


#############################################
## logSFR and logMstar completeness limits ##
#############################################
'''
* define logsfr, logmstar completeness limits for the sample here
* see mass_sfr_completeness.ipynb for details.
* set either to None if you do not want that completeness limit applied.
'''
#LOGSFR_LIM=-3.065   #limit for all galaxies, using W3 SNR
#LOGSFR_LIM=-3.309   #limit for Vcosmic<2000. sample, using W3 SNR
LOGSFR_LIM=None
LOGMSTAR_LIM=8.15


#########################################
## Inclination limit (motivated by LCS ##
#########################################
'''
* Uses Local Cluster Survey inclination cut
* B/A > 0.25
* Set to True if wanted; set to False to disable 
'''
AXISRATIO_LIM=True


##############################
## bands and GALFIT columns ##
##############################
'''
DEFINE THE BANDS AND COLUMNS IN THE GALFIT DATA TABLE
z-band table is entirely empty. all zeros and False bools. as such, the band is excluded here. I also exclude W4 due to its low SNR.
Combined, COLUMNS & BANDS_TO_CLUSTER comprise the columns used in the clustering algorithm.
BANDS are simply all bands that I want in the dataframe.
'''
BANDS=['g','r','W1-fixBA','W3-fixBA']  #ALL (GALFIT) bands
BANDS_TO_CLUSTER=['g','W1-fixBA']      #(GALFIT) bands that are considered for the clustering algorithm

COLUMNS=['CRE','CN','CXC','CNumerical_Error','CRE_ERR', 'CN_ERR'] #NOTE THESE FEATURE LABELS ARE LATER CHANGED
                                                        #CRE --> Effective Radius
                                                        #CN --> Sersic Index
                                                        #CXC --> central x pixel; used to diagnose whether GALFIT ran
                                                        #CNumerical Error --> a flag indicating the robustness of the GALFIT model
                                                        #CRE_ERR --> model error for CRE
                                                        #CN_ERR --> model error for CN
#pixel to arcsec conversion scale
PSCALE={'g':0.262,'r':0.262,'z':0.262,
        'W1':2.75,'W1-fixBA':2.75,
        'W2':2.75,
        'W3':2.75,'W3-fixBA':2.75,
        'W4':2.75}  #from mucho-galfit code


################################
###    HDBSCAN PARAMETERS    ###
################################
'''
define the HDBSCAN parameters!
'''
MIN_SAMPLES=3
MIN_CLUSTER_SIZE=80
METRIC='canberra'
SELECTION_METHOD='eom'

#optimize parameters flag! script will optimize HDBSCAN parameters via grid-search and a modified elbow method
OPTIMIZE_HDB_PARAMS=False


###############################
####   KMEANS PARAMETERS   ####
###############################
'''
The desired k limit for clipping outliers from the data distribution via interquartile ranges. 
* For reference:
    * IQRCLIP=1.5 has a Gaussian equivalent of ~3sigma
    * IQRCLIP=2.5 has a Gaussian equivalent of ~4sigma
#######################
# ~How does it work?~ #
#######################
    * IQR is the spread of the middle 50% of the data
        #the "typical" data.
    * Q1 is the bound below which 25% of the data falls
    * Q3 is the bound above which 25% of the data falls and below which 75% of the data falls
    * for IQR clipping: lower bound is (Q1 - IQRCLIP * IQR), upper bound is (Q3 + IQRCLIP * IQR)
        #say I set IQRCLIP = 1.5 (standard Tukey method)
        #take original Q1 and Q3 and *move* them outward such that the new IQR is 1.5x its original size
        #any points not in this bloated IQR are outliers. goodbye outliers.
* set to None to include full range of parameters.
'''
IQRCLIP=None  #None   #2

'''
* List of VFIDs to exclude from the sample! This list should specifically encompass galaxies with unreliable GALFIT models that somehow snuck past other quality checks (e.g., numerical error flag).
* Set to None if no galaxies excluded.
'''
EXCLUDE_LIST=['VFID0293','VFID0455','VFID0800','VFID1435','VFID1580',
              'VFID1721','VFID2090','VFID2252','VFID2318','VFID2399',
              'VFID2567','VFID2977','VFID2996','VFID3127','VFID3155',
              'VFID3649','VFID4056','VFID4064','VFID4086','VFID4186',
              'VFID4196','VFID4390','VFID4587','VFID5056','VFID5204',
              'VFID5289','VFID5234','VFID5289','VFID5515','VFID5747',
              'VFID6042']

'''
#number of clusters to use for kmeans
#if you want the code to optimize k using the silhouette method, set K=None
'''
K=3


##############################################
###   PLOTTING PARAMETERS -- KMEANS ONLY   ###
##############################################

#set to True to output a plot of the silhouette values vs. number of clusters
PLOT_SILHOUETTES=True

#set to True feature reduction for plotting the kmeans clusters in 2D space
#if False, will use X and Y
PCA_FOR_PLOTTING=True

#plot the feature vector components of each PCA!
#will be ignored if PCA_FOR_PLOTTING=False
PLOT_PCA_COMPONENTS=False

#If PCA_FOR_PLOTTING is False, choose X and Y columns below for plotting feature clusters in 2D space
X='CN_W1-fixBA_unscaled'
Y='CN_W3-fixBA_unscaled'


###############################################
##    PLOTTING PARAMETERS -- HDBSCAN ONLY    ##
###############################################

UMAP_FOR_PLOTTING=True


##########################################
###   PLOTTING PARAMETERS -- GENERAL   ###
##########################################

#set to True for the script to generate a 2D projection of the feature clusters!
#can dictate the axes with X and Y above, assuming PCA_FOR_PLOTTING=False
PLOT_CLUSTERS=True

#set to True for the script to generate a corner plot of all feature clusters in a 
#physically meaningful space (i.e., feature vs. feature)
PLOT_CORNER=False

#set to True for the script to create subplots of the median galaxy features (including Size Ratio, NUV-r, and W1-W3)
#COMPANION TO LAYOUT_DICT.
PLOT_MEDIANS=True

'''
The subplot coordinate/columnname dictionary to help organize the figure layout. If None, will default to using W1, W3, g-band Re+nser, as well as Size Ratio, NUV-r, and W1-W3.
* Must be a python dictionary. for example, if you only want one subplot with g-band effective radius:
    LAYOUT_DICT = {(0, 0): 'CRE_g_unscaled'}
* COMPANION TO PLOT_MEDIANS
'''
LAYOUT_DICT=   {(0, 0): 'CRE_g_unscaled',
                (0, 1): 'CRE_W1-fixBA_unscaled',
                (1, 0): 'CN_g_unscaled',
                (1, 1): 'CN_W1-fixBA_unscaled',
                (2, 0): 'Size Ratio',
                (2, 1): 'NUV_r',
                (3, 0): 'W1_W3'}

#LAYOUT_DICT=   {(0, 0): 'CRE_g_unscaled',
#                (0, 1): 'CRE_W1-fixBA_unscaled',
#                (0, 2): 'CRE_W3-fixBA_unscaled',
#                (1, 0): 'CN_g_unscaled',
#                (1, 1): 'CN_W1-fixBA_unscaled',
#                (1, 2): 'CN_W3-fixBA_unscaled',
#                (2, 0): 'Size Ratio',
#                (2, 1): 'NUV_r',
#                (2, 2): 'W1_W3'}


#set to True for the script to create raincloud plots showing the distribution of the galaxy features
#for each feature group. 
#COMPANION TO FEATURE_LIST
PLOT_RAINCLOUDS=False

#the feature list dictionary to dictate what features are included in the raincloud plots above. if None, will default to using W1, W3, g-band Re+nser, as well as Size Ratio, NUV-r, and W1-W3.
#must be a python list, with elements written as strings and reflecting actual column names in the input data table.
#COMPANION TO PLOT_RAINCLOUDS
FEATURE_LIST=None

#set to True for script to plot the fraction of galaxies in one of five VFS environments from Castignani+2022 (pure field, filament, poor group, rich group, cluster). these fractions are split up into however many feature clusters the user defines.
    #e.g., feature cluster 0 will be divided into five environments, feature cluster 1 will be divided into five environments, etc.
    #note that the environments, save for the pure field, are not entirely decoupled from one another. one galaxies could belong to multiple environments.
PLOT_ENV_FRACTION=True

#plot each feature cluster on an SFR vs. Mstar plot, with histogram sub-axes to show the scatter distribution
PLOT_SFRMSTAR=True


###############################
###############################
###############################

#using to define parameters; do NOT type beyond this line!
#this is known as a DATACLASS! it's singular purpose is to be a container for variables :-)
class Params():
    def __init__(self):
        
        self.BANDS = BANDS
        self.BANDS_TO_CLUSTER = BANDS_TO_CLUSTER
        self.COLUMNS = COLUMNS
        
        self.PSCALE = PSCALE
        self.LOGSFR_LIM = LOGSFR_LIM
        self.LOGMSTAR_LIM = LOGMSTAR_LIM
        self.AXISRATIO_LIM = AXISRATIO_LIM
        
        self.LOADTABLE = LOADTABLE
        self.KMEANS_DF_PATH = KMEANS_DF_PATH
        self.HDBSCAN_DF_PATH = HDBSCAN_DF_PATH
        self.SAVETABLE = SAVETABLE
        
        self.K = K
        self.MIN_CLUSTER_SIZE = MIN_CLUSTER_SIZE
        self.MIN_SAMPLES = MIN_SAMPLES
        self.METRIC = METRIC
        self.SELECTION_METHOD = SELECTION_METHOD
        self.OPTIMIZE_HDB_PARAMS = OPTIMIZE_HDB_PARAMS
        
        self.IQRCLIP = IQRCLIP
        self.EXCLUDE_LIST = EXCLUDE_LIST
        
        self.PLOT_SILHOUETTES = PLOT_SILHOUETTES
        self.PLOT_CORNER = PLOT_CORNER
        self.PLOT_CLUSTERS = PLOT_CLUSTERS
        
        self.PLOT_MEDIANS = PLOT_MEDIANS
        self.LAYOUT_DICT = LAYOUT_DICT
        
        self.PLOT_RAINCLOUDS = PLOT_RAINCLOUDS
        self.FEATURE_LIST = FEATURE_LIST
        
        self.PCA_FOR_PLOTTING = PCA_FOR_PLOTTING
        self.UMAP_FOR_PLOTTING = UMAP_FOR_PLOTTING
        self.PLOT_PCA_COMPONENTS = PLOT_PCA_COMPONENTS
        self.PLOT_ENV_FRACTION = PLOT_ENV_FRACTION
        self.PLOT_SFRMSTAR = PLOT_SFRMSTAR
        
        self.X = X
        self.Y = Y