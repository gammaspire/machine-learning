##############################
##############################
##############################

##############################
## bands and GALFIT columns ##
##############################

#z-band table is entirely empty. all zeros and False bools. as such, the band is excluded here.
BANDS=['g','r','W1-fixBA','W2','W3-fixBA','W4']
COLUMNS=['CXC','CRE','CN','CNumerical_Error']
PSCALE={'g':0.262,'r':0.262,'z':0.262,
        'W1':2.75,'W1-fixBA':2.75,'W2':2.75,'W3':2.75,'W3-fixBA':2.75,'W4':2.75}  #from mucho-galfit code


###############################
####   DBSCAN PARAMETERS   ####
###############################

#define eps and min_samples. set to None for the script to optimize these parameters via Grid-Search and the
#silhouette method.
EPS=None
MIN_SAMPLES=None


###############################
####   KMEANS PARAMETERS   ####
###############################

#The desired k limit for clipping outliers from the data distribution via interquartile ranges. 
#Set IQRCLIP=None to include the full range of parameters.
#For reference:
    #IQRCLIP=1.5 has a Gaussian equivalent of ~3sigma
    #IQRCLIP=2.5 has a Gaussian equivalent of ~4sigma
#######################
# ~How does it work?~ #
#######################
    #IQR is the spread of the middle 50% of the data
        #the "typical" data.
    #Q1 is the bound below which 25% of the data falls
    #Q3 is the bound above which 25% of the data falls and below which 75% of the data falls
    #for IQR clipping: lower bound is (Q1 - IQRCLIP * IQR), upper bound is (Q3 + IQRCLIP * IQR)
        #say I set IQRCLIP = 1.5 (standard Tukey method)
        #take original Q1 and Q3 and *move* them outward such that the new IQR is 1.5x its original size
        #any points not in this bloated IQR are outliers. goodbye outliers.
        
IQRCLIP=1.5

#number of clusters to use for kmeans
#if you want the code to optimize k using the silhouette method, set K=None
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
X='CRE_W1'
Y='CRE_W3'


##############################################
###   PLOTTING PARAMETERS -- DBSCAN ONLY   ###
##############################################

UMAP_FOR_PLOTTING=True



##########################################
###   PLOTTING PARAMETERS -- GENERAL   ###
##########################################

#set to True for the script to generate a 2D projection of the feature clusters!
#can dictate the axes with X and Y above, assuming PCA_FOR_PLOTTING=False
PLOT_CLUSTERS=True

#set to True for the script to generate a corner plot of all feature clusters in a 
#physically meaningful space (i.e., feature vs. feature)
PLOT_CORNER=True

#set to True for script to plot the fraction of galaxies in one of five VFS environments from Castignani+2022 (pure field, filament, poor group, rich group, cluster). these fractions are split up into however many feature clusters the user defines.
    #e.g., feature cluster 0 will be divided into five environments, feature cluster 1 will be divided into five environments, etc.
    #note that the environments, save for the pure field, are not entirely decoupled from one another. one galaxies could belong to multiple environments.
PLOT_ENV_FRACTION=True


###############################
###############################
###############################

#using to define parameters; do NOT type beyond this line!
#this is known as a DATACLASS! it's singular purpose is to be a container for variables :-)
class Params():
    def __init__(self):
        self.BANDS = BANDS
        self.COLUMNS = COLUMNS
        self.PSCALE = PSCALE
        self.EPS = EPS
        self.MIN_SAMPLES = MIN_SAMPLES
        self.IQRCLIP = IQRCLIP
        self.K = K 
        self.PLOT_SILHOUETTES = PLOT_SILHOUETTES
        self.PLOT_CORNER = PLOT_CORNER
        self.PLOT_CLUSTERS = PLOT_CLUSTERS
        self.PCA_FOR_PLOTTING = PCA_FOR_PLOTTING
        self.UMAP_FOR_PLOTTING = UMAP_FOR_PLOTTING
        self.PLOT_PCA_COMPONENTS = PLOT_PCA_COMPONENTS
        self.PLOT_ENV_FRACTION = PLOT_ENV_FRACTION
        self.X = X
        self.Y = Y