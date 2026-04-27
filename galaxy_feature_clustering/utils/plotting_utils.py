from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

from feature_utils import make_label_dictionary, get_feature_label, make_env_defs
from data_utils import get_bootstrap_confint, get_ms_line

#needed to apply certain quality cuts for magnitude colors and size ratios
from table_utils import trim_colors, trim_ratios  

from scipy.stats import ks_2samp
from itertools import combinations

#editing feature labels! global variable!
LABEL_DICT = make_label_dictionary()

#lastly, lastly...globally set the fontsize of tickmark labels
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)


######################################
######################################
# Defining a Consistent Plot Palette #
######################################
######################################

def marker_palette(feature_data):
    
    shapes = ['<', 's', '^', '*', 'D', 'v', 'X', '<', 'h', '>']
        
    try:
        clusters = feature_data['Feature Cluster'].unique()
        k = len(clusters)  #number of Feature Clusters
        noise_flag = (-1 in clusters)
        k = k-1 if noise_flag else k
    except:
        k = len(feature_data)  #also number of Feature Clusters --> for one row per Feature Cluster
                               #only needed if plotting medians
    
    if k == 3:
        cluster_colors = ['darkorange','seagreen','deeppink']
        edge_colors = ['orangered', 'green', 'crimson']
        
    elif k == 4:
        cluster_colors = ['darkorange','seagreen','deeppink','indigo']
        edge_colors = ['orangered', 'green', 'crimson', 'black']
    
    else:
        print(k, 'clusters')
        cluster_colors = sns.color_palette('husl', len(feature_data['Feature Cluster'].unique()))
        edge_colors = cluster_colors
    
    marker_shapes = [shapes[i % len(shapes)] for i in range(k)]
    
    if -1 in feature_data['Feature Cluster'].unique():
        cluster_colors.insert(0, 'lightgray')
        edge_colors.insert(0, 'darkgray')
        marker_shapes.insert(0, 'o')
    
    return cluster_colors, edge_colors, marker_shapes


#####################################
#####################################
# Plotting Silhouette Method Output #
#####################################
#####################################

def plot_silhouette(K, silhouettes):
    
    plt.figure(figsize=(7,5))
    plt.plot(K, silhouettes, 'o-', color='green')
    
    plt.xlabel('Number of Clusters (k)',fontsize=14)
    plt.ylabel('Silhouette Score',fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

############################################################
############################################################
# Visualizing the Feature Clusters in 2D PCA or UMAP Space #
############################################################
############################################################

def plot_clusters(feature_data, x=None, y=None, PCA=False, UMAP=False):
    
    #pull the colors...
    cluster_colors, _, cluster_shapes = marker_palette(feature_data)
    
    #sort the unique Feature Clusters numerically
    unique_clusters = sorted(feature_data['Feature Cluster'].unique())
    
    # -- 
    #NOTE: these maps make easier the use of sns scatterplots...
    # --
    
    #create custom label map! example -- {0: 'FC0 (Ngal)'}
    label_map= {c: f"FC{c} ({len(feature_data[feature_data['Feature Cluster']==c])})" for c in unique_clusters if c!=-1}
    color_map = {label_map[c]: cluster_colors[i] for i, c in enumerate(unique_clusters)}
    marker_map = {label_map[c]: cluster_shapes[i] for i, c in enumerate(unique_clusters)}
    
    #PCA and UMAP flag
    flag = (PCA | UMAP)
    
    if x is None and not flag:
        print('Unable to generate plot! Please either specify x, y columns, or set PCA=True or UMAP=True.')
        return
    
    x = 'Comp1' if flag else x
    y = 'Comp2' if flag else y
    
    
    print(label_map)
    
    if -1 not in feature_data['Feature Cluster'].unique():
        plt.figure(figsize=(8,6))
        
        ax = sns.scatterplot(data=feature_data, x=x, y=y, 
                             hue=feature_data['Feature Cluster'].map(label_map),palette=color_map, 
                             style=feature_data['Feature Cluster'].map(label_map),markers=marker_map,
                             s=100,alpha=0.5, edgecolor='w', linewidth=0.4)
    else:
        ax = sns.scatterplot(x=x, y=y, data=feature_data[feature_data['Feature Cluster'] == -1], alpha=0.1,
                            color='lightgray', edgecolor='w', linewidth=0.4, label='Noise')
        
        sns.scatterplot(x=x, y=y, data=feature_data[feature_data['Feature Cluster'] != -1], 
                        hue=feature_data['Feature Cluster'].map(label_map), palette=color_map, 
                        style=feature_data['Feature Cluster'].map(label_map), markers=marker_map,
                        alpha=0.7, edgecolor='w', linewidth=0.4,
                        ax=ax)
    
    plt.xlabel('Component One',fontsize=14)
    plt.ylabel('Component Two',fontsize=14)
    
    ax.grid(alpha=0.2)

    ax.legend(fontsize='large', title_fontsize='large', title=None)
    
    #plt.title(f'Structural Clusters in 2D Space')
    plt.tight_layout()
    plt.show()

    
##################################
##################################
# Plotting PCA Vector Components #
##################################
##################################

def plot_pca_components(feature_data, features, pca, cmap_name='tab20'):
    '''
    * Visualize PCA feature vectors for the first two components.
    * Features are sorted by total loading strength (or rather, the magnitude of their contribution).
    '''
    from matplotlib.cm import get_cmap
    
    n_features = len(features)
    components = pca.components_
    
    #calculate total contribution magnitude for each feature (will need later!)
    loading_strength = np.sqrt(components[0]**2 + components[1]**2)
    
    #sort features by loading strength (descending)
    sorted_indices = np.argsort(loading_strength)[::-1]
    features_sorted = [features[i] for i in sorted_indices]
    
    cmap = get_cmap(cmap_name)
    hatch_options = ['**', '||', '..', 'OO', 'xx', 'oo', 'OO', '..', '**']
    
    #cmap.N gives the number of colors available in the cmap
    #i % cmap.N --> for each i, calculates remainder of i/cmap.N.
        #if cmap.N=10, then i%cmap.N = 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,...
        #this means the cmap color will go cyclically
    colors = [cmap(i % cmap.N) for i in range(n_features)]  
    
    #do the same for hatch!
    hatch_ = [hatch_options[i % len(hatch_options)] for i in range(n_features)]
    
    plt.figure(figsize=(8, 6))

    for rank, i in enumerate(sorted_indices):
        x, y = components[0, i], components[1, i]
        plt.arrow(0, 0, x, y,
                head_width=0.06, head_length=0.07, linewidth=1.8, alpha=0.6,
                fc=colors[rank], ec='black', hatch=hatch_[rank],
                label=f"{get_feature_label(features[i], LABEL_DICT)} ({loading_strength[i]:.2f})")

    plt.xlabel("Component One",fontsize=14)
    plt.ylabel("Component Two",fontsize=14)
    #plt.title("PCA Vectors")
    plt.grid(alpha=0.2)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)

    plt.legend(loc='upper left', fontsize=12, markerscale=2, labelspacing=1.2, framealpha=0.3) #bbox_to_anchor=(1.05, 1), 
    
    plt.tight_layout()
    plt.show()


###################################################
###################################################
# Plotting Feature Cluster Environment Properties #
###################################################
###################################################
    
def plot_env_fraction(feature_data, main_only=False, envfrac=False, envcomp=False):
    '''
    INTERPRETATIONS:
        if envcomp=True:
            * "Given a galaxy in environment E, what is the probability it belongs
              to Feature Cluster k?"
            * That is, what fractions of the environment belong to what FG?
            * Each point is [FG_in_env / total_env]
        If envfrac=True:
             * "Given a galaxy in Feature Cluster k, what is the probability it belongs
               to X environment?"
             * That is, what fractions of the FG belong to what environment?
             * Each point is [env_in_FG / total_FG]
    
    envcomp   : bool
        * if True, plot FG fractions in each environment
        * default is False
    envfrac   : bool
        * if True, plot environment fractions in each FG
        * default is False
    main_only : bool
        * if True, plot cluster, rich group, poor group, filament, field environments only (no nuance)
        * if False, plot all orthogonal environments -- pure field/cluster/rg/pg/filament, 
                                                      rg+fil, pg+fil, clus+fil
    '''        
    if (not envcomp and not envfrac) or (envcomp and envfrac):
        print('plot_env_frac() -- either choose envcomp or envfrac. Do not set both True or both False.')
        return
        
    #define Feature Cluster colors
    colors, edgecolors, marker_shapes = marker_palette(feature_data)

    env_defs = make_env_defs(feature_data, main_only=main_only)
    env_names = list(env_defs.keys())
    
    #create array of k values
    try:
        unique_clusters = sorted(np.unique(feature_data['Feature Cluster']))
    except:
        print('"Feature Cluster" column not found. Please run k-means or HDBSCAN clustering before continuing!')
        return
    
    #create indices for these galaxies (for the x-axis)
    index = np.arange(1,len(env_names)+1,1)
    
    #initialize the figure
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    
    #create storage variables so that I can connect the dots when the loop finishes. yay dots.
    line_x = {k_cluster: [] for k_cluster in unique_clusters}
    line_y = {k_cluster: [] for k_cluster in unique_clusters}
    err_y_low = {k_cluster: [] for k_cluster in unique_clusters}
    err_y_up = {k_cluster: [] for k_cluster in unique_clusters}
    
    #for every environment, plot its corresponding fraction and uncertainty in every Feature Cluster
    #OR
    #for every environment, plot each constituent Feature Cluster's fraction and uncertainty
    for i, (env_name, env_flag) in enumerate(env_defs.items()):
        
        #pull the env flag from feature_data
        env = feature_data[env_flag]
        
        for k_cluster in unique_clusters:
            
            #total galaxies in the feature cluster (will need for plotting as well!)
            feature_group = feature_data.loc[feature_data['Feature Cluster'] == k_cluster]
            Ngal_feature_group = len(feature_group)
            
            ###########
            # ENVFRAC #
            ###########
            
            if envfrac:
                #get the total number of galaxies in the feature cluster
                total = Ngal_feature_group
                
                #of the galaxies in the feature cluster, how many belong to x environment?
                #creates an array of 0s and 1s; 1=part of subset, 0=not part of subset
                #the average of this, in fact, IS the subset / total fraction!
                subset_data = (env_flag[feature_data['Feature Cluster'] == k_cluster].values).astype(int)
                
                title_ = 'Environment Distribution Within each Feature Cluster'
                legend_loc = 'upper left'
                ylim1 = 0
                ylim2 = None

            ###########
            # ENVCOMP #
            ###########
            
            #otherwise, get total number galaxies in the environment
            if envcomp:
                total = len(env)
                
                #of galaxies_in_env, how many are in Feature Cluster k?
                #creates an array of 0s and 1s; 1=part of subset, 0=not part of subset
                #the average of this, in fact, IS the subset / total fraction!
                subset_data = (env['Feature Cluster'].values == k_cluster).astype(int)
                
                title_ = 'Feature Cluster Composition Within each Environment'
                legend_loc = 'center left'
                ylim1 = 0
                ylim2 = 0.85
            
            ########
            # BOTH #
            ########
            
            #if total = 0...no use in including the data.
            if total == 0:
                line_x[k_cluster].append(index[i])
                line_y[k_cluster].append(0)
                err_y_low[k_cluster].append(0)
                err_y_up[k_cluster].append(0)
                continue
            
            #calculate fraction and bootstrap uncertainty
            #the uncertainty is on the mean of the data. remember that mean = subset/total when we convert
            #the subset array to 0s and 1s!
            fraction = np.mean(subset_data)
            ci_low, ci_up = get_bootstrap_confint(subset_data, bootfunc=np.mean)
            
            #convert bounds to asymmetric errorbars around the point
            unc_low = max(0.0, fraction - ci_low)
            unc_up  = max(0.0, ci_up - fraction)
                        
            #store the line variables!
            line_x[k_cluster].append(index[i])
            line_y[k_cluster].append(fraction)
            err_y_low[k_cluster].append(unc_low)
            err_y_up[k_cluster].append(unc_up)
            
            #define label for legend, but only for the first point of each FG (to avoid redundancies)
            label_ = None if i!=0 else f'FC{k_cluster} ({Ngal_feature_group})'
            
            ax.scatter(index[i], fraction,  color=colors[k_cluster], label=label_, s=90, 
                       edgecolor=edgecolors[k_cluster], marker=marker_shapes[k_cluster], zorder=3)
            
            #plot the asymmetric error bars
            err = ax.plot([index[i], index[i]], [fraction-unc_low, fraction+unc_up], 
                          color=colors[k_cluster], alpha=0.5, lw=2.5, zorder=2)

    for k_cluster in unique_clusters:
        
        #connect the dots using the stored values
        ax.plot(line_x[k_cluster], line_y[k_cluster], color=edgecolors[k_cluster], 
                linewidth=2.2, alpha=0.3, zorder=1)
        
        #create shaded regions between uncertainties, also using stored values!
        ax.fill_between(line_x[k_cluster], 
                        np.asarray(line_y[k_cluster])-np.asarray(err_y_low[k_cluster]), 
                        np.asarray(line_y[k_cluster])+np.asarray(err_y_up[k_cluster]), 
                        color=colors[k_cluster], alpha=0.2, zorder=0)
    
    ax.set_xticks(index, env_names, rotation=45, fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(alpha=0.2)
    
    ax.set_ylim(ylim1, ylim2)
    ax.set_ylabel('Fraction of Galaxies',fontsize=17)

    #ax.set_title(title_)
    
    ax.legend(loc=legend_loc, fontsize=14)
    
    plt.show()

    
def plot_group_features(median_data, layout_dict=None, nser_ylim=None, re_ylim=None):
    '''
    AIM: create multiple subplots showing each group's features and their associated uncertainties (taken from bootstrapping)
    * median_data should comprise a dataframe table of feature medians and lower+upper uncertainties for each 
      of the Feature Clusters.
    * layout_dict must be a python dictionary comprising the coordinates on a 3x3 grid where each feature 
       will be plotted, as well as the column name of that feature in median_data
        * If None, defaults to Size Ratio, NUV-r and W1-W3 colors
    * nser_ylim should be a tuple of integers (ymin, ymax) dictating the axes limits for the Sersic index plots.
        * If None, default is (0, 2)
    * re_ylim should be a tuple of integers (ymin, ymax) dictating the axes limits for the effective radius plots.
        * If None, default is (0, 5)
    '''
    
    from math import ceil
    
    #extract the colors...
    cluster_colors, edge_colors, marker_shapes = marker_palette(median_data)
    
    #in the median_data table, there is one row for every kth Feature Cluster
    k = len(median_data)
    
    #intended layout dictionary for subpl0ts. the 'default' is a failsafe. :-)
    if layout_dict is None or not isinstance(layout_dict, dict):
        
        message = 'Using default layout dictionary for median group feature subplots...'
        
        print('#'*len(message))
        print(message)
        print('#'*len(message))

        layout_dict =  {(0, 0): 'Size Ratio',
                        (0, 1): 'NUV_r',
                        (1, 0): 'W1_W3'}
        ncol = 2
        nrow = 2
            
    else:
        #the last layout_dict entry is (i, j), where i=nrow and j=ncol
        #sort coordinates from least to greatest, pull the "greatest" from the list
        #then nrow = (i_last + 1)
        #for ncol...isolate the first row and determine the maximum column in that row.
        sorted_keys = sorted(layout_dict.keys())
        
        last_row = sorted_keys[-1][0]
        last_col = [n for n in sorted_keys if n[0]==0][-1][1]
        
        nrow = last_row + 1
        ncol = last_col + 1

    #desired dimensions per subplot (e.g., 4.5 inches wide, 3.5 inches high)
    #just to, y'know, semi-automate the scaling.
    subplot_width_inches = 4.5
    subplot_height_inches = 3.5
    
    #calculate total figure size
    fig_width = ncol * subplot_width_inches
    fig_height = nrow * subplot_height_inches
    
    #determine the unique cluster IDs
    unique_clusters = sorted(median_data['Feature Cluster'].unique())

    #INITIATE
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(fig_width, fig_height), constrained_layout=True)
    
    axes = np.atleast_2d(axes)   #so that I can use i,j indices
    
    #read values from the dictionary, 
    for (i, j), med_label in layout_dict.items():        

        ax = axes[i, j]   #i=row, j=column
        
        lowerr_label = med_label + '_err_low'
        upperr_label = med_label + '_err_high'
        
        #plot every Feature Cluster's median + uncertainty
        for k_cluster in unique_clusters:
            
            #pull the feature cluster number, ignore 0th index 
            row = median_data.loc[median_data['Feature Cluster'] == k_cluster].iloc[0]
            
            median  = row[med_label]
            low_err = row[lowerr_label]
            upp_err = row[upperr_label]
                        
            #this line will only plot one point per iteration of the k_cluster 'for' loop
            im = ax.scatter(k_cluster, median, s=100, 
                            edgecolor=edge_colors[k_cluster], marker=marker_shapes[k_cluster],
                            color=cluster_colors[k_cluster], zorder=2, label=f'FC {k_cluster}')
            
            #plot the error bars
            err = ax.plot([k_cluster, k_cluster], [median-low_err, median+upp_err], 
                          color=edge_colors[k_cluster], zorder=1)
        
        #assign row y-axes limits
        if 'CN' in med_label:
            ylims = nser_ylim
            if nser_ylim is None:
                ylims = (0.0,5)
        elif 'CRE' in med_label:
            ylims = re_ylim
            if re_ylim is None:
                ylims = (0,6)
        else:
            ylims = ()  #no limits :-)
        
        ax.set_ylim(*ylims)  #cute way to put in a tuple for the ymin and ymax arguments!
        
        #set appropriate x-limits
        ax.set_xlim(min(unique_clusters)-0.5, max(unique_clusters)+0.5)
        
        #make x-axis increments of 1, since k is an integer!
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=1.0))
        
        ax.set_xlabel('Feature Cluster [k]',fontsize=14)
        ax.set_ylabel(get_feature_label(med_label, LABEL_DICT),fontsize=14)   #need the fancy schmancy name!
        ax.grid(alpha=0.1)
        
        if (i==0) and (j==0):
            ax.legend(fontsize=13)
        
    #lastly...remove axes not used in the layout_dict
    used_axes = set(layout_dict.keys())
    
    for i in range(nrow):
        for j in range(ncol):
            if (i, j) not in used_axes:
                fig.delaxes(axes[i, j])

    plt.show()
    return


def virgowise_median_plot(feature_data, plot_paper1=False):
    '''
    AIM: for each of the feature clusters, reproduce median size ratio vs. environment plot from Conger+2025.
    plot_paper1 : bool, indicates whether plots should include the conger+2025 medians+uncertainties
    '''
        
    #define Feature Cluster colors
    colors, edgecolors, marker_shapes = marker_palette(feature_data)

    #apply the W3 SNR > 10. condition
    feature_data = trim_ratios(feature_data.copy(), print_=True)
    
    env_defs = make_env_defs(feature_data, main_only=True)
    env_names = list(env_defs.keys())
        
    index = np.arange(1,len(env_names)+1,1)
    
    #create array of k values
    try:
        unique_clusters = sorted(np.unique(feature_data['Feature Cluster']))
    except:
        print('"Feature Cluster" column not found. Please run k-means or HDBSCAN clustering before continuing!')
        return
    
    for k in unique_clusters:
        
        kflag = (feature_data['Feature Cluster']==k)
    
        #will generate the self.outlier_flag variable needed to, well, trim the outliers.
        ratios = feature_data['Size Ratio'][kflag].values

        re_data = [ratios[flag[kflag]] for flag in env_defs.values()]
                
        print([f'{env} count: {len(x)}' for env, x in list(zip(env_names, re_data))])

        central_pts = []
        
        err_upper_bootstrap = []
        err_lower_bootstrap = []

        for j,data in enumerate(re_data):      #j==index, data==value

            central_pts.append(np.median(data))
            lower_err, upper_err = get_bootstrap_confint(data,bootfunc=np.median,nboot=1000)

            err_upper_bootstrap.append(upper_err)
            err_lower_bootstrap.append(lower_err)

        fig, ax = plt.subplots(1,1,figsize=(10,6))
        ax.scatter(index,central_pts,color=colors[k],s=50,zorder=2,edgecolors=edgecolors[k],marker=marker_shapes[k],
                   label=f'FC{k} Median')

        xmin,xmax = ax.get_xlim()
        xfield = np.linspace(xmin,xmax,50)

        ins = ax.inset_axes([0.536,0.533,0.46,0.46])
        ins.scatter(index,central_pts,color=colors[k],s=20,zorder=2,edgecolors=edgecolors[k],marker=marker_shapes[k])
        ins.grid(alpha=0.2)
        ins.tick_params(axis='y',which='major',labelsize=15)
        ins.tick_params(axis='x',which='both',bottom=False,labelbottom=False)

        for n in range(5):
            ax.plot([index[n],index[n]], [err_lower_bootstrap[n],err_upper_bootstrap[n]], color=edgecolors[k], zorder=1)
            ins.plot([index[n],index[n]], [err_lower_bootstrap[n],err_upper_bootstrap[n]], color=edgecolors[k], zorder=1)

        #field ymin, ymax for shaded region
        ymax = np.ones(50)*(err_upper_bootstrap[-1])
        ymin = np.ones(50)*(err_lower_bootstrap[-1])

        ax.fill_between(xfield,ymax,ymin,color='crimson',alpha=.1)
        ax.set_ylim(0.4,2.0)

        ins.fill_between(xfield,ymax,ymin,color='crimson',alpha=0.1)

        ax.set_xticks(index, env_names, rotation=10, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(alpha=0.2)
        ax.set_ylabel(r'R$_{12}$/R$_{3.4}$',fontsize=20)
        
        #include results from Virgo Filaments I (Conger+2025 -- that's me)
        if plot_paper1:
            medians=[0.848,0.928,0.920,0.879,0.949]
            lower=[0.819,0.909,0.901,0.856,0.939]
            upper=[0.880,0.950,0.933,0.906,0.954]
            ax.scatter(index,medians,color='gray',s=50,zorder=2,edgecolors='black',alpha=0.3,label=f'Conger+2025')
            ins.scatter(index,medians,color='gray',s=50,zorder=2,edgecolors='black',alpha=0.3)
        
            for n in range(5):
                ax.plot([index[n],index[n]], [lower[n],upper[n]], color='black', zorder=1, alpha=0.3)
                ins.plot([index[n],index[n]], [lower[n],upper[n]], color='black', alpha=0.3)
        
            ins.set_ylim(0.67,1.09)
            
        ax.legend(loc='upper left',fontsize=14)
        plt.show()        
        

def feature_rainclouds(feature_data, feature_list=None):
    '''
    AIM: create multiple raincloud plots showing each group's features and their associated distributions.
        * 1/2 Violin Plot
        * Boxplot
        * Scatterplot
    * feature_data should comprise a dataframe table of all galaxies and their feature values & 
        lower+upper uncertainties.
    * feature_list must be a python list of strings comprising the table column names of features the 
        user would like to be plotted.
    
    * inspired by https://arxiv.org/pdf/2512.15137
    
    '''  
    
    #create palette maps
    color_map, edge_map, _ = marker_palette(feature_data)
    
    #grab number of unique Feature Clusters
    k_clusters = sorted(feature_data['Feature Cluster'].unique())
        
    #intended layout dictionary for subpl0ts.
    if feature_list is None or not isinstance(feature_list, list):
        
        message = 'Using default feature list for feature raincloud plots:'
        
        feature_list =  [#'CRE_g_unscaled',
                         #'CRE_W1-fixBA_unscaled',
                         #'CRE_W3-fixBA_unscaled',
                         #'CN_g_unscaled',
                         #'CN_W1-fixBA_unscaled',
                         #'CN_W3-fixBA_unscaled',
                         'Size Ratio',
                         'NUV_r',
                         'W1_W3']
        
        print('#'*len(message))
        print(message)
        print(feature_list)
        print('#'*len(message))
    
    message = f'Incoming...expect an output of {len(feature_list)} plot(s).'
        
    print('#'*len(message))
    print(message)
    print('#'*len(message))
    
    #I will begin with one plot per feature. here's hoping the number of output figures is not too obnoxious.
    for feature_name in feature_list:
        
        #create copy of full feature_data table
        mod_df = feature_data.copy()
        
        if feature_name in ['NUV_r','W1_W3']:
            mod_df = trim_colors(mod_df, print_=False)   #remove illegitimate magnitude entries
        elif feature_name == 'Size Ratio':
            mod_df = trim_ratios(mod_df, print_=False)   #remove ratios calculated with W3 SNR < 10. AND those with np.nan. etc.
        
        #create bool flags for each k feature cluster
        kflags = {k: (mod_df['Feature Cluster'].values==k) for k in k_clusters}
        
        fig, ax = plt.subplots(figsize=(10,6))
        data_x = [mod_df[feature_name][kflags[k]] for k in k_clusters]
        
        #here is where you can impose outlier flags, if necessary
        #data_x = [data[data[feature_name] < some_bound] for data in data_x]
        
        #create the botplox (boxplot)
        bp = ax.boxplot(data_x, patch_artist=True, vert=False, showfliers=False,
                medianprops=dict(color='k', linewidth=1.5), widths=0.1)
        
        #change colors, add some transparency
        for patch, color in zip(bp['boxes'], color_map):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        
        #create the violin plot
        vp = ax.violinplot(data_x, points=300, showmeans=False, showextrema=False, showmedians=False, vert=False)
        
        for i, b in enumerate(vp['bodies']):

            #clip violin plot so we only see the upper half
            #b is a single violin plot "body," which is drawn symmetrically about i+1
            #get_paths()[0] collects polygon which defines that body
            #.vertices[:, 1] extracts all of the y-components of the polygon vertices
            #np.clip(a, low, high)
                # any value below 'low' is set to 'low'
                # any value above 'high' is set to 'high'
                # as such, any part of the violin below i+1 and above i+1.6 is trimmed (clipped)
            yvals = b.get_paths()[0].vertices[:, 1]
            b.get_paths()[0].vertices[:, 1] = np.clip(yvals, i+1, i+1.6)
            
            #change the desired color
            b.set_color(edge_map[i])
            b.set_alpha(0.4)
        
        for i, features in enumerate(data_x):
            
            #add some "jitter" so the points of the features do not overlap one another on the y-axis
            #indeed...without jitter, all points would like on a horizontal line. not helpful.
            
            #np.full is like np.zeros, but with i+8 values
            #this will anchor all Feature Cluster points to some fixed y level
            y = np.full(len(features), i + 0.8)
            
            #add some random vertical displacement to the y array
            y += np.random.uniform(low=-0.05, high=0.05, size=len(y))
            
            #now...plot the scattered points.
            plt.scatter(features, y, s=10, c=edge_map[i], alpha=0.2)
        
        ax.set_yticks([k+1 for k in k_clusters])
        ax.set_yticklabels([f'FC{k}' for k in k_clusters], fontsize=15)
        ax.set_xlabel(get_feature_label(feature_name, LABEL_DICT), fontsize=15)   #need the fancy schmancy name
        
        if feature_name=='Size Ratio':
            ax.set_xlim(0,3)
        if 'CRE' in feature_name:
            ax.set_xlim(-0.25,15.25)
        
        plt.show()

        
####################################################################
####################################################################
# Plotting Feature Cluster Physical Properties per dSFR Population #
####################################################################
####################################################################

def plot_median_nser_pop(feature_data, n_pop):    
    '''
    Aim: Use plot_group_features to generate n_pop 1x2 subplots of Sersic index distributions for the Feature Clusters.
    * This code is for a specific set of science plots involving W1 and g-band!
    '''
    from table_utils import dsfr_columns, create_median_table

    if n_pop not in [2,3]:
        print('n_pop variable must be 2 or 3. Unable to continue. Gob job.')
        return
    if 'ms_pop' not in feature_data.columns:
        print('Need to run $/table_utils/dsfr_columns.py before creating this plot.')
        return
    
    layout_dict =  {(0, 0): 'CN_g_unscaled',
                    (0, 1): 'CN_W1-fixBA_unscaled'}

    df = dsfr_columns(feature_data, n_pop)

    for pop in ['ms_pop','suppressed_pop','passive_pop']:
        if (pop=='suppressed_pop') and (n_pop==2):
            continue #go to next iteration; if n_pop=2, then there is no transition population
        print(pop)
        flag=df[pop]
        df_med = create_median_table(df[flag], ['CN_g','CN_W1-fixBA'])  
        plot_group_features(df_med, layout_dict=layout_dict, nser_ylim=None, re_ylim=None)
        
        
def plot_ttype_pop(feature_data, n_pop):    
    '''
    Aim: Use plot_group_features to generate n_pop (2 or 3) 1x2 subplots of Sersic index distributions for the Feature Clusters.
    * This code is for a specific set of science plots involving W1 and g-band!
    '''
    from table_utils import dsfr_columns, create_median_table

    if n_pop not in [2,3]:
        print('n_pop variable must be 2 or 3. Unable to continue. Gob job.')
        return
    if 'suppressed_pop' not in feature_data.columns:
        print('Need to run $/table_utils/dsfr_columns.py before creating this plot.')
        return

    df = dsfr_columns(feature_data, n_pop)

    for pop in ['ms_pop','suppressed_pop','passive_pop']:
        if (pop=='ms_pop') and (n_pop==2):
            continue #go to next iteration; if n_pop=2, then there is no transition population
        flag=df[pop]
        
        #drop any NaN values, indicating that the galaxy has no t-type available
        init_len = len(df[flag])
        df = df.copy().dropna(subset=['t_type'])
        post_len = len(df[flag])
        
        print(f'Dropping {init_len-post_len} NaN values from {pop} samples.')
                
        feature_rainclouds(df[flag], feature_list=['t_type'])


def plot_pop_frac(feature_data, n_pop=3):
    '''
    Aim: Create scatterplot of the fractional composition of each FC per dSFR population: main sequence, suppressed, and passive.
        * roughly mimics the setup of env_fraction
    '''
    if n_pop not in [2,3]:
        print('n_pop variable must be 2 or 3. Unable to continue. Gob job.')
        return
    if 'ms_pop' not in feature_data.columns:
        print('Need to run $/table_utils/dsfr_columns.py before creating this plot.')
        return
    
    #create array of k values
    try:
        unique_clusters = sorted(np.unique(feature_data['Feature Cluster']))
    except:
        print('"Feature Cluster" column not found. Please run k-means or HDBSCAN clustering before continuing!')
        return
    
    #define Feature Cluster colors
    colors, edgecolors, marker_shapes = marker_palette(feature_data)
        
    #define x-axis indices...
    index = np.arange(1,n_pop+1,1)
    
    #dSFR population flags
    pop_dict = {'ms_pop':'Main\n Sequence','suppressed_pop':'Suppressed','passive_pop':'Passive'}
    dsfr_flags = [feature_data[name] for name in pop_dict.keys()]
    
    #initialize the figure
    fig, ax = plt.subplots(1,1,figsize=(7,5))
    
    #now, loop through every k cluster and plottt.
    for k_cluster in unique_clusters:
        
        #isolate total number of galaxies in the FC
        fc_total = len(feature_data[feature_data['Feature Cluster']==k_cluster])

        #of the galaxies in the feature cluster, how many belong to x dSFR population?
        
        #creates an array of 0s and 1s; 1=part of subset, 0=not part of subset
        #the average of this, in fact, IS the subset / total fraction!
        #do this N times, once per population type -- creates a list of length N
        subset_data = [(x[feature_data['Feature Cluster'] == k_cluster].values).astype(int) for x in dsfr_flags]
        
        #calculate fraction and bootstrap uncertainty
        #the uncertainty is on the mean of the data. remember that mean = subset/total when we convert
        #the subset array to 0s and 1s.
        fractions = [np.mean(dat) for dat in subset_data]
        CIs = [(get_bootstrap_confint(dat, bootfunc=np.mean)) for dat in subset_data]

        #define label for legend, but only for the first point of each FG (to avoid redundancies)
        label_ = f'FC{k_cluster} ({fc_total})'

        ax.scatter(index, fractions, color=colors[k_cluster], label=label_, s=150, 
                   edgecolor=edgecolors[k_cluster], marker=marker_shapes[k_cluster], zorder=3)

        #plot the asymmetric error bars
        for n in range(n_pop):
                        
            #convert CIs to asymmetric errorbars around the mean. choose 0 if the errorbar is < 0.
            #CIs[0] is lower 68%, CIs[1] is upper 68%
            unc_low, unc_up = (max(0.0, fractions[n] - CIs[n][0]), max(0.0, CIs[n][1] - fractions[n]))
            
            err = ax.plot([index[n], index[n]], [fractions[n]-unc_low, fractions[n]+unc_up], 
                          color=colors[k_cluster], alpha=0.5, lw=2.5, zorder=2)
    
    ax.set_xticks(index, pop_dict.values(), rotation=10, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    ax.set_ylabel('FC Subset / FC Total',fontsize=18)
    
    #ax.set_title('4$\sigma$ Limit',fontsize=18)
    
    ax.legend(fontsize=14)
    plt.show()


#############################################
#############################################
# Plotting Feature Clusters (d)SFR v. Mstar #
#############################################
#############################################

def plot_sfrmstar(feature_data, mstar_lim=None, sfr_lim=None, y='delta_logsfr', rectangle=False, MS_1SIGMA=None):
    '''
    AIM: plot Feature Clusters on [delta_logSFR] vs. [logMstar] axes.
    * Alternatively plots [logSFR] vs. [logMstar] with completeness limits shown.
    
    * If rectangle=True and y='delta_logsfr', uses MS_1SIGMA to draw population rectangles on the dsfr vs. mstar figure.
    '''
    #need delta_logsfr, logmstar, and Feature Cluster columns. otherwise, quit.
    if y not in feature_data.columns or 'logmstar' not in feature_data.columns:
        print(f'Need "logmstar" and {y} columns to use this function!')
        return
    if 'Feature Cluster' not in feature_data.columns:
        print('Need "Feature Cluster" column to use this function!')
        return
    
    #get palette colors
    palette, _, markers = marker_palette(feature_data)
    
    g = sns.JointGrid(data=feature_data, x="logmstar", y=y, height=5)

    # ---- MAIN SCATTER ----
    g.plot_joint(sns.scatterplot, data=feature_data, color='lightgray', alpha=0.4, linewidth=0.3)
    
    g.plot_joint(sns.scatterplot, data=feature_data, hue="Feature Cluster", 
                 palette=palette, style='Feature Cluster', markers=markers,
                 alpha=0.4, edgecolor="w", linewidth=0.3)
    
    # --- preserve the existing cluster legend (created by seaborn) ---
    legend_clusters = g.ax_joint.legend_
    
    # --- initialize the "handles" and "labels" lists that will be used for the plot legend
    handles = []
    labels = []
    
    if y=='delta_logsfr':
        y_label = r'$\Delta$log(SFR)'
        
        if rectangle and type(MS_1SIGMA) is float:
            #    add population rectangles!
            # main sequence   -1.5sig < dsfr < 1.5sig
            # suppressed   -1.5sig < dsfr < 4sig
            # passive      dsfr < 4sig

            from matplotlib.patches import Rectangle

            xmin, xmax = g.ax_joint.get_xlim()   #get width of x-axis for rectangles
            width = xmax - xmin
            ymin, ymax = g.ax_joint.get_ylim()   #get y-axis limits for rectangle

            #(lower left coordinate), width of rectangle, height of rectangle
            rect_ms = Rectangle((xmin, -1.5*MS_1SIGMA), width, ymax - (-1.5*MS_1SIGMA),        #height from -1.5sig to ymax
                                facecolor='lightblue', alpha=0.2, edgecolor='blue', zorder=0)
            rect_trans = Rectangle((xmin,-4*MS_1SIGMA), width, (-1.5*MS_1SIGMA) - (-4*MS_1SIGMA),    #height from -4sig to -1.5sig
                                   facecolor='gray', alpha=0.2, edgecolor='black', zorder=0)
            rect_sup = Rectangle((xmin,ymin), width, (-4*MS_1SIGMA)-ymin,            #height from ymin to -4*MS_1SIGMA
                                   facecolor='orangered', alpha=0.1, edgecolor='crimson', zorder=0)

            g.ax_joint.add_patch(rect_ms)
            g.ax_joint.add_patch(rect_trans)
            g.ax_joint.add_patch(rect_sup)
        
        else:
            print('Note: either the rectangle arg is set to False and/or MS_1SIGMA arg is not set to a float.')
        
    else:
        y_label='log(SFR)'
        
        #add sfr, mstar, ssfr limits
        if sfr_lim is not None:
            h_sfr = g.ax_joint.axhline(y=sfr_lim, color='crimson', linestyle='--', alpha=0.9, 
                                       linewidth=1.5)
            handles.append(h_sfr)
            labels.append('log(SFR) limit')
        
        if mstar_lim is not None:
            h_mstar = g.ax_joint.axvline(x=mstar_lim, color='blue', linestyle='--', alpha=0.9, 
                                         linewidth=1.5)
            handles.append(h_mstar)
            labels.append(f'log(Mstar) > {mstar_lim}')
        
        xplot = np.sort(feature_data['logmstar'],axis=None)
        
        h_ssfr, = g.ax_joint.plot([xplot[0],xplot[-1]], [-11.5+xplot[0],-11.5+xplot[-1]], color='gray', 
                                  linestyle='-.', alpha=1, linewidth=1.5)
        handles.append(h_ssfr)
        labels.append('log(sSFR) > -11.5')
        
        #add MS line!
        m, b = get_ms_line(feature_data['logmstar'], feature_data['logsfr'])
        
        h_ms, = g.ax_joint.plot([xplot[0], xplot[-1]], [m*xplot[0]+b, m*xplot[-1]+b], color='k', 
                                  linestyle='--', alpha=0.8, linewidth=1.5)
        handles.append(h_ms)
        labels.append('Main Sequence Line')
        
        # --- create 2nd legend for the limit lines ---
        legend_limits = g.ax_joint.legend(handles=handles,
                                          labels=labels,
                                          loc='lower right')

    g.ax_joint.set_xlabel('log(Mstar)',fontsize=14)
    g.ax_joint.set_ylabel(y_label,fontsize=14)
    
    # ---- KDE MARGINALS (the histogram distributions) ----
    for k, color in enumerate(palette):
        subset = feature_data[(feature_data["Feature Cluster"] == k)]

        #top marginal (logmstar)
        sns.kdeplot(x=subset["logmstar"], ax=g.ax_marg_x, color=color, fill=True, alpha=0.3, linewidth=1.2)

        #right marginal (dlogsfr)
        sns.kdeplot(y=subset[y], ax=g.ax_marg_y, color=color, fill=True, alpha=0.3, linewidth=1.2)
    
    # --- put back the Feature Cluster legend ---
    if legend_clusters is not None:
        g.ax_joint.add_artist(legend_clusters)
    
    g.fig.set_size_inches(12, 6)
    
    plt.show()


######################################
######################################
# Plotting Feature Cluster dSFR KDEs #
######################################
######################################

def plot_dSFR_KDEs(feature_data):
    '''
    AIM: plot dlog(SFR) KDE plots for all Feature Clusters; include KS-test statistics.
    '''
    
    #if the required columns are not present, cannot run the function. derp.
    if 'delta_logsfr' not in feature_data.columns:
        print('Need "delta_logsfr" column before proceeding!')
        return
    if 'Feature Cluster' not in feature_data.columns:
        print('Need "Feature Cluster" column before proceeding!')
        return
    
    #color/marker bookkeeping...
    colors, edgecolors, marker_shapes = marker_palette(feature_data)
    
    #grab number of unique Feature Clusters
    k_clusters = np.sort(feature_data['Feature Cluster'].unique())
    
    #create bool flags for each k feature cluster
    kflags = {k: (feature_data['Feature Cluster'].values==k) for k in k_clusters}
    
    for k in k_clusters:
        dlogsfr = feature_data['delta_logsfr'][kflags[k]]
        bin_width=0.2
        
        bins = np.arange(min(dlogsfr), max(dlogsfr) + bin_width, bin_width) 
        hist_plot = plt.hist(dlogsfr, color=colors[k], bins=bins,
                                density=True, alpha=0.2, edgecolor='gray')

        fig = sns.kdeplot(dlogsfr, color=colors[k], label=f'FC{k}: {len(feature_data[kflags[k]])}')
    
    plt.xlabel(r'$\Delta$logSFR',fontsize=14)
    plt.ylabel('Density',fontsize=14)

    fig.legend()
    plt.show()   
    
    #create unique pairs for K-S TEST...
    k_pairs = list(combinations(k_clusters, 2))
    
    #isolate the components of each kpair, then put into ks_2samp.
    for k1, k2 in k_pairs:
        
        ks_stat, p_value = ks_2samp(feature_data['delta_logsfr'][kflags[k1]],feature_data['delta_logsfr'][kflags[k2]])
        print('--------------------')
        print(f'ks stat (FG{k1} & FG{k2}): {ks_stat}')
        print(f'p-value (FG{k1} & FG{k2}): {p_value}')
        print('--------------------')
    
    return 


#######################################################
#######################################################
# Plotting Feature Clusters dSFR KDEs PER ENVIRONMENT #
#######################################################
#######################################################

def plot_KDE_env(feature_data, dsfr=True, w1ser=False, gser=False, main_only=False, stats=False):
    '''
    AIM: Create individual panels of the KDE distributions of k Feature Clusters separated by environment. 
    If stats=True, will print K-S test results
    This distribution is dictated by setting one of the following variables to True:
        * dsfr = dlogSFR
        * gser = g-band Sersic Index
        * w1ser = W1 Sersic Index
    If >1 is set to True, default will be dsfr, then w1ser. 
    '''

    if dsfr:
        prefix='delta_logsfr'
        xlims=(-6,2)
        bin_width = 0.3
    elif w1ser:
        prefix='CN_W1'
        xlims=(0,6)
        bin_width = 0.2
    elif gser:
        prefix='CN_g'
        xlims=(0,6)
        bin_width = 0.2
    else:
        print('Need to set dsfr, w1ser, or gser to True.')
        return
    
    #create list of columns corresponding to the prefix set above
    #[expression for item in iterable if condition]
    columns = [n for n in feature_data.columns if prefix in n]
    
    #if no columnnames match the prefix or Feature Cluster not present, then we cannot run the function. derp.
    if (len(columns)<1) or ('Feature Cluster' not in feature_data.columns):
        print(f'Need {prefix} or its _unscaled variant in the input dataframe. Might also be missing "Feature Cluster" column.')
        return
        
    #pull the "last" columnname. this will be the _unscaled variant of the prefix (if present) or whatever single
    #columnname remains (if len(columns)==1)
    colname = columns[-1]
    
    #next, define xaxis label.
    xaxis_label = get_feature_label(colname, LABEL_DICT)

    #color/marker bookkeeping...
    colors, edgecolors, marker_shapes = marker_palette(feature_data)

    #grab number of unique Feature Clusters
    k_clusters = np.sort(feature_data['Feature Cluster'].unique())

    #drop noise label if present (k-means won't have it)
    k_clusters = [k for k in k_clusters if k != -1]

    #create bool flags for each k feature cluster
    kflags = {k: (feature_data['Feature Cluster'] == k) for k in k_clusters}

    #this will output a dictionary of environment names/labels and their corresponding boolean flags!
    env_defs = make_env_defs(feature_data, main_only=main_only)
    
    #this next part is a bit tricky. we first define an array of empty strings ('', '', ...), but with an 
    #optional maximum length of len(xaxis_label)
    xlabels=np.zeros(len(env_defs), dtype=f'<U{len(xaxis_label)}')
    
    #then, since we want the LAST set of axes only to have the xaxis label, we assign the label to the -1st index
    #of the xlabels array
    xlabels[-1]=xaxis_label
    
    #plotting time.
    #first, the canvas. 1 column, len(env_defs) rows -- one per environment.
    fig = plt.figure(figsize=(8,18))
    plt.subplots_adjust(hspace=0.2)

    #loop over environments (dict: env name -> boolean mask)
    for n, (env_name, env_mask) in enumerate(env_defs.items()):
        ax = fig.add_subplot(len(env_defs), 1, n+1)

        for k in k_clusters:
            #isolates all galaxies in FGk and the env_mask environment
            x_feature = feature_data.loc[kflags[k] & env_mask, colname]

            #skip if too few points to make a meaningful histogram
            if len(x_feature) < 5:
                continue
            
            bins = np.arange(xlims[0], xlims[1] + bin_width, bin_width) 
            
            hist_plot = ax.hist(x_feature, color=colors[k], bins=bins,
                                density=True, alpha=0.2, edgecolor='gray')
            
            kde = sns.kdeplot(x_feature, color=colors[k], label=f'Feature Cluster {k}', ax=ax,
                             common_norm=False)

        ax.set_xlim(*xlims)
        ax.set_xlabel(xlabels[n], fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        
        #add environment label to the plot panel
        ax.text(0.02, 0.95, env_name, transform=ax.transAxes, ha='left', va='top', fontsize=11)
        
        #for env==0 (first box), add legend.
        if n == 0:
            ax.legend(loc='upper right')
            
        # --- KS TESTTTTT --- #    
        
        if stats:
            #create unique pairs for K-S TEST...
            k_pairs = list(combinations(k_clusters, 2))

            #isolate the components of each kpair, then put into ks_2samp.
            for k1, k2 in k_pairs:

                print(env_name)

                ks_stat, p_value = ks_2samp(feature_data[prefix][kflags[k1] & env_mask],feature_data[prefix][kflags[k2] & env_mask])
                print('--------------------')
                print(f'ks stat (FG{k1} & FG{k2}): {ks_stat}')
                print(f'p-value (FG{k1} & FG{k2}): {p_value}')
                print('--------------------')

    plt.show()

    
#######################################################################
#######################################################################
# Plotting dSFR cumulative histograms per environment for a single FC #
#######################################################################
#######################################################################

# HELPER FUNCTION FOR PLOTTING THE (EMPIRICAL) CUMULATIVE DISTRIBUTION FUNCTION 
# (in other words, "HOW TO PLOT CUMULATIVE HISTOGRAMS WITHOUT THE MATPLOTLIB VERTICAL LINE")
def plot_ecdf(data, **kwargs):
    x = np.sort(data)   #sort the x data from least to greatest
    y = np.arange(1, len(x)+1) / len(x)  #every y value from 0 to 1
    plt.step(x, y, where='post', **kwargs)   #create the step function...

def plot_cum_env(feature_data, fc=0, dsfr=True, w1ser=False, gser=False, main_only=False):
    '''
    AIM: create a single figure of the cumulative histogram curves for a given Feature Cluster in each environment.
    * fc must be an integer
    '''
    
    if dsfr:
        prefix='delta_logsfr'
    elif w1ser:
        prefix='CN_W1'
    elif gser:
        prefix='CN_g'
    else:
        print('Need to set dsfr, w1ser, or gser to True.')
        return
    
    #create colormap! ranges from 0 to 1. 
    cmap = plt.colormaps.get_cmap('viridis_r')
    
    #isolate all galaxies in fc
    fc_galaxies = feature_data[feature_data['Feature Cluster']==fc]
    
    #create dictionary of environment name : environment flag
    env_dict = make_env_defs(fc_galaxies, main_only=main_only)
    
    #initialize the figure
    fig = plt.figure(figsize=(8,5))
    
    #loop through every environment
    for i, (env_name, env_flag) in enumerate(env_dict.items()):
        
        #cmap takes floats (cmap(float)). choose a color for the corresponding environment.
        #if there is only one environment, default to the middle of the cmap
        color = cmap(i / (len(env_dict) - 1)) if len(env_dict) > 1 else cmap(0.5)
        
        if i==0:
            color='gold'
        
        plot_ecdf(fc_galaxies[prefix][env_flag], linewidth=2,
                  label=env_name.replace('\n',' ').replace('   ',' '),
                  color=color)
            
    plt.xlabel(LABEL_DICT[prefix],fontsize=14)
    plt.ylabel('Fraction of Galaxies',fontsize=14)
    plt.legend()
    
    title_dict = {0: 'Small disks', 1: 'Spheroids', 2: 'Large disks'}
    plt.title(f'FC{fc} ({title_dict[fc]})', fontsize=15)
    
    plt.xlim(-5.5,1.5)
    
    plt.show()
    
    #create unique pairs for K-S TEST...
    k_pairs = list(combinations(env_dict.keys(), 2))
    
    #isolate the components of each kpair, then put into ks_2samp.
    for env1, env2 in k_pairs:

        ks_stat, p_value = ks_2samp(fc_galaxies[prefix][env_dict[env1]],fc_galaxies[prefix][env_dict[env2]])
        env1_name=env1.replace('\n',' ').replace('   ',' ')
        env2_name=env2.replace('\n',' ').replace('   ',' ')
        print('--------------------')
        print(f"ks stat ({env1_name} || {env2_name}): {ks_stat:.3f}")
        print(f"p-value ({env1_name} || {env2_name}): {p_value:.3e}")
        print('--------------------')