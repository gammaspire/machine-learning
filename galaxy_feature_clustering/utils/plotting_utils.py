from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

from feature_utils import make_label_dictionary, get_feature_label, make_env_defs
from data_utils import get_bootstrap_confint, get_ms_line


######################################
######################################
# Defining a Consistent Plot Palette #
######################################
######################################

def marker_palette(feature_data):
    
    shapes = ['<', 's', '^', '*', 'D', 'v', 'X', '<', 'h', '>']
        
    try:
        clusters = feature_data['Feature Cluster'].unique()
        k = len(clusters)  #number of feature groups
        noise_flag = (-1 in clusters)
        k = k-1 if noise_flag else k
    except:
        k = len(feature_data)  #also number of feature groups --> for one row per feature group
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
    
    plt.figure(figsize=(8,6))
    plt.plot(K, silhouettes, 'o-', color='green')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show()
    

##########################################################
##########################################################
# Visualizing the Feature Groups in 2D PCA or UMAP Space #
##########################################################
##########################################################

def plot_clusters(feature_data, x=None, y=None, PCA=False, UMAP=False):
    
    #pull the colors...
    cluster_colors, _, _ = marker_palette(feature_data)
    
    #sort the unique feature groups numerically
    unique_clusters = sorted(feature_data['Feature Cluster'].unique())
    #create a dictionary mapping each color to a feature group --> {k: color}
    color_map = {c: cluster_colors[i] for i, c in enumerate(unique_clusters)}
    
    #PCA and UMAP flag
    flag = (PCA | UMAP)
    
    if x is None and not flag:
        print('Unable to generate plot! Please either specify x, y columns, or set PCA=True or UMAP=True.')
        return
    
    x = 'Comp1' if flag else x
    y = 'Comp2' if flag else y
    
    if -1 not in feature_data['Feature Cluster'].unique():
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=feature_data, x=x, y=y, hue='Feature Cluster',
                        palette=color_map, alpha=0.7, edgecolor='w', linewidth=0.4)
    else:
        ax = sns.scatterplot(x=x, y=y, data=feature_data[feature_data['Feature Cluster'] == -1], alpha=0.1,
                            color='lightgray', edgecolor='w', linewidth=0.4, label='Noise')
        sns.scatterplot(x=x, y=y, data=feature_data[feature_data['Feature Cluster'] != -1], hue='Feature Cluster',
                        palette=color_map, alpha=0.7, edgecolor='w', linewidth=0.4, ax=ax)
    
    plt.xlabel('Component One')
    plt.ylabel('Component Two')
    
    plt.title(f'Structural Clusters in 2D Space')
    plt.tight_layout()
    plt.show()

    
##################################
##################################
# Plotting PCA Vector Components #
##################################
##################################

def plot_pca_components(feature_data, features, pca, cmap_name='tab20'):
    '''
    * Visualize PCA feature loadings for the first two components.
    * Features are sorted by total loading strength (or rather, the magnitude of their contribution.
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
                label=f"{features[i]} ({loading_strength[i]:.2f})")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Loadings Plot")
    plt.grid(alpha=0.2)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)

    plt.tight_layout()
    plt.show()


#################################################
#################################################
# Plotting Feature Group Environment Properties #
#################################################
#################################################
    
def plot_env_fraction(feature_data, main_only=False, envfrac=False, envcomp=False):
    '''
    INTERPRETATIONS:
        if envcomp=True:
            * "Given a galaxy in environment E, what is the probability it belongs
              to feature group k?"
            * That is, what fractions of the environment belong to what FG?
            * Each point is [FG_in_env / total_env]
        If envfrac=True:
             * "Given a galaxy in feature group k, what is the probability it belongs
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
    
    from feature_utils import make_env_defs
    
    #define feature group colors
    colors, edgecolors, marker_shapes = marker_palette(feature_data)

    env_defs = make_env_defs(feature_data, main_only=main_only)
    env_names = list(env_defs.keys())
    
    #create array of k values
    try:
        unique_clusters = sorted(np.unique(feature_data['Feature Cluster']))
    except:
        print('"Feature Cluster" column not found. Please run k-means or HDBSCAN clustering before continuing!')
        return
    
    #create dictionaries!
    color_map  = {c: colors[i]        for i, c in enumerate(unique_clusters)}
    edge_map   = {c: edgecolors[i]    for i, c in enumerate(unique_clusters)}
    shape_map  = {c: marker_shapes[i] for i, c in enumerate(unique_clusters)}
    
    #create indices for these galaxies (for the x-axis)
    index = np.arange(1,len(env_names)+1,1)
    
    #initialize the figure
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    
    #create storage variables so that I can connect the dots when the loop finishes. yay dots.
    line_x = {k_cluster: [] for k_cluster in unique_clusters}
    line_y = {k_cluster: [] for k_cluster in unique_clusters}
    err_y_low = {k_cluster: [] for k_cluster in unique_clusters}
    err_y_up = {k_cluster: [] for k_cluster in unique_clusters}
    
    #for every environment, plot its corresponding fraction and uncertainty in every feature group
    #OR
    #for every environment, plot each constituent feature group's fraction and uncertainty
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
                
                title_ = 'Environment Distribution Within each Feature Group'
                ylim1 = 0
                ylim2 = None

            ###########
            # ENVCOMP #
            ###########
            
            #otherwise, get total number galaxies in the environment
            if envcomp:
                total = len(env)
                
                #of galaxies_in_env, how many are in feature group k?
                #creates an array of 0s and 1s; 1=part of subset, 0=not part of subset
                #the average of this, in fact, IS the subset / total fraction!
                subset_data = (env['Feature Cluster'].values == k_cluster).astype(int)
                
                title_ = 'Feature Group Composition Within each Environment'
                ylim1 = 0
                ylim2 = 0.80
            
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
            label_ = None if i!=0 else f'Feature Group {k_cluster} (Ngal={Ngal_feature_group})'
            
            ax.scatter(index[i], fraction,  color=color_map[k_cluster], label=label_, s=90, 
                       edgecolor=edge_map[k_cluster], marker=shape_map[k_cluster], zorder=3)
            
            #plot the asymmetric error bars
            err = ax.plot([index[i], index[i]], [fraction-unc_low, fraction+unc_up], 
                          color=color_map[k_cluster], alpha=0.5, lw=2.5, zorder=2)

    for k_cluster in unique_clusters:
        
        #connect the dots using the stored values
        ax.plot(line_x[k_cluster], line_y[k_cluster], color=edge_map[k_cluster], 
                linewidth=2.2, alpha=0.3, zorder=1)
        
        #create shaded regions between uncertainties, also using stored values!
        ax.fill_between(line_x[k_cluster], 
                        np.asarray(line_y[k_cluster])-np.asarray(err_y_low[k_cluster]), 
                        np.asarray(line_y[k_cluster])+np.asarray(err_y_up[k_cluster]), 
                        color=color_map[k_cluster], alpha=0.2, zorder=0)
    
    ax.set_xticks(index, env_names, rotation=45, fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(alpha=0.2)
    
    ax.set_ylim(ylim1, ylim2)
    ax.set_ylabel('Fraction of Galaxies',fontsize=17)

    ax.set_title(title_)
    
    ax.legend(loc='upper left')
    
    plt.show()


def plot_env_stacked_hist(feature_data, main_only=False):
    '''
    AIM:
        For each environment, plot the fractional composition of feature groups as a stacked histogram.

    INTERPRETATION:
        "Given a galaxy in environment E, what is the probability it belongs
         to feature group k?"

    main_only:
        If True, restricts to the five primary environments.
    '''

    #color/marker bookkeeping
    colors, edgecolors, marker_shapes = marker_palette(feature_data)

    #define environments using boolean masks
    env_defs = make_env_defs(feature_data, main_only=main_only)
        
    #isolate the name strings
    env_names = list(env_defs.keys())

    #identify feature groups (ignore noise if present)
    feature_groups = sorted(c for c in feature_data['Feature Cluster'].unique() if c != -1)

    #number of environments and feature groups!
    n_env = len(env_names)
    n_fg  = len(feature_groups)

    #storage array for fractions...
    fractions = np.zeros((n_env, n_fg))

    #now compute the fractions
    for i, (env_name, env_flag) in enumerate(env_defs.items()):

        #find subset of galaxies belonging to environment E
        env_data = feature_data[env_flag]
        total = len(env_data)   #total galaxies in environment E

        for j, k in enumerate(feature_groups):

            #number of galaxies in environment E AND feature group k
            subset = np.sum(env_data['Feature Cluster'] == k)

            #avoid division by zero (empty environments)
            if total == 0:
                fractions[i, j] = 0.0
                continue

            #fraction
            fractions[i, j] = subset / total   #ith environment (row), jth feature group (column)

    #PLOTTING ZEIT
    fig, ax = plt.subplots(figsize=(10, 6))

    #tracks the bottom of each stacked bar. this prevents the components from overlapping!
    bottom = np.zeros(n_env)

    #x locations for environments
    x = np.arange(n_env)

    #draw stacked bars
    for j, k in enumerate(feature_groups):

        ax.bar( x,
                fractions[:, j],
                bottom=bottom,
                color=colors[j],
                edgecolor=edgecolors[j],
                label=f'Feature Group {k}',
                zorder=2)

        #update bottom for next feature group
        bottom += fractions[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=45, fontsize=13)
    ax.set_ylabel('Fraction of Galaxies', fontsize=15)
    ax.set_ylim(0, 1)

    ax.set_title('Feature Group Composition by Environment', fontsize=16)

    ax.grid(axis='y', alpha=0.25)
    #ax.legend(title='Feature Group', bbox_to_anchor=(1.02, 1),
    #          loc='upper left', frameon=False)

    plt.tight_layout()
    plt.show()
    

##############################################
##############################################
# Plotting Feature Group Physical Properties #
##############################################
##############################################

def plot_corner(feature_data, features=None):
    #suppress those WARNINGS PLS
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    #for editing the column labels so axes are readable!
    from feature_utils import get_feature_label, make_label_dictionary
    
    cluster_colors, _, _ = marker_palette(feature_data)
    
    #we want to plot the UNSCALED features!
    #I am also dictating the plotted features. not oops.
    if features is None:
        features=['CRE_W1-fixBA_unscaled', 'CN_W1-fixBA_unscaled', 
                  'CRE_W3-fixBA_unscaled', 'CN_W3-fixBA_unscaled',
                  'CRE_g_unscaled', 'CN_g_unscaled']
        #if 'AVG_RE_gr_unscaled' in feature_data.columns:
        #    features[4] = 'AVG_RE_gr_unscaled'
        #if 'AVG_RE_W1W2_unscaled' in feature_data.columns:
        #    features[0] = 'AVG_RE_W1W2_unscaled'
    
    unique_clusters = sorted(feature_data['Feature Cluster'].unique())
    color_map = {c: cluster_colors[i] for i, c in enumerate(unique_clusters)}

    g = sns.pairplot(feature_data, vars=features, hue='Feature Cluster', 
                     palette=color_map, corner=True,
                     plot_kws={'alpha': 0.6, 's': 20}, diag_kind=None)  #'kde')
    
    #edit the labels!
    label_dict = make_label_dictionary()

    #pull the axes from the corner plot
    axes_flat = [ax for ax in g.axes.flat if ax is not None]
    for ax in axes_flat:
        if ax.get_xlabel():
            ax.set_xlabel(get_feature_label(ax.get_xlabel(), label_dict))
        if ax.get_ylabel():
            ax.set_ylabel(get_feature_label(ax.get_ylabel(), label_dict))
    
    g._legend.remove()
    #g.fig.suptitle('Feature Clusters in Physical Space', y=1.02)
    plt.show()

    
def plot_group_features(median_data, layout_dict=None, nser_ylim=None, re_ylim=None):
    '''
    AIM: create multiple subplots showing each group's features and their associated uncertainties (taken from bootstrapping)
    * median_data should comprise a dataframe table of feature medians and lower+upper uncertainties for each 
      of the feature groups.
    * layout_dict must be a python dictionary comprising the coordinates on a 3x3 grid where each feature 
       will be plotted, as well as the column name of that feature in median_data
        * If None, defaults to W1, W3, and g-band Re and nser; Size Ratio, NUV-r and W1-W3 colors
    * nser_ylim should be a tuple of integers (ymin, ymax) dictating the axes limits for the Sersic index plots.
        * If None, default is (0, 2)
    * re_ylim should be a tuple of integers (ymin, ymax) dictating the axes limits for the effective radius plots.
        * If None, default is (0, 5)
    '''
    
    from math import ceil
    
    #extract the colors...
    cluster_colors, edge_colors, marker_shapes = marker_palette(median_data)
    
    #extract the label dictionary for the y-axis!
    label_dict = make_label_dictionary()
    
    #in the median_data table, there is one row for every kth feature group
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
    
    axes = np.atleast_2d(axes)
    
    #read values from the dictionary, 
    for (i, j), med_label in layout_dict.items():        

        ax = axes[i, j]   #i=row, j=column
        
        lowerr_label = med_label + '_err_low'
        upperr_label = med_label + '_err_high'
        
        #plot every feature group's median + uncertainty
        for k_cluster in unique_clusters:
            
            #pull the feature cluster number, ignore 0th index 
            row = median_data.loc[median_data['Feature Cluster'] == k_cluster].iloc[0]
            
            median  = row[med_label]
            low_err = row[lowerr_label]
            upp_err = row[upperr_label]
                        
            #this line will only plot one point per iteration of the k_cluster 'for' loop
            im = ax.scatter(k_cluster, median, s=100, 
                            edgecolor=edge_colors[k_cluster], marker=marker_shapes[k_cluster],
                            color=cluster_colors[k_cluster], zorder=2, label=f'Feature Group {k_cluster}')
            
            #plot the error bars
            err = ax.plot([k_cluster, k_cluster], [median-low_err, median+upp_err], 
                          color=edge_colors[k_cluster], zorder=1)
        
        #assign row y-axes limits
        if 'CN' in med_label:
            ylims = nser_ylim
            if nser_ylim is None:
                ylims = (0.0,4)
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
        
        ax.set_xlabel('Feature Group [k]')
        ax.set_ylabel(get_feature_label(med_label, label_dict))   #need the fancy schmancy name!
        ax.grid(alpha=0.1)
        
    #lastly...remove axes not used in the layout_dic\\
    used_axes = set(layout_dict.keys())

    for i in range(nrow):
        for j in range(ncol):
            if (i, j) not in used_axes:
                fig.delaxes(axes[i, j])

    plt.show()
    return


def plot_median_nser_pop(feature_data, n_pop):    
    '''
    Aim: Use plot_group_features to generate n_pop (2 or 3) 1x2 subplots of Sersic index distributions for the feature groups.
    * This code is for a specific set of science plots involving W1 and g-band!
    '''
    from table_utils import dsfr_columns, create_median_table

    layout_dict =  {(0, 0): 'CN_g_unscaled',
                    (0, 1): 'CN_W1-fixBA_unscaled'}

    df = dsfr_columns(feature_data, 3)

    for pop in ['ms_pop','transition_pop','suppressed_pop']:
        flag=df[pop]
        df_med = create_median_table(df[flag], ['CN_g','CN_W1-fixBA'])  
        plot_group_features(df_med, layout_dict=layout_dict, nser_ylim=None, re_ylim=None)



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
    
    #create label dictionary for Feature Group features, this time for x-axis
    label_dict = make_label_dictionary()
    
    #grab number of unique feature groups
    k_clusters = sorted(feature_data['Feature Cluster'].unique())
    
    #create bool flags for each k feature cluster
    kflags = {k: (feature_data['Feature Cluster'].values==k) for k in k_clusters}
    
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
    
    message = f'Incoming...expect an output of {len(feature_list)} plots.'
        
    print('#'*len(message))
    print(message)
    print('#'*len(message))
    
    #I will begin with one plot per feature. here's hoping the number of output figures is not too obnoxious.
    for feature_name in feature_list:
        fix, ax = plt.subplots(figsize=(10,6))
        data_x = [feature_data[feature_name][kflags[k]] for k in k_clusters]
        
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
            #this will anchor all feature group points to some fixed y level
            y = np.full(len(features), i + 0.8)
            
            #add some random vertical displacement to the y array
            y += np.random.uniform(low=-0.05, high=0.05, size=len(y))
            
            #now...plot the scattered points.
            plt.scatter(features, y, s=10, c=edge_map[i], alpha=0.05)
        
        ax.set_yticks([k+1 for k in k_clusters])
        ax.set_yticklabels([f'Feature Cluster {k}' for k in k_clusters])
        ax.set_xlabel(get_feature_label(feature_name, label_dict))   #need the fancy schmancy name!

        plt.show()


###########################################
###########################################
# Plotting Feature Groups (d)SFR v. Mstar #
###########################################
###########################################

def plot_sfrmstar(feature_data, mstar_lim=None, sfr_lim=None, y='delta_logsfr'):
    '''
    AIM: plot feature groups on [delta_logSFR] vs. [logMstar] axes.
    * Alternatively plots [logSFR] vs. [logMstar] with completeness limits shown.
    '''
    #need delta_logsfr, logmstar, and Feature Cluster columns!
    if y not in feature_data.columns or 'logmstar' not in feature_data.columns:
        print(f'Need "logmstar" and {y} columns to use this function!')
        return
    if 'Feature Cluster' not in feature_data.columns:
        print('Need "Feature Cluster" column to use this function!')
        return
    
    #get palette colors
    palette, _, _ = marker_palette(feature_data)
    
    g = sns.JointGrid(data=feature_data, x="logmstar", y=y, height=5)

    # ---- MAIN SCATTER ----
    g.plot_joint(sns.scatterplot, data=feature_data, color='lightgray', alpha=0.4, linewidth=0.3)
    
    g.plot_joint(sns.scatterplot, data=feature_data, hue="Feature Cluster", 
                 palette=palette, alpha=0.4, edgecolor="w", linewidth=0.3)
    
    # --- preserve the existing cluster legend (created by seaborn) ---
    legend_clusters = g.ax_joint.legend_
    
    if y=='delta_logsfr':
        y_label = r'$\Delta$log(SFR)'

    else:
        y_label='log(SFR)'
        
        #add sfr, mstar, ssfr limits
        
        h_sfr = g.ax_joint.axhline(y=sfr_lim, color='crimson', linestyle='--', alpha=0.9, 
                                   linewidth=1.5)
        h_mstar = g.ax_joint.axvline(x=mstar_lim, color='blue', linestyle='--', alpha=0.9, 
                                     linewidth=1.5)
        
        xplot = np.sort(feature_data['logmstar'],axis=None)
        
        h_ssfr, = g.ax_joint.plot([xplot[0],xplot[-1]], [-11.5+xplot[0],-11.5+xplot[-1]], color='gray', 
                                  linestyle='-.', alpha=1, linewidth=1.5)
        
        #add MS line!
        m, b = get_ms_line(feature_data['logmstar'], feature_data['logsfr'])
        h_ms, = g.ax_joint.plot([xplot[0], xplot[-1]], [m*xplot[0]+b, m*xplot[-1]+b], color='k', 
                                  linestyle='--', alpha=0.8, linewidth=1.5)
        
        # --- create 2nd legend for the limit lines ---
        legend_limits = g.ax_joint.legend(handles=[h_ms, h_sfr, h_mstar, h_ssfr],
                                          labels=['Main Sequence Line', 'log(SFR) limit', 'log(Mstar) limit', 
                                                  'log(sSFR) limit'],
                                          loc='lower right',
                                          title='Completeness Limits')

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


#####################################
#####################################
# Plotting Feature Groups dSFR KDEs #
#####################################
#####################################

def plot_dSFR_KDEs(feature_data):
    '''
    AIM: plot dlog(SFR) KDE plots for all feature groups; include KS-test statistics.
    '''
    from scipy.stats import ks_2samp
    from itertools import combinations
    
    #if the required columns are not present, cannot run the function. derp.
    if 'delta_logsfr' not in feature_data.columns:
        print('Need "delta_logsfr" column before proceeding!')
        return
    if 'Feature Cluster' not in feature_data.columns:
        print('Need "Feature Cluster" column before proceeding!')
        return
    
    #color/marker bookkeeping...
    colors, edgecolors, marker_shapes = marker_palette(feature_data)
    
    #grab number of unique feature groups
    k_clusters = np.sort(feature_data['Feature Cluster'].unique())
    
    #create bool flags for each k feature cluster
    kflags = {k: (feature_data['Feature Cluster'].values==k) for k in k_clusters}
    
    for k in k_clusters:
        dlogsfr = feature_data['delta_logsfr'][kflags[k]]
        fig = sns.kdeplot(dlogsfr, color=colors[k], label=f'Feature Group {k}: {len(feature_data[kflags[k]])}')
    
    plt.xlabel(r'$\Delta$logSFR')
    plt.title(r'$\Delta$logSFR KDE Distribution per Feature Group')
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


#####################################################
#####################################################
# Plotting Feature Groups dSFR KDEs PER ENVIRONMENT #
#####################################################
#####################################################

def plot_KDE_env(feature_data, dsfr=True, w1ser=False, gser=False, main_only=False):
    '''
    AIM: Create individual panels of the KDE distributions of k Feature Groups separated by environment. 
    This distribution is dictated by setting one of the following variables to True:
        * dsfr = dlogSFR
        * gser = g-band Sersic Index
        * w1ser = W1 Sersic Index
    If >1 is set to True, default will be dsfr, then w1ser. 
    '''
    
    #from scipy.stats import ks_2samp
    #from itertools import combinations

    if dsfr:
        prefix='delta_logsfr'
        xlims=(-6,2)
        bin_width = 0.3
    elif w1ser:
        prefix='CN_W1'
        xlims=(0,5)
        bin_width = 0.2
    elif gser:
        prefix='CN_g'
        xlims=(0,5)
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
    
    #using this, we can then create the xlabel. first, create LABEL dictionary
    label_dict = make_label_dictionary()
    
    #next, define xaxis label.
    xaxis_label = get_feature_label(colname, label_dict)

    #color/marker bookkeeping...
    colors, edgecolors, marker_shapes = marker_palette(feature_data)

    #grab number of unique feature groups
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
            kde = sns.kdeplot(x_feature, color=colors[k], label=f'FG{k}', ax=ax)

        ax.set_xlim(*xlims)
        ax.set_xlabel(xlabels[n])
        
        #add environment label to the plot panel
        ax.text(0.02, 0.95, env_name, transform=ax.transAxes, ha='left', va='top', fontsize=11)
        
        #for env==0 (first box), add legend.
        if n == 0:
            ax.legend(loc='upper right')

    plt.show()