from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

from feature_utils import make_label_dictionary, get_feature_label


def marker_palette(feature_data):
    
    shapes = ['o', 's', '^', '*', 'D', 'v', 'X', '<', 'h', '>']
        
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
        cluster_colors = sns.color_palette('colorblind', len(feature_data['Feature Cluster'].unique()))
        edge_colors = cluster_colors
    
    marker_shapes = [shapes[i % len(shapes)] for i in range(k)]
    
    if -1 in feature_data['Feature Cluster'].unique():
        cluster_colors.insert(0, 'lightgray')
        edge_colors.insert(0, 'darkgray')
        marker_shapes.insert(0, 'o')
        #data_labels.insert(0, 'Noise')
    
    return cluster_colors, edge_colors, marker_shapes


def plot_silhouette(K, silhouettes):
    
    plt.figure(figsize=(8,6))
    plt.plot(K, silhouettes, 'o-', color='green')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show()
    
    
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
        ax = sns.scatterplot(x=x, y=y, data=feature_data[feature_data['Feature Cluster'] == -1], alpha=0.2,
                            color='lightgray', edgecolor='w', linewidth=0.4)
        sns.scatterplot(x=x, y=y, data=feature_data[feature_data['Feature Cluster'] != -1], hue='Feature Cluster',
                        palette=color_map, alpha=0.7, edgecolor='w', linewidth=0.4, ax=ax)
    
    plt.xlabel('Component One')
    plt.ylabel('Component Two')
    
    plt.title(f'Structural Clusters in 2D Space')
    plt.tight_layout()
    plt.show()


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
    
    
def plot_env_fraction(feature_data, main_only=True):
    '''
    main_only --> plot cluster, rich group, poor group, filament, field environments only (no nuance)
    '''
    from stat_utils import binomial_uncertainty
    
    #define feature group colors
    colors, edgecolors, marker_shapes = marker_palette(feature_data)

    #unpack the flags
    clusflag = feature_data['cluster_member']
    rgflag = feature_data['rich_group_memb']
    pgflag = feature_data['poor_group_memb']
    filflag = feature_data['filament_member']
    fieldflag = feature_data['pure_field']
    
    #defining more nuanced flags
    clus_only = (clusflag) & (~filflag)
    fil_clus = (filflag) & (clusflag) & (~rgflag) & (~pgflag)
    fil_only = (filflag) & (~clusflag) & (~rgflag) & (~pgflag)
    rg_only = (rgflag) & (~filflag)
    pg_only = (pgflag) & (~filflag)
    
    #create array of k values
    try:
        k = np.unique(feature_data['Feature Cluster'])
    except:
        print('"Feature Cluster" column not found. Please run k-means or HDBSCAN clustering before continuing!')
        return
    
    #create dictionaries!
    unique_clusters = sorted(k)
    color_map  = {c: colors[i]        for i, c in enumerate(unique_clusters)}
    edge_map   = {c: edgecolors[i]    for i, c in enumerate(unique_clusters)}
    shape_map  = {c: marker_shapes[i] for i, c in enumerate(unique_clusters)}
    
    #set up the flags, data, indices, x-axis environment names
    env_names = np.array(['Pure Cluster',
                 'All Cluster', 
                 'Filament\n&\nCluster',
                 'Filament\n&\nRich Group',
                 'Pure Rich \n Group',
                 'All Filament\n(PG+RG+CLUS)',
                 'Filament \n & \n Poor Group',
                 'Pure Poor \n Group',
                 'Pure Filament',
                 'Pure Field'])
        
    #place the flags in a neat and tidy list
    flags = [clus_only, clusflag, fil_clus, rgflag, rg_only, filflag, pgflag, pg_only, fil_only, fieldflag]
    
    #if the user only wants the main five environments, trim the lists accordingly
    if main_only:
        flags=[clusflag, rgflag, pgflag, filflag, fieldflag]
        env_names = ['Cluster', 'Rich Group', 'Poor Group', 'Filament', 'Pure Field']
    
    #place the environment data in a neat and tidy list
    env_galaxies = [feature_data[flag] for flag in flags]
    index = np.arange(1,len(env_names)+1,1)
    
    #initialize the figure
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    
    #create storage variables so that I can connect the dots when the loop finishes.
    line_x = {k_cluster: [] for k_cluster in k}
    line_y = {k_cluster: [] for k_cluster in k}
    
    #for every environment, plot its corresponding fraction and uncertainty for every feature group
    for i, env in enumerate(env_galaxies):
        
        for k_cluster in k:
            
            #define label for legend, but only for the first environment (to avoid redundancies)
            label_ = None
            if i==0:
                label_ = f'Feature Group {k_cluster}' if k_cluster != -1 else 'Noise Galaxies'
            
            #get the total number of galaxies in the feature cluster
            total = len(feature_data.loc[feature_data['Feature Cluster'] == k_cluster])
            
            #pull the galaxies in env that are also in k_cluster
            feature_members = env.loc[env['Feature Cluster'] == k_cluster]
            subset = len(feature_members)   #fraction of env galaxies in k_cluster
            
            #calculate the fraction of feature group galaxies in a given environment
            #the subset is the number of env environment galaxies in the feature group
            #the total is ALL of the galaxies in the feature group
                #effectively, this fraction is a probability of plucking a galaxy in X environment 
                #if it belongs to k feature group
            
            #if total = 0...no use in including the data.
            if total == 0:
                continue

            fraction = subset / total
            unc = binomial_uncertainty(subset, total)
            
            #store the line variables!
            line_x[k_cluster].append(index[i])
            line_y[k_cluster].append(fraction)
            
            ax.scatter(index[i], fraction,  color=color_map[k_cluster], label=label_, s=80, 
                       edgecolor=edge_map[k_cluster], marker=shape_map[k_cluster], zorder=3)
            ax.errorbar(index[i] ,fraction, yerr=unc, color=color_map[k_cluster],
                        alpha=0.5, lw=2.5, zorder=2)  #do not need fmt='None' since we are only
                                                      #plotting one data point per iteration

            print(f'Environment {i} cluster {k_cluster}: fraction {fraction:.3f}+/-{unc:.3f}')
    
    #connect the dots using the stored values!
    for k_cluster in unique_clusters:
        ax.plot(line_x[k_cluster], line_y[k_cluster], color=color_map[k_cluster], 
                linewidth=2.2, alpha=0.8, zorder=1)
    
    ax.set_xticks(index, env_names, rotation=45, fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(alpha=0.2)
    ax.set_ylabel('Fraction of Galaxies',fontsize=17)

    ax.set_title('Fraction of Galaxy Environments per Feature Group')
    
    ax.legend(loc='upper left')
    
    plt.show()


def plot_group_features(median_data, layout_dict=None):
    '''
    AIM: create multiple subplots showing each group's features and their associated uncertainties (taken from bootstrapping)
    * median_data should comprise a dataframe table of feature medians and lower+upper uncertainties for each of the feature groups.
    * layout_dict --> a python dictionary comprising the coordinates on a 3x3 grid where each feature 
        will be plotted, as well as the column name of that feature in median_data
            * If None, defaults to W1, W3, and g-band Re and nser; Size Ratio, NUV-r and W1-W3 colors
    *
    '''
    
    from math import ceil
    
    #extract the colors...
    cluster_colors, edge_colors, marker_shapes = marker_palette(median_data)
    
    #extract the label dictionary for the y-axis!
    label_dict = make_label_dictionary()
    
    #in the median_data table, there is one row for every kth feature group
    k = len(median_data)
    
    #intended layout dictionary for subpl0ts.
    if layout_dict is None or not isinstance(layout_dict, dict):
        
        message = 'Using default layout dictionary for median group feature subplots!'
        
        print('#'*len(message))
        print(message)
        print('#'*len(message))
        
        layout_dict =  {(0, 0): 'CRE_g_unscaled',
                        (0, 1): 'CRE_W1-fixBA_unscaled',
                        (0, 2): 'CRE_W3-fixBA_unscaled',
                        (1, 0): 'CN_g_unscaled',
                        (1, 1): 'CN_W1-fixBA_unscaled',
                        (1, 2): 'CN_W3-fixBA_unscaled',
                        (2, 0): 'Size Ratio',
                        (2, 1): 'NUV_r',
                        (2, 2): 'W1_W3'}
        ncol = 3
        nrow = 3
    
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
    
    #determine the unique cluster IDs (ignore noise)
    unique_clusters = sorted(c for c in median_data['Feature Cluster'].unique() if c != -1)

    #INITIATE
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(fig_width, fig_height), constrained_layout=True)
    
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
            im = ax.scatter(k_cluster, median, s=80, 
                            edgecolor=edge_colors[k_cluster], marker=marker_shapes[k_cluster],
                            color=cluster_colors[k_cluster], zorder=2, label=f'Feature Group {k_cluster}')
            
            #plot the error bars
            err = ax.plot([k_cluster, k_cluster], [median-low_err, median+upp_err], 
                          color=edge_colors[k_cluster], zorder=1)
        
        #assign row limits
        if 'CN' in med_label:
            ax.set_ylim(0,2)
        elif 'CRE' in med_label:
            ax.set_ylim(0,5)
        
        #set appropriate x-limits
        ax.set_xlim(min(unique_clusters)-0.5, max(unique_clusters)+0.5)
        
        #make x-axis increments of 1, since k is an integer!
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=1.0))
        
        ax.set_xlabel('Feature Group [k]')
        ax.set_ylabel(get_feature_label(med_label, label_dict))   #need the fancy schmancy name!
        ax.grid(alpha=0.1)
    
    plt.show()
    return




def plot_sfrmstar(feature_data):
    
    #need logsfr, logmstar, and Feature Cluster columns!
    if 'logsfr' not in feature_data.columns or 'logmstar' not in feature_data.columns:
        return 'Need "logmstar" and "logsfr" columns to use this function!'
    if 'Feature Cluster' not in feature_data.columns:
        return 'Need "Feature Cluster" column to use this function!'
    
    palette, _, _ = marker_palette(feature_data)
    
    g = sns.JointGrid(data=feature_data, x="logmstar", y="logsfr", height=5)

    # ---- MAIN SCATTER ----
    g.plot_joint(sns.scatterplot, data=feature_data, hue="Feature Cluster", 
                 palette=palette, alpha=0.4, edgecolor="w", linewidth=0.3)
    g.ax_joint.set_xlim(5,)
    g.ax_joint.set_ylim(-7.5,)
    
    # ---- KDE MARGINALS (the histogram distributions) ----
    for k, color in enumerate(palette):
        subset = feature_data[(feature_data["Feature Cluster"] == k)]

        #top marginal (logmstar)
        sns.kdeplot(x=subset["logmstar"], ax=g.ax_marg_x, color=color, fill=True, alpha=0.3, linewidth=1.2)

        #right marginal (logsfr)
        sns.kdeplot(y=subset["logsfr"], ax=g.ax_marg_y, color=color, fill=True, alpha=0.3, linewidth=1.2)
    
    g.fig.set_size_inches(12, 6)
    
    plt.show()
    return