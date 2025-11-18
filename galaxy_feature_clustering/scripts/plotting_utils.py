from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def cluster_color_palette(feature_data):
    
    if len(feature_data['Feature Cluster'].unique()) == 3:
        cluster_colors = ['darkorange','seagreen','deeppink']
    elif len(feature_data['Feature Cluster'].unique()) == 4:
        cluster_colors = ['darkorange','seagreen','deeppink','indigo']
    else:
        cluster_colors = sns.color_palette('colorblind', len(feature_data['Feature Cluster'].unique()))
    
    return cluster_colors


def plot_silhouette(K, silhouettes):
    
    plt.figure(figsize=(8,6))
    plt.plot(K, silhouettes, 'o-', color='green')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show()
    
    
def plot_kmeans_clusters(feature_data, x=None, y=None, PCA=False):
    
    cluster_colors = cluster_color_palette(feature_data)
    
    if x is None and not PCA:
        print('Unable to generate plot! Please either specify x, y columns or set PCA=True.')
        return
    
    x = 'PCA1' if PCA else x
    y = 'PCA2' if PCA else y
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=feature_data, x=x, y=y, hue='Feature Cluster',
                    palette=cluster_colors, alpha=0.7, edgecolor='w', linewidth=0.4)
    plt.title(f'K-Means Structural Clusters in PCA Space')
    plt.tight_layout()
    plt.show()
    

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

def plot_pca_components(feature_data, features, pca, cmap_name='tab20'):
    '''
    * Visualize PCA feature loadings for the first two components.
    * Features are sorted by total loading strength (or rather, the magnitude of their contribution.
    '''
    
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
    
    cluster_colors = cluster_color_palette(feature_data)
    
    #we want to plot the UNSCALED features!
    #I am also dictating the plotted features. not oops.
    if features is None:
        features=['CRE_W1-fixBA_unscaled', 'CN_W1-fixBA_unscaled', 
                  'CRE_W3-fixBA_unscaled', 'CN_W3-fixBA_unscaled',
                  'CRE_r_unscaled', 'CN_r_unscaled']
        if 'AVG_RE_gr_unscaled' in feature_data.columns:
            features[4] = 'AVG_RE_gr_unscaled'
        if 'AVG_RE_W1W2_unscaled' in feature_data.columns:
            features[0] = 'AVG_RE_W1W2_unscaled'
        
    
    g = sns.pairplot(feature_data, vars=features, hue='Feature Cluster', 
                     palette=cluster_colors, corner=True,
                     plot_kws={'alpha': 0.6, 's': 20}, diag_kind=None)  #'kde')

    #pull the axes from the corner plot
    axes_flat = [ax for ax in g.axes.flat if ax is not None]
    
    #each diagonal (axis 0, 2, 5, 9, ...) shows the variable feature_data[feature]
        #that colname is index 0, 1, 2, 3, ... as you go down the diagonal
    #this dictionary converts the axis number of the diagonal with the index of the corresponding colname
    #ax_dict={0:'0', 2:'1', 5:'2', 9:'3', 14:'4', 20:'5', 27:'6'}
    
    #for axes NOT on the diagonal...use log-log space
    #for i, ax in enumerate(axes_flat):
    #    if i not in ax_dict:
    #    ax.set_xscale('log')
    #    ax.set_yscale('log')
    
    g._legend.remove()
    g.fig.suptitle('Feature Clusters in Physical Space', y=1.02)
    plt.show()
