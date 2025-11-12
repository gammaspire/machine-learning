from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def cluster_color_palette(feature_data):
    
    if len(feature_data['Feature Cluster'].unique()) == 3:
        cluster_colors = ['darkorange','seagreen','deeppink']
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
    

def plot_pca_components(feature_data, features, pca):
    """
    Visualize the PCA loadings (feature contributions) for the first two components.
    Includes arrow scaling, clearer labels, and optional auto-label adjustment.
    """
    from adjustText import adjust_text
    n_features = len(features)
    
    #vector components of each feature in PCA space
    components = pca.components_
    
    text_labels = []
    cmap = plt.cm.viridis(np.linspace(0, 1, n_features))
    
    plt.figure(figsize=(8, 6))
    
    for i, feature in enumerate(features):
        x, y = components[0, i], components[1, i]
        plt.arrow(0, 0, x, y, head_width=0.025, head_length=0.04, fc=cmap[i], ec=cmap[i], linewidth=1.8)
        text = plt.text(x*1.2, y*1.2, feature, color=cmap[i], ha='center', va='center')
        text_labels.append(text)
    
    #automatically adjust overlapping text labels
    adjust_text(text_labels,  arrowprops=dict(arrowstyle='-', color=None, lw=0.5, alpha=0.6))
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Loadings Plot")
    plt.grid(alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)
    plt.show()


def plot_corner(feature_data, features):
    #suppress those WARNINGS PLS
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    cluster_colors = cluster_color_palette(feature_data)
    
    features=['CRE_W1-fixBA', 'CN_W1-fixBA', 'CRE_W3-fixBA', 'CN_W3-fixBA']
    
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
    for i, ax in enumerate(axes_flat):
    #    if i not in ax_dict:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    g._legend.remove()
    g.fig.suptitle('Feature Clusters in Physical Space', y=1.02)
    plt.show()