'''
AIM: For a given Feature Group in either the main sequence population, transition population, or suppressed population of the dlogSFR vs. logMstar plot -- 
    * pull .jpg postage stamps of galaxies from Legacy Survey
    * compile into some sort of grid of plots
    * save as a scrollable PDF, with multiple galaxies per PDF page
    * mayhap add .pdf to hostable website?
'''

import os
import requests

import numpy as np
import pandas as pd
import itertools as it

from tqdm import tqdm
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed   #threading. yay.

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


#create global population type dictionary for separating the 
#main sequence, transition, and suppressed populations of galaxies.
POP_DICT = {1 : 'ms_pop', 2 : 'transition_pop', 3 : 'suppressed_pop'}

#create "session" so that I do not have to make repeated requests to pull from a website. this way, the channel only opens once.
SESSION = requests.Session()    


def legacy_link(ra,dec):
    return "https://www.legacysurvey.org/viewer/cutout.jpg?ra={:.4f}&dec={:.4f}&layer=ls-dr9&zoom=12".format(ra,dec)   



####################################################################
# BELOW ARE OBSOLETE FUNCTIONS TO PULL+SAVE .JPG FOR VFID GALAXIES #
####################################################################

def remove_ls_jpg(vfid):
    '''
    AIM: delete the galaxy file corresponding to the input VFID.
    '''
    path = f'data/images/{vfid}.jpg'
    try:
        os.remove(path)
    except FileNotFoundError:  #if file does not exist...ignore.
        print(f'Tried to remove {vfid} image, but {path} does not exist.')
        pass


def pull_ls_jpg(vfid,ra,dec):
    '''
    AIM: pull ls JPG image and write to disk; return path to saved image.
    '''
    path = os.path.join('data/images', f"{vfid}.jpg")
    
    if os.path.exists(path):   #if the image already exists, skip re-download
        return path
    
    url = legacy_link(ra, dec)
    
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()   #raises error if HTTP times out, etc.
    
    with open(path, "wb") as f:   #'wb' is 'write binary' -- used for non-text file writing
        f.write(r.content)
    
    return path


def get_galaxy_jpgs(df_subset):
    '''
    AIM: pull all Legacy Survey JPG images from the inputted dataframe of galaxies, saves to disk.
    Need RA, DEC, VFID.
    * companion function to pull_ls_jpg()
    '''
    
    if 'RA' not in df_subset.columns or 'DEC' not in df_subset.columns:
        print('Need "RA" and "DEC" columns!')
        return
        
    #quickly add data/images directory, if it does not already exist.
    os.makedirs('data/images', exist_ok=True)
    
    for row in df_subset.itertuples():
        
        #the 'utf-8' removes the prepended b in the string (e.g., b'VFID0000')
        vfid = str(row.VFID, 'utf-8')
        
        #pull JPG image and save to disk.
        pull_ls_jpg(vfid, row.RA, row.DEC)
        

########################################################
# PULL IMG NUMPY ARRAYS, CAN USE DIRECTLY FOR PLOTTING #
#    (I.E., DO NOT HAVE TO SAVE .JPG FILES TO DISK)    #
########################################################

def pull_ls_img(ra, dec):
    '''
    AIM: pull ls JPG from Legacy Survey Viewer, keep data MATRIX in memory; return np array of image.
    * advantage of this function is that I can avoid saving the .jpg files
    '''
    #define the url to the galaxy image
    url = legacy_link(ra, dec)
    
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()   #raises error if HTTP times out, etc.
    
    #grab the .jpg content pulled from the LS Viewer; convert to RGB matrices
    img = Image.open(BytesIO(r.content)).convert("RGB")
    
    #return numpy array equivalent of the RGB matrices
    return np.asarray(img)


def get_page_images(df_subset):
    """
    Return dict {vfid: np.ndarray image}.
    * companion function to pull_ls_img
    """
    imgs = {}
    
    #for every row in the df_subset, written as a TUPLE ((index=_, VFID=_, etc.)...much faster), 
    for row in df_subset.itertuples(): 
        
        #the 'utf-8' removes the prepended b in the string (e.g., b'VFID0000')
        vfid=str(row.VFID, 'utf-8')
        
        #grab the ls array from Legacy Survey Viewer, add to dictionary!
        imgs[vfid] = pull_ls_img(row.RA, row.DEC)
    
    return imgs


def get_page_images_parallel(df_subset, max_workers=2):
    '''
    Pull numpy arrays of Legacy Survey optical images "semi-concurrently." Issues multiple server requests at once!
    * Returns a dictionary mapping `vfid (str)` -> `image (np.ndarray)` where the array is an RGB image returned by `pull_ls_img`.
    * for Legacy Survey Viewer, need to use max_workers=2 to avoid "overloading the server" with requests. Lol.
    '''
    #initialize dictionary
    imgs = {}
    
    #define a fetch function that ThreadPoolExecutor will use in its loop. similar to the job scheduler!
    #returns galaxy VFID, image numpy array for the input row
    def fetch(row):
        vfid = str(row.VFID, 'utf-8')
        return vfid, pull_ls_img(row.RA, row.DEC)

    #initialize the threads...
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        
        #submit jobs for the rows
        futures = [ex.submit(fetch, row) for row in df_subset.itertuples()]
        
        #for the completed jobs (as they finish)...grab VFID and image numpy array and add to imgs dictionary
        for fut in as_completed(futures):
            vfid, img = fut.result()
            imgs[vfid] = img
    
    #return dictionary
    return imgs


def create_figure(df_subset, ncol=4, nrow=4, page_dict=None):
    '''
    pseudocode:
    - create 4x4 grid
    - for every set of axes, 
        - grab galaxy row from dataframe
        - load image data for that galaxy
        - add image to axis set
        - include VFID label in image
    '''
    
    if len(df_subset)>ncol*nrow:
        print(f'Maximum grid size is {ncol}x{nrow}. Do not input a df that has more than {ncol*nrow} rows!')
        return
    
    #pull the df_subset indices
    inds = df_subset.index
    
    #initialize the canvas
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(25,25))
    
    #trim the white space between panels
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for n, ax in enumerate(axes.flatten()):
        if n>(len(df_subset)-1):   #if we are beyond the number of rows in df, delete the axis and continue.
            fig.delaxes(ax)
            continue
    
        #otherwise, add the galaxy .jpg to the panel.
        
        #determine index corresponding to this current figure panel
        ind = inds[n]
        
        #the 'utf-8' removes the prepended b in the string (e.g., b'VFID0000')
        vfid_raw = df_subset['VFID'][ind]
        vfid=str(vfid_raw, 'utf-8')
        
        if page_dict is not None:
            img = page_dict[vfid]
        
        else:
            #if page_dict is None, then function will assume that JPG are saved to disk
            import matplotlib.image as mpimg
            impath = f'data/images/{vfid}.jpg'

            try:
                #read the image data
                img = mpimg.imread(impath)
            except:
                print(f'{impath} not found. Make sure you have the JPG images saved to disk.')
                return
            
        #display the image on the panel
        ax.imshow(img, aspect='auto')   #aspect='auto' ensures the figure honors the subplots_adjust arguments.
        
        #remove axes so the full grid looks a wee bit cleaner
        ax.axis('off')
        
        #add VFID, RA, DEC text (upper left)
        ax.text(0.05, 0.95, f"{vfid}\n\nRA = {df_subset['RA'][ind]:.3f}\nDEC = {df_subset['DEC'][ind]:.3f}", 
            transform=ax.transAxes, 
            fontsize=12, 
            color='white',
            fontweight='bold',
            va='top',
            ha='left',
            bbox=dict(facecolor='black', alpha=0.3))
        
        #add kcluster text (bottom left)
        ax.text(0.05, 0.05, f"Feature Group {df_subset['Feature Cluster'][ind]}", 
            transform=ax.transAxes, 
            fontsize=12, 
            color='white',
            fontweight='bold',
            va='bottom',
            ha='left',
            bbox=dict(facecolor='black', alpha=0.3))
        
        #if there is no page_dict, delete the saved jpg (no longer needed).
        if page_dict is None:
            remove_ls_jpg(vfid)
        
    #close figure to preserve space or something
    plt.close(fig)
    return fig


def create_subset(df, k=0, pop=0):
    '''
    AIM: create and apply boolean flags to extract 
    k   : int [0, 1, 2]
    pop : int [1, 2, 3]
        * 1 = main sequence population
        * 2 = transition population
        * 3 = suppressed population
    '''
    
    if type(k) is not int and type(pop) is not int:
        print('Both k and pop variables must be integers!')
        return
    
    df_k = (df['Feature Cluster']==k)
    df_pop = (df[POP_DICT[pop]])
    
    df_subset = df[df_k & df_pop]
    
    return df_subset


def create_pdf(figs, k=0, pop=0):
    '''
    AIM: convert all figures to a scrollable PDF format.
    Be sure to input a LIST of figures!
    '''
    
    with PdfPages(f'k{k}_{POP_DICT[pop]}.pdf') as pdf:
        
        for fig in figs:
            #save each figure as a new page
            pdf.savefig(fig, bbox_inches='tight')

            #annnd close the figure.
            plt.close(fig)
            
    return


def run_all(df, k_list, pop_list, jpg=False):
    '''
    AIM: from inputs, generate PDFs for all galaxy subsets.
    '''    
    #create unique pairs of k and pop. (0,1), (0,2), (0,3), (1,1), etc.
    combs = list(it.product(k_list, pop_list))
    
    #for every subset, divide into sets of <= 16 galaxies and generate a figure. add the figure to 
    #a figure master list.
    for k, pop in combs:
        
        #create subset
        subset = create_subset(df, k, pop)
        
        #grab list of df indices, convert to numpy array
        inds = subset.index.to_numpy()

        #define number of pages needed for the pdf. if the subset is not divisible by 16, need an extra page.
        #note: int() cuts off the decimal component; does NOT round up/down.
        npages = int(len(subset)/16) + (1 if len(subset)%16 !=0 else 0)  #can also use (len(subset)+15) // 16
        
        outpdf = f'k{k}_{POP_DICT[pop]}.pdf'
        
        with PdfPages(outpdf) as pdf:
            for pg in tqdm(range(npages), desc=f'k{k} {POP_DICT[pop]}'):
                
                start = pg * 16   #start index -- 0, 16, 32, 48, ...
                stop  = min((pg + 1) * 16, len(inds)) #end index -- 16, 32, 48, 64, ... (OR end index of inds)
                fig_indices = inds[start:stop] #isolate indices for the figures
                fig_galaxies = subset.loc[fig_indices] #pull the galaxy rows corresponding to these indices
                
                if jpg:
                    #pull images for this page
                    get_galaxy_jpgs(fig_galaxies)
                    page_dict=None
                else:
                    page_dict = get_page_images_parallel(fig_galaxies)
                
                #create the figure!
                fig = create_figure(fig_galaxies, page_dict=page_dict)
                
                #...save the figure
                pdf.savefig(fig, bbox_inches='tight')
                
                #and close the figure.
                plt.close(fig)