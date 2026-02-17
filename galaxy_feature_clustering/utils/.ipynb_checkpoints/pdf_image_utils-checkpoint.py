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


def legacy_link(ra,dec,size=256,pixscale=0.7,layer='ls-dr9'):
    return f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&layer={layer}&size={size}&pixscale={pixscale}"


############################################################################
# BELOW ARE OBSOLETE FUNCTIONS TO PULL+SAVE OPTICAL .JPG FOR VFID GALAXIES #
############################################################################

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
    
    for row in df_subset.itertuples(index=False):
        
        #the 'utf-8' removes the prepended b in the string (e.g., b'VFID0000')
        vfid_raw = row.VFID
        vfid = vfid_raw.decode("utf-8") if isinstance(vfid_raw, (bytes, bytearray)) else str(vfid_raw)
        
        #pull JPG image and save to disk.
        pull_ls_jpg(vfid, row.RA, row.DEC)
        

########################################################
# PULL IMG NUMPY ARRAYS, CAN USE DIRECTLY FOR PLOTTING #
#    (I.E., DO NOT HAVE TO SAVE .JPG FILES TO DISK)    #
########################################################

def pull_ls_img(ra, dec, layer='ls-dr9'):
    '''
    AIM: pull optical or infrared JPG from Legacy Survey Viewer, keep data MATRIX in memory; return np array of image.
    * advantage of this function is that I can avoid saving the .jpg files
    * layer 'ls-dr9' --> optical (grz)
    * layer 'unwise-neo11' --> infrared (W1+W2)
    '''
    #define the url to the galaxy image
    url = legacy_link(ra, dec, layer=layer)
    
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()   #raises error if HTTP times out, etc.
    
    #grab the .jpg content pulled from the LS Viewer; convert to RGB matrices
    img = Image.open(BytesIO(r.content)).convert("RGB")
    
    #return numpy array equivalent of the RGB matrices
    return np.asarray(img)


def get_page_images(df_subset):
    """
    Return dict {vfid: np.ndarray image}.
    * runs 'in series' -- one galaxy at a time.
    * companion function to pull_ls_img
    """
    imgs = {}
    
    #for every row in the df_subset, written as a TUPLE ((index=_, VFID=_, etc.)...much faster), 
    for row in df_subset.itertuples(index=False): 
        
        #the 'utf-8' removes the prepended b in the string (e.g., b'VFID0000')
        vfid_raw = row.VFID
        vfid = vfid_raw.decode("utf-8") if isinstance(vfid_raw, (bytes, bytearray)) else str(vfid_raw)
        
        #grab the optical and infrared image matrices from Legacy Survey Viewer
        img_optical = pull_ls_img(row.RA, row.DEC, layer='ls-dr9')
        img_infrared = pull_ls_img(row.RA, row.DEC, layer='unwise-neo11')
        
        #concatenate into a 256 (rows) x 512 (columns) x 3 (RGB) matrix
        comb = np.concatenate([img_optical, img_infrared], axis=1)
        
        #add to imgs dictionary!
        imgs[vfid] = comb

    return imgs


def get_page_images_parallel(df_subset, max_workers=2):
    '''
    Pull numpy arrays of Legacy Survey optical images "semi-concurrently." Issues multiple server requests at once!
    * for Legacy Survey Viewer, need to use max_workers=2 to avoid "overloading the server" with requests. Lol.
    
    * Return dict {vfid: np.ndarray image}.
    * companion function to pull_ls_op_img and pull_ls_w1_img
    '''
    #initialize dictionary
    imgs = {}
    
    #define a fetch function that ThreadPoolExecutor will use in its loop. similar to the job scheduler!
    #returns galaxy VFID, image numpy array for the input row
    def fetch(row):
        vfid_raw = row.VFID
        vfid = vfid_raw.decode("utf-8") if isinstance(vfid_raw, (bytes, bytearray)) else str(vfid_raw)
        return vfid, pull_ls_img(row.RA, row.DEC, layer='ls-dr9'), pull_ls_img(row.RA, row.DEC, layer='unwise-neo11')

    #initialize the threads...
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        
        #submit jobs for the rows
        futures = [ex.submit(fetch, row) for row in df_subset.itertuples(index=False)]
        
        #for the completed jobs (as they finish)...grab VFID and image numpy array and add to imgs dictionary
        for fut in as_completed(futures):
            
            vfid, img1, img2 = fut.result()
            
            #concatenate into a 256 (rows) x 512 (columns) x 3 (RGB) matrix
            combo = np.concatenate([img1, img2], axis=1)
    
            imgs[vfid] = combo
            
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
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(40,20))
    
    #trim the white space between panels
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for n, ax in enumerate(axes.flatten()):
        if n>=len(df_subset):   #if we are beyond the number of rows in df, delete the axis and continue.
            fig.delaxes(ax)
            continue
            
        #determine index corresponding to this current figure panel
        ind = inds[n]
        
        #the 'utf-8' removes the prepended b in the string (e.g., b'VFID0000')
        vfid_raw = df_subset['VFID'][ind]
        vfid = vfid_raw.decode("utf-8") if isinstance(vfid_raw, (bytes, bytearray)) else str(vfid_raw)
        
        if page_dict is not None:
            img = page_dict[vfid]
        
        else:
            #if either dictionary is None, then function will assume that JPGs are saved to disk
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


def run_all(df, k_list, pop_list, jpg=False, max_workers=2):
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
        
        outpdf = f'pdfs/k{k}_{POP_DICT[pop]}.pdf'
        
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
                    page_dict = get_page_images_parallel(fig_galaxies, max_workers=max_workers)
                
                #create the figure!
                fig = create_figure(fig_galaxies, page_dict=page_dict)
                
                #...save the figure
                pdf.savefig(fig, bbox_inches='tight')
                
                #and close the figure.
                plt.close(fig)