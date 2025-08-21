# machine-learning
A repository containing some applications of machine learning (ML) to my research projects.

## Predicting_M200
- As part of the WISESize Project, one aim I had was to characterize environments for our sample of about 30,000 galaxies. One way to do so is to find each galaxy's halo mass (M200) and associate that with environment. We loosely define environment according to 
    * Field: log(M200) < 12.5
    * Group: 12.5 < log(M200) < 14.0
    * Cluster: log(M200) > 14.0
- However, not all of the galaxies in our sample had an M200 value. After I matched our WISESize table to the group catalog from Tempel+2017, which did contain halo masses for each of their group galaxies, 25% of our sample did not have a corresponding Tempel+2017 entry. 
- I decided to infer these galaxies' M200 by trying to find relationships between the M200 of the matched galaxies and something like $\Sigma_M$ or $\Sigma_5$ (the 2D projected mass density and fifth-nearest-neighbor 2D projected density, respectively), which probe a galaxy's immediate environment; however, each metric introduced too much scatter to be reliable for interence purposes.
- Instead, I created a Random Forest ML model to derive halo masses using an array of these environment probes at different scales (close to vs. far from the galaxy, or the size of the "slice" of space that I flattened into a 2D plane).