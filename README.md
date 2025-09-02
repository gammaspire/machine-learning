# machine-learning
A repository containing some applications of machine learning (ML) to both my current and side-research projects.
## predicting_M200
  - Using Random Forest Regression models to infer halo masses of galaxies with different density measurements to probe their environments.
  - The figure below plots the predicted vs. "true" (Tempel+2017) halo masses for galaxies in our sample, color-coded by the number of galaxies in its overall group/cluster. In order to probe how well the model predicts "true" halo masses, I calculate the median and 68% confidence intervals (1-sigma uncertainty) of "true" halo masses in bins of predicted halo mass with a width of 0.3 dex. For example, I take all of the "true" halo masses predicted to be 11 $\pm$ 0.15 dex, and then calculate the median and scatter of that distribution of "true" halo masses. A small width indicates the relative robustness of the model at inferring halo masses just based upon the environment measures I feed into it. 

<div align="center">
  <img alt="Comparison of 'true' halo masses of Tempel+2017 group galaxies and the values predicted by a trained Random Forest Regression machine learning model. The points are color-coded by the number of galaxies in the group/cluster." src="https://github.com/gammaspire/machine-learning/blob/main/rfr_example.png?raw=true">
</div>
