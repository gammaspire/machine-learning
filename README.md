# machine-learning
A repository containing some applications of machine learning (ML) to both my current and side-research projects.
## predicting_M200
  - Using Random Forest Regression models to infer halo masses of galaxies with different density measurements to probe their environments.
  - The first figure plots the predicted vs. "true" (Tempel+2017) halo masses for galaxies in our sample, color-coded by the number of galaxies in its overall group/cluster. In order to probe how well the model predicts "true" halo masses, I calculate the median and 68% confidence intervals (1-sigma uncertainty) of "true" halo masses in bins of predicted halo mass with a width of 0.3 dex. For example, I take all of the "true" halo masses predicted to be 11 $\pm$ 0.15 dex, and then calculate the median and scatter of that distribution of "true" halo masses. A small width indicates the relative robustness of the model at inferring halo masses just based upon the environment measures I feed into it. 

<div align="center">
  <img alt="Comparison of 'true' halo masses of Tempel+2017 group galaxies and the values predicted by a trained Random Forest Regression machine learning model. The points are color-coded by the number of galaxies in the group/cluster." src="https://github.com/gammaspire/machine-learning/blob/main/images/rfr_example.png?raw=true">
</div>

- The second figure below compares the performance of the Random Forest Regression model with that of any single environment density measurement. The latter model involves fitting a line to the feature vs. Tempel+2017 halo mass plot and using the equation of this line to predict halo mass. Each line in the plot is the result of calculating the 68% confidence intervals for these simple-model predicted halo masses, as well as a solid line tracing the CIs for the ML model. The ML model performs either on-par with or more robustly than the cases of a simple linear model fit to a single feature. In fact, in the 12-14 predicted log(M200) regime, where most of the sample galaxies reside, the model's predictions are more robust by a factor of around 0.5 dex.

<div align="center">
  <img alt="Comparison of 68% Confidence Intervals for the predicted halo masses of Tempel+2017 group galaxies. Each line traces the CIs for either a simple linear fit model or the Random Forest Regression machine learning model." src="https://github.com/gammaspire/machine-learning/blob/main/images/parameter_linear_fits_CIs.png?raw=true">
</div>