# DeepMediation: De-biased Deep Learning for Semi-parametric Causal Mediation Analysis
DeepMediation is an approach for semi-parametric causal mediation analysis to estimate the natrual (in)direct effects of a binary treatment on an outcome of interet. DeepMediation adopts the deep neural networks to estimate the nuisance parameters involved in the influence functions of the natrual (in)direct effects.
## Setup
The DeepMediation package will use the R package "keras" to establish the neural networks. Therefore, make sure that this package has 
been installed, which is available on R CRAN https://CRAN.R-project.org/package=keras. 

Use the following command in R to install the package:
```
install.packages(pkgs="keras")  # install the "keras" package
library(devtools)
install_github("siqixu/DeepMediation",ref="main") # install the "DeepMediation" package
```
## Usage
```
medDML_ann(y,d,m,x,trim=0.05,hyper_nn)
```
y: The outcome variable in causal mediation analysis.

d: The exposure variable in causal mediation analysis.

m: The mediator variable in causal mediation analysis.

x: The covariates in causal mediation analysis.

trim: The trimming rate. The observations with a propensity score smaller than the trimming rate will be removed from the analysis.

hyper_nn: The hyperparameters in neural network.
