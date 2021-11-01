\name{medDML_ann}
\alias{medDML_ann}
\title{DeepMediation: De-biased Deep Learning for Semi-parametric Causal Mediation Analysis}
\usage{
medDML_ann(y,d,m,x,trim=0.05,hyper_nn)
}
\description{De-biased Deep Learning for Semi-parametric Causal Mediation Analysis
}

\arguments{
  \item{y}{The outcome variable in causal mediation analysis.}
  \item{d}{The exposure variable in causal mediation analysis.}
  \item{m}{The mediator variable in causal mediation analysis.}
  \item{x}{The covariates in causal mediation analysis.}
  \item{trim}{The trimming rate to prevent propensity score from being zero.}
  \item{hyper_nn}{The hyperparameters in neural network.}
}
