# Robust and Scalable Bayesian Online Changepoint Detection

This repository contains all code and data needed to reproduce the results in the paper "Robust and Scalable Bayesian Online Changepoint Detection". 

## Reproducing experiments

- The folder `data` contains all the datasets used for the experiments.
- The folder `notebooks` contains notebooks to recreate all the experiments.
- The file `models.py` contains all the probability models implemented: DSM - Bayes and standard Bayes.
- The file `bocpd.py` contains the main function to run the algorithm.

## Requirements 
- Python == 3.9.*
- Numpy == 1.20.3
- SciPy == 1.7.1
- Jax == 0.4.1
