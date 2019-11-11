## Overview
This repo contains the scripts to reproduce the simulation results for mutual information correction manuscript.

## Prerequisite software
+ python3 (>=3.4.3)
+ pickle
+ [numpy](http://www.numpy.org/)
+ [tqdm](https://tqdm.github.io/)
+ [scipy](https://www.scipy.org/)
+ [scikit-learn](https://scikit-learn.org/stable/)
+ [pandas](https://pandas.pydata.org/)
+ [matplotlib](https://matplotlib.org/)
+ [seaborn](https://seaborn.pydata.org/)

## Simulation for semi-discrete case
Running the following scripts will simulate observations in the semi-discrete case, compute the baseline and corrected mutual information, and plot the comparison figure. The comparison result and the figure will be stored in `results` folder.
```
python src/simulation_half.py
```

## Simulation for continuous case
Running the following script will simulate the observations in the continuous case, compute the baseline and corrected mutual information based on KDE with the optimal bandwidth, and plot the comparison figure. The estimated mutual information and the figure will be stored in `results` folder.
```
python src/simulation_continuous.py
```
