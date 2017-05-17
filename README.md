# modRSW_EnKF
## An idealised convective-scale forecast-assimilation framework

This repository aims to facilitate the transfer of knowledge and continued use of an idelaised convective-scale forecast -assimilation system. It should contain sufficient instruction for users to implement and adapt the source code (briefly comprising Python scripts for the numerical solver, idealised forecast-assimilation routines, plotting and data analysis). This source code was developed during TK's PhD, a pdf is available [here](http://etheses.whiterose.ac.uk/17269/).

***CAVEAT: this is not a black-box model and should accordingly be used with care and curiosity!***

How to get started with the source code and files is outlined below. There are some model-only and DA test cases to check things are up and running properly.

For further details, including what should be modified in the scripts for different experiemtnal set-ups, please see the pdf document (TO DO).

----

## Getting started
### Versions
All of the source code is written in Python and relies heavily on numpy, amongst others. The plotting routines require matplotlib. The versions used in the development are tabled below. Other versions may work, but should not be relied upon.

Software      | Version
------------- | -------------
Python  | 2.7.7
Matplotlib  | 1.3.1
Numpy  | 1.8.1

To check python version, from the terminal:
```
python --version
```

To check numpy version, open python in the terminal, import it and use the version attribute:
```
>>> import numpy
>>> numpy.__version__
```
Same for Matplotlib.

### Downloading 
Direct download: 
* click on the download link on the repository homepage [https://github.com/tkent198/modRSW_EnKF](https://github.com/tkent198/modRSW_EnKF)

Clone: 
* the most up-to-date version is hosted here by github. From the command line
```
git clone https://github.com/tkent198/modRSW_EnKF.git
```

### Running the code
To run a script (e.g., `fname.py`) from the terminal:
```
python fname.py
```

## Test cases

### Model only: integration and dynamics

### Simple forecast-assimilation experiment

## Brief overview of files

```parameters.py```: List of all parameters pertaining to the model itself and the forecast-assimilation framework.

### Model only

```run_modRSW.py```: Runs modRSW with no assimilation, plotting at given intervals. Use to check model-only dynamics.

```init_cond_modRSW.py```: Functions for different initial conditions, detailed within.

```f_modRSW.py```: Functions required for the numerical integration of the modRSW model with and without topography.
* make_grid()           : generates mesh for given length and gridcell number
* NCPflux_topog()       : calculates numerical flux as per the theory of Kent et al., 2017
* time_step()           : calculates stable time step for integration
* step_forward_topog()  : integrates forward one time step (forward euler) using NCP-Audusse
* heaviside()           : vector-aware implementation of heaviside (also works for scalars

### Assimilation framework

```main_p.py```: main run script for idealised forecast-assimilation experiments
* specifies outer loop parameters, generates truth , make relevant directory `dirname` etc., and ...
* Enter EnKF outer loop using ```subr_enkf_modRSW_p.py```


```create_readme.py```: function creates readme.txt file for summarising experiments, saved in `dirname` to accompany outputted data from main run script and EnKF subroutine.



```subr_enkf_modRSW_p.py```: Subroutine accessed by ```main_p``` for performing EnKF given outer-loop parameters in ```main_p```

```f_enkf_modRSW.py```: Collection of functions related to the assimilation specifically, incl.:
* generate_truth()      : simulates truth trajectory at given resolution and stores run at given observing times.
* analysis_step_enkf()  : performs perturbed obs. enkf analysis step, returns updated ensemble and output data for saving.
* gasp_cohn()           : Gaspari-Cohn taper function for ensemble localisation.


```localisation.py```: Initial investigations and calculation of localisation taper function and matrices using ```gasp_cohn```.

```model_error_Q.py```: Initial investigations of additive inflation and model error, incl Q matrix computation and sampling

### Plotting and data analysis

```plot_truth.py```: A few checks on characteristics of the nature run: plot all trajectories, e.g., check height and rain extremes.

```plot_func_x.py```: Plotting routine: loads saved data in specific directories and produces domain plots at a given assimilation time. To use, specify (1) `dir_name`, (2) combination of parameters `ijk`, (3) time level ```T = time_vec[ii]```, i.e., choose ```ii```.

```plot_func_t.py```: Produces domain-averaged error, CPRS, and OI plots as a function of time.  To use, specify (1) `dir_name`, (2) combination of parameters `ijk`.

```analysis_diag_stats.py```: > ave_stats: function calculates summary statistics `DIAGS.npy` for each experiment `ijk` in given `dirname` and returns averaged spread, error, crps, OI for forecast and analysis ensembles.

```compare_stats.py```: Each directory has i*j*k experiments with different parameter combinations. This script looops through ave_stats `ijk` and plots 2d summary matrix a la Poterjoy and Zhang for comparison. If `DIAGS.npy` exists, straight to plotting. If not, calculate statistics and save before plotting.

```crps_calc_fun.py```: Function: calculate the CRPS of an on ensemble of forecast variables following the theory of Hersbach (2000), and as applied in Bowler et al (2016) and DAESR5.
    
```run_modRSW_EFS.py```: Script runs ensemble forecasts of length Tfc, initialised from analysed ensembles at a given time T0. Saves forecasts as `X_EFS_array_Tn.npy` for n = T0, to be used e.g. to calculate error growth statistics.
    
```EFS_stats.py```: Computes and plots error growth, crps for the EFS data produced in ```run_modRSW_EFS.py```. Also computes and saves error doubling times, to be used in ```err_doub_hist```.
    
```err_doub_hist.py```: Plots error doubling time histograms from saved data ```err_doub_Tn.npy```.

### .npy data

```U_tr_array.npy```: nature run for current set-up for U = [h,u,r].

```B_tr.npy```: topography projected on to 'nature' resolution.

```Q_offline.npy```: a static Q matrix for additive inflation, generated in ```offlineQ.py```.

### Directories
