####################################################################
## 			 FIXED PARAMETERS   			  ##
####################################################################
'''
Author: T. Kent  (tkent198@gmail.com)
List of fixed parameters for model integration and EnKF.
'''

import numpy as np

''' MODEL PARAMETERS '''

Neq = 3 # number of equations in system (3 with topography, 4 with rotation_
L = 1.0 # length of domain (non-dim.)

Nk_fc = 200                                 # forecast resolution
dres = 4                                     # refinement factor for truth gridsize
Nk_tr = dres*Nk_fc                           # truth resolution

cfl_fc = 1. # Courant Friedrichs Lewy number for time stepping
cfl_tr = 1.

Ro = 'Inf'          # Rossby no. Ro ~ V0/(f*L0)
Fr = 1.1            # froude no.
g = Fr**(-2) 		# effective gravity, determined by scaling.
A = 0.1
V = 1.

# threshold heights
H0 = 1.0
Hc = 1.02
Hr = 1.05

# remaining parameters related to hr
beta = 0.2
alpha2 = 10
c2 = g*Hr
cc2 = 4*0.025*c2

''' FILTER PARAMETERS '''

n_ens = 40                              # number of ensembles
Nmeas = 48                               # number of cycles
tn = 0.0                                # initial time
#tmax = Nmeas*0.144                      # end time = Nmeas*1hr real time
#dtmeasure = tmax/Nmeas                  # length of each window
dtmeasure = 0.144
tmax = Nmeas*dtmeasure
tmeasure = dtmeasure
assim_time = np.linspace(tn,tmax,Nmeas+1) # vector of times when system is observed
#
sig_ic = [0.05,0.05,0.0]                # initial ens perturbations [h,hu,hr]
ob_noise = [0.1,0.05,0.005]            # ob noise for [h,hu,hr]
o_d = 50                                # ob density: observe every o_d elements
#

''' OUTER LOOP'''
'''
Parameters for outer loop are specified in main_p.py 
loc     : localisation scale
add_inf : additive infaltaion factor
inf     : mult. inflation factor
'''

##################################################################
#			END OF PROGRAM				 #
##################################################################
