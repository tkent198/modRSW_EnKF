#######################################################################
# Perturbed obs EnKF for 1.5D SWEs with rain variable and topography
#               (T. Kent: mmtk@leeds.ac.uk)
#######################################################################

'''
4/2/2016: 
main run script for batch-processing of EnKF jobs. 
Define outer-loop through parameter space and run the EnKF subroutine for each case.

p: 
> truth generated outside of outer loop as this is the same for all experiments (as in v2)
> make use of pars_enkf (as in v2)
> uses subroutine <subr_enkf_modRSW_p> that parallelises ensemble forecasts using multiprocessing module
'''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from parameters import *
from f_modRSW import make_grid 
from f_enkf_modRSW import generate_truth
from init_cond_modRSW import init_cond_topog4, init_cond_topog_cos
from create_readme import create_readme
from subr_enkf_modRSW_p import run_enkf

#################################################################
# create directory for output
#################################################################
cwd = os.getcwd()
dirname = str('/experi')
dirn = str(cwd+dirname)
#check if dir exixts, if not make it
try:
    os.makedirs(dirn)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

# parameters for outer loop
loc = [1e-10, 1., 2.5, 4.]
add_inf = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8]
inf = [1.01, 1.05, 1.1]



#################################################################
# CHOOSE INITIAL PROFILE FROM init_cond_modRSW:
#################################################################
ic = init_cond_topog_cos

##################################################################    
# Mesh generation and IC for truth 
##################################################################
tr_grid =  make_grid(Nk_tr,L) # truth
Kk_tr = tr_grid[0] 
x_tr = tr_grid[1] 

### Truth ic
U0_tr, B_tr = ic(x_tr,Nk_tr,Neq,H0,L,A,V)
np.save(str(cwd+dirname+'/B_tr'),B_tr) #save topog for plotting

U_tr_array = np.empty([Neq,Nk_tr,Nmeas+1])
U_tr_array[:,:,0] = U0_tr

f_path_name = str(cwd+dirname+'/U_tr_array.npy')

try:
    ' *** Loading truth trajectory... *** '
    U_tr_array = np.load(f_path_name)
except:
    print ' *** Generating truth trajectory... *** '
    U_tr_array = generate_truth(U_tr_array, B_tr, Nk_tr, tr_grid, assim_time, f_path_name)


##################################################################    
# EnKF: outer loop 
##################################################################
print ' '
print ' ------- ENTERING EnKF OUTER LOOP ------- '  
print ' '
for i in range(3,len(loc)):
    for j in range(0,len(add_inf)):
        for k in range(0,len(inf)):
            run_enkf(i,j,k, loc[i], add_inf[j], inf[k] , ic , U_tr_array, dirname)
    

print ' '   
print ' --------- FINISHED OUTER LOOP -------- '
print ' '   
print ' ------------ END OF PROGRAM ---------- '  
print ' ' 
    
##################################################################    
#                        END OF PROGRAM                          #
##################################################################
