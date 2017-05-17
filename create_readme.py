#######################################################################
# Create readme.txt file for summarising experiments, output saved in <dirn> to accompany outputted data from main run script and EnKF subroutine.
# (T. Kent: mmtk@leeds.ac.uk)
#######################################################################

from parameters import *
from datetime import datetime
import os

def create_readme(dirn , PARS , ic):
    '''
    INPUT:
    > dirn = directory path
    > PARS = [Nk_fc, Nk_tr, n_ens, assim_time, pars_ob, pars_enda, sig_ic, n_obs]
    where pars = [obs_dens, inf, ob_noise]  
    > ic = initial condition
        
    OUTPUT:
    > fname.txt file saved as dirn
    '''   
        
    fname = str(dirn+'/readme.txt')

    f = open(fname,'w')
    print >>f, ' ------------- FILENAME ------------- ' 
    print >>f, fname   
    print >>f, ' '   
    print >>f, 'Created: ', str(datetime.now())   
    print >>f, ' '        
    print >>f, 'Output <.npy> saved to directory:' 
    print >>f, str(dirn)   
    print >>f, ' '   
    print >>f, ' -------------- SUMMARY: ------------- '  
    print >>f, ' ' 
    print >>f, 'Dynamics:'
    print >>f, ' ' 
    print >>f, 'Ro =', Ro  
    print >>f, 'Fr = ', Fr
    print >>f, '(H_0 , H_c , H_r) =', [H0, Hc, Hr] 
    print >>f, '(alpha, beta, c2) = ', [alpha2, beta, cc2]
    print >>f, '(cfl_fc, cfl_tr) = ', [cfl_fc, cfl_tr]
    print >>f, 'Initial condition =', str(ic)
    print >>f, 'IC noise for initial ens. generation: ', PARS[6]
    print >>f, ' ' 
    print >>f, 'Assimilation:'
    print >>f, ' ' 
    print >>f, 'Forecast resolution (number of gridcells) =', PARS[0]
    print >>f, 'Truth resolution (number of gridcells) =', PARS[1]   
    if PARS[0] == PARS[1]: # perfect model
        print >>f, '            >>> perfect model scenario'
    else: # imperfect model
        print >>f, '            >>> imperfect model scenario' 
    print ' '  
    print >>f, 'Number of ensembles =', PARS[2]  
#    print >>f, 'Assimilation times  =', PARS[3][1:]
    print >>f, 'Observation density: observe every', PARS[4][0], 'gridcells...'
    print >>f, 'i.e., total no. of obs. =', PARS[7]
    print >>f, 'Observation noise =', PARS[4][1]  
    if PARS[5][0] != 1.0: # inflate the ensemble
        print >>f, 'Multiplicative inflation factor =', PARS[5][0]
    else: # no inflation
        print >>f, 'No inflation applied'
    print >>f, 'Additive inflation factor =', PARS[5][2]
    print >>f, 'Localisation lengthscale =', PARS[5][1]
    print >>f, ' '   
    print >>f, ' ----------- END OF SUMMARY: ---------- '  
    print >>f, ' '  
    f.close()
