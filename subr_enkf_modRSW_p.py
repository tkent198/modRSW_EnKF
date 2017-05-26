#######################################################################
# Perturbed obs EnKF for modRSW with topography
#               (T. Kent: tkent198@gmail)
#######################################################################

'''
May 2017

SUBROUTINE (p) for batch-processing EnKF experiments. 
Given parameters and truth run supplied by <main>, the function <run_enkf> carries out ensemble integratiopns 
IN PARALLEL using the multiprocessing module.

'''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import multiprocessing as mp
from datetime import datetime

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from parameters import * # module storing fixed parameters
from f_modRSW import make_grid, step_forward_topog, time_step, ens_forecast, ens_forecast_topog
from f_enkf_modRSW import analysis_step_enkf
from create_readme import create_readme

def run_enkf(i,j,k,loc,add_inf,inf,ic,U_tr_array,dirname):

    print ' '
    print '---------------------------------------------------'
    print '----------------- EXPERIMENT '+str(i+1)+str(j+1)+str(k+1)+' ------------------'
    print '---------------------------------------------------'
    print ' '

    obs_dens = o_d
    pars_ob = [obs_dens, ob_noise]
    pars_enda = [inf, loc, add_inf]
    
    #################################################################
    # create directory for output
    #################################################################
    cwd = os.getcwd()
    #e.g. if i,j,k... etc are coming from outer loop:
    dirn = str(cwd+dirname+dirname+str(i+1)+str(j+1)+str(k+1))
    #check if dir exixts, if not make it
    try:
        os.makedirs(dirn)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    ##################################################################
    # Mesh generation for forecasts
    ##################################################################

    fc_grid =  make_grid(Nk_fc,L) # forecast

    Kk_fc = fc_grid[0]
    x_fc = fc_grid[1]
    xc_fc = fc_grid[2]

    ##################################################################    
    #  Apply initial conditions
    ##################################################################
    print ' ' 
    print '---------------------------------------------------' 
    print '---------      ICs: generate ensemble     ---------'
    print '---------------------------------------------------' 
    print ' ' 
    print 'Initial condition =', str(ic), '(see <init_cond_modRSW.py> for deatils ... )'
    print ' ' 
    ### Forecast ic 
    U0_fc, B = ic(x_fc,Nk_fc,Neq,H0,L,A,V) # control IC to be perturbed

    U0ens = np.empty([Neq,Nk_fc,n_ens])

    print 'Initial ensemble perurbations:'
    print 'sig_ic = [sig_h, sig_hu, sig_hr] =', sig_ic    

    # Generate initial ensemble
    for jj in range(0,Neq):
        for N in range(0,n_ens):
            # add sig_ic to EACH GRIDPOINT
            U0ens[jj,:,N] = U0_fc[jj,:] + sig_ic[jj]*np.random.randn(Nk_fc)
            # add sig_ic to TRAJECTORY as a whole
            #U0ens[jj,:,N] = U0_fc[jj,:] + sig_ic[jj]*np.random.randn(1)


    ##################################################################
    #%%%-----        Define arrays for outputting data       ------%%%
    ##################################################################
    nd = Neq*Nk_fc                          # total # d. of f.
    n_obs = nd/pars_ob[0]                      # total # obs
    X_array = np.empty([nd,n_ens,Nmeas+1])
    Xan_array = np.empty([nd,n_ens,Nmeas+1])
    X_tr_array = np.empty([nd,1,Nmeas+1])
    Y_obs_array = np.empty([n_obs,n_ens,Nmeas+1])
    OI = np.empty([Neq+1,Nmeas+1])

    # create readme file of exp summary and save
    PARS = [Nk_fc, Nk_tr, n_ens, assim_time, pars_ob, pars_enda, sig_ic, n_obs]
    create_readme(dirn, PARS, ic)
    
    ##################################################################
    #  Integrate ensembles forward in time until obs. is available   #
    ##################################################################
    print ' '
    print '-------------------------------------------------'
    print '------ CYCLED FORECAST-ASSIMILATION SYSTEM ------'
    print '-------------------------------------------------'
    print '--------- ENSEMBLE FORECASTS + EnKF--------------'
    print '-------------------------------------------------'
    print ' '
    
    # Initialise...
    U = U0ens
    index = 0 # to step through assim_time
    tmeasure = dtmeasure # reset tmeasure
    
    
    while tmeasure-dtmeasure < tmax:
        
        print ' '
        print '----------------------------------------------'
        print '------------ FORECAST STEP: START ------------'
        print '----------------------------------------------'
        print ' '
        
        num_cores_tot = mp.cpu_count()
        num_cores_use = num_cores_tot/2
        
        print 'Starting ensemble integrations from time =', assim_time[index],' to',assim_time[index+1]
        print 'Number of cores available:', num_cores_tot
        print 'Number of cores used:', num_cores_use
        print  ' *** Started: ', str(datetime.now())

        print np.shape(U)
        
        pool = mp.Pool(processes=num_cores_use)
        
        mp_out = [pool.apply_async(ens_forecast_topog, args=(N, U, B, Nk_fc, Kk_fc, assim_time, index, tmeasure)) for N in range(0,n_ens)]
        
        U = [p.get() for p in mp_out]
        
        pool.close()
        
        print ' All ensembles integrated forward from time =', assim_time[index],' to',assim_time[index+1]
        print ' *** Ended: ', str(datetime.now())
        print np.shape(U)

        U=np.swapaxes(U,0,1)
        U=np.swapaxes(U,1,2)
   
        print np.shape(U)
        
        print ' '
        print '----------------------------------------------'
        print '------------- FORECAST STEP: END -------------'
        print '----------------------------------------------'
        print ' '
        
        ##################################################################
        #  calculate analysis at observing time then integrate forward  #
        ##################################################################
        
        U_an, U_fc, X_array[:,:,index+1], X_tr_array[:,:,index+1], Xan_array[:,:,index+1], Y_obs_array[:,:,index+1], OI[:,index+1] = analysis_step_enkf(U, U_tr_array, tmeasure, dtmeasure, index, pars_ob, pars_enda)
        
        U = U_an # update U with analysis ensembles for next integration
        
#        np.save(str(dirn+'/U'),U)
        np.save(str(dirn+'/B'),B)
        np.save(str(dirn+'/X_array'),X_array)
        np.save(str(dirn+'/X_tr_array'),X_tr_array)
        np.save(str(dirn+'/Xan_array'),Xan_array)
        np.save(str(dirn+'/Y_obs_array'),Y_obs_array)
        np.save(str(dirn+'/OI'),OI)
        
        print ' *** Data saved in :', dirn
        print ' ' 
       
        # on to next assim_time
        index = index + 1
        tmeasure = tmeasure + dtmeasure
        

    ##################################################################


    PARS = [Nk_fc, Nk_tr, n_ens, assim_time, pars_ob, pars_enda, sig_ic, n_obs]

    # create readme file and save
    create_readme(dirn, PARS, ic)

    # print summary to terminal aswell
    print ' ' 
    print '---------------------------------------' 
    print '--------- END OF ASSIMILATION ---------' 
    print '---------------------------------------' 
    print ' '   
    print ' -------------- SUMMARY: ------------- '  
    print ' ' 
    print 'Dynamics:'
    print 'Ro =', Ro  
    print 'Fr = ', Fr
    print '(H_0 , H_c , H_r) =', [H0, Hc, Hr] 
    print '(alpha, beta, c2) = ', [alpha2, beta, cc2]
    print '(cfl_fc, cfl_tr) = ', [cfl_fc, cfl_tr]
    print 'Initial condition =', str(ic)
    print ' ' 
    print 'Assimilation:'
    print 'Forecast resolution (number of gridcells) =', Nk_fc
    print 'Truth resolution (number of gridcells) =', Nk_tr   
    if Nk_fc == Nk_tr: # perfect model
        print '>>> perfect model scenario'
    else:
        print '>>> imperfect model scenario' 
    print ' '  
    print 'Number of ensembles =', n_ens  
#    print 'Assimilation times  =', assim_time[1:]
    print 'Observation density: observe every', pars_ob[0], 'gridcells...'
    print 'i.e., total no. of obs. =', Nk_fc*Neq/pars_ob[0]
    print 'Observation noise =', pars_ob[1]  
    if pars_enda[0] != 1.0: # inflate the ensemble
        print 'Multiplicative (ensemble) inflation factor =', pars_enda[0]
    else:
        print 'No inflation applied'
    print 'Additive inflation factor =', pars_enda[2]
    print 'Localisation lengthscale =', pars_enda[1]
    print ' '   
    print ' ----------- END OF SUMMARY: ---------- '  
    print ' '  





    ##################################################################
    #                       END OF PROGRAM                           #
    ##################################################################


