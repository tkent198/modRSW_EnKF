#######################################################################
# Collection of functions used in EnKF scripts
# (T. Kent: mmtk@leeds.ac.uk)
#######################################################################
'''
Functions:
        > generate_truth(): simulates truth trajectory at refined resolution and stores run at given observing times.
        > gasp_cohn(): gaspari cohn localisation matrix 
        >  analysis_step_enkf(): updates ensemble given observations from truth, returns analysis ensemble and forecast ensemble.
'''
import math as m
import numpy as np
from parameters import *
import os

##################################################################
#------------------ Compute truth trajectories ------------------
##################################################################


def generate_truth(U_tr_array, B_tr, Nk_tr, tr_grid, assim_time, f_path_name):
    
    from f_modRSW import time_step, step_forward_topog

    Kk_tr = tr_grid[0] 
    x_tr = tr_grid[1]
    
    tn = assim_time[0]
    tmax = assim_time[-1]
    dtmeasure = assim_time[1]-assim_time[0]
    tmeasure = dtmeasure
    
    U_tr = U_tr_array[:,:,0]
    
    index = 1 # for U_tr_array (start from 1 as 0 contains IC).
    while tn < tmax:
        dt = time_step(U_tr,Kk_tr,cfl_tr)
        tn = tn + dt

        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12
            tn = tmeasure + 1e-12
    
        U_tr = step_forward_topog(U_tr,B_tr,dt,tn,Nk_tr,Kk_tr)
        print 't_tr =',tn

        if tn > tmeasure:
            U_tr_array[:,:,index] = U_tr
            print '*** STORE TRUTH at observing time = ',tmeasure,' ***'
            tmeasure = tmeasure + dtmeasure
            index = index + 1
            
    np.save(f_path_name,U_tr_array)
    
    print '* DONE: truth array saved to:', f_path_name, ' with shape:', np.shape(U_tr_array), ' *'
        
    return U_tr_array    
    
    
##################################################################
# GASPARI-COHN TAPER FUNCTION FOR COV LOCALISATION
# from Jeff Whitaker's github: https://github.com/jswhit/
################################################################    
def gaspcohn(r):
    # Gaspari-Cohn taper function
    # very close to exp(-(r/c)**2), where c = sqrt(0.15)
    # r should be >0 and normalized so taper = 0 at r = 1
    rr = 2.0*r
    rr += 1.e-13 # avoid divide by zero warnings from numpy
    taper = np.where(r<=0.5, \
                     ( ( ( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2 + 1.0,\
                     np.zeros(r.shape,r.dtype))

    taper = np.where(np.logical_and(r>0.5,r<1.), \
                    ( ( ( ( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
                    + 4.0 - 2.0 / (3.0 * rr), taper)
    return taper    


##################################################################
#'''------------------ ANALYSIS STEP ------------------'''
##################################################################

def analysis_step_enkf(U_fc, U_tr, tmeasure, dtmeasure, index, pars_ob, pars_enda):
    '''
    INPUTS
    U_fc: ensemble trajectories , shape (Neq,Nk_fc,n_ens)
    U_tr: truth trajectory, shape (Neq,Nk_tr,Nmeas+1)
    tmeasure: time of assimilation
    dtmeasure: length of window
    pars_ob: vector of parameter values relating to obs (density and error)
    pars_enda: vector of parameters relating to DA fixes (inflation and localisation)
    '''
    
    print ' '
    print '----------------------------------------------'
    print '------------ ANALYSIS STEP: START ------------'
    print '----------------------------------------------'
    print ' '

    Nk_fc = np.shape(U_fc)[1] # fc resolution (no. of cells)
    Kk_fc = L/Nk_fc # fc resolution (cell length)
    Nk_tr = np.shape(U_tr)[1] # tr resolution (no. of cells)
    n_ens = np.shape(U_fc)[2]
    n_d = Neq*Nk_fc # total number of variables (dgs of freedom)
    dres = Nk_tr/Nk_fc # ratio of resolutions
    inf = pars_enda[0] # inflation factor
    loc = pars_enda[1]
    add_inf = pars_enda[2]
    obs_dens = pars_ob[0] 
    ob_noise = pars_ob[1] # 4-vector of obs noise
    
    
    print ' ' 
    print '--------- ANALYSIS: perturbed obs EnKF ---------' 
    print ' '     
    print 'Assimilation time = ', tmeasure
    print 'Number of ensembles = ', n_ens

    # project truth onto forecast grid so that U and U_tr are the same dimension
    U_tmp = np.empty([Neq,Nk_fc])
    for i in range(0,Nk_fc):
        U_tmp[:,i] = U_tr[:,i*dres:(i+1)*dres,index+1].mean(axis=-1)
    U_tr = U_tmp

    # for assimilation, work with [h,u,r]
    U_fc[1:,:,:] = U_fc[1:,:,:]/U_fc[0,:,:]
    U_tr[1:,:] = U_tr[1:,:]/U_tr[0,:]

    X_tr = U_tr.flatten()
    X_tr = np.matrix(X_tr).T 

    # state matrix (flatten the array)
    X = np.empty((n_d,n_ens))
    for N in range(0,n_ens):
        X[:,N] = U_fc[:,:,N].flatten()


    #### CALCULATE KALMAN GAIN, INNOVATIONS, AND ANALYSIS STATES ####
    
    ONE = np.ones([n_ens,n_ens])
    ONE = ONE/n_ens # NxN array with elements equal to 1/N
    Xbar = np.dot(X,ONE) # mean
    Xdev = X - Xbar # deviations

    # covariance matrix
    Pf = np.dot(Xdev, Xdev.T)
    Pf = Pf/(n_ens - 1)
    Cf = np.corrcoef(Xdev) # correlation matrix
    
    # make pseudo-obs by perturbing the truth.
    n_obs = n_d/obs_dens # no. of observations
    print 'Total no. of obs. =', n_obs
    # observation operator
    H = np.zeros((n_obs,n_d))
    row_vec = range(obs_dens,n_d+1,obs_dens)
    for i in range(0,n_obs):
        H[i,row_vec[i]-1]=1

    # for ensemble of observations
    ob_noise = np.repeat(ob_noise,n_obs/Neq) 
    obs_pert = ob_noise[:,None]*np.random.randn(n_obs,n_ens)
    print 'obs_pert shape =', np.shape(obs_pert)
    Y_obs = np.empty([n_obs,n_ens])
    Y_mod = np.dot(H, X_tr)
    print 'Y_mod shape =', np.shape(Y_mod)    
    Y_obs = Y_mod + obs_pert #y_o = y_m + e_o
    

    # construct localisation matrix rho based on Gaspari Cohn function
    loc_rho = pars_enda[1] # loc_rho is form of lengthscale.
    rr = np.arange(0,loc_rho,loc_rho/Nk_fc) 
    vec = gaspcohn(rr)
    
    rho = np.zeros((Nk_fc,Nk_fc))
    for i in range(Nk_fc):
        for j in range(Nk_fc):
            rho[i,j] = vec[np.min([abs(i-j),abs(i+Nk_fc-j),abs(i-Nk_fc-j)])]
    rho = np.tile(rho, (Neq,Neq))
    print 'loc matrix rho shape: ', np.shape(rho)

    # construct K
    R = ob_noise*ob_noise*np.identity(n_obs) # obs cov matrix
    K = np.dot(H, np.dot(rho*Pf,H.T)) + R # H B H^T + R
    K = np.linalg.inv(K) # [H B H^T + R]^-1
    K = np.dot(np.dot(rho*Pf,H.T), K) # (rho Pf)H^T [H (rho Pf) H^T + R]^-1

    # compute innovation d = Y-H*X
    D = Y_obs - np.dot(H,X)

    # compute analysis
    Xan = X + np.dot(K,D) # kalman update step
    Xanbar = np.dot(Xan,ONE) # analysis mean
    Xandev = Xan - Xanbar  # analysis deviations

    Pa = np.dot(Xandev,Xandev.T)
    Pa = Pa/(n_ens - 1) # analysis covariance matrix
    Ca = np.corrcoef(Xandev) # analysis correlation matrix


    ### ADDITIVE INFLATION ...### (NEEDS MOVING after analysis...)
    cwd = os.getcwd()
    Q = np.load(str(cwd+'/Q_offline.npy'))
    print 'max Q value: ', np.max(Q)
    print 'min Q value: ', np.min(Q)

    q = add_inf*np.random.multivariate_normal(np.zeros(n_d), Q, n_ens)
    q = q.T
    Xan = Xan + q # x(t+1) = M(x(t)) + q



    # masks for locating model variables in state vector
    h_mask = range(0,Nk_fc)
    hu_mask = range(Nk_fc,2*Nk_fc)
    hr_mask = range(2*Nk_fc,3*Nk_fc)

    obs_mask = np.array(row_vec[0:n_obs/Neq])-1
    h_obs_mask = range(0,n_obs/Neq)
    hu_obs_mask = range(n_obs/Neq,2*n_obs/Neq)
    hr_obs_mask = range(2*n_obs/Neq,3*n_obs/Neq)

    # make matrices arrays for manipulating and then saving
    X = np.array(X)
    X_tr = np.array(X_tr)
    Xbar = np.array(Xbar)
    Xan = np.array(Xan)
    Xanbar = np.array(Xanbar)
    Xandev = np.array(Xandev)
    Y_obs = np.array(Y_obs)

    # transform to x = (h,hu,hr) for inflation (to avoid double inflation of hu,hr)
    Xan_new = Xan
    Xan_new[hu_mask,:] = Xan_new[hu_mask,:]*Xan_new[h_mask,:]
    Xan_new[hr_mask,:] = Xan_new[hr_mask,:]*Xan_new[h_mask,:]

    ### MULTIPLICATIVE INFLATION
    if inf != 1.0: # inflate the ensemble
        print 'Covariance (ensemble) inflation factor =', inf
        Xanbar_new = np.dot(Xan_new, ONE)
        Xandev_new = Xan_new - Xanbar_new
        Xandev_new = inf*Xandev_new
        Xan_new = Xandev_new + Xanbar_new # inflated analysis ensemble
        Xan_new = np.array(Xan_new)
    else:
        print 'No multiplicative inflation applied'

    # if hr < 0, set to zero:
    hr = Xan_new[hr_mask,:]
    hr[hr < 0.] = 0.
    Xan_new[hr_mask,:] = hr
    
    # if h < 0, set to epsilon:
    h = Xan_new[h_mask,:]
    h[h < 0.] = 1e-3
    Xan_new[h_mask,:] = h

    # transform from X to U for next integration:
    U_an = np.empty((Neq,Nk_fc,n_ens))
    for N in range(0,n_ens):
        U_an[:,:,N] = Xan_new[:,N].reshape(Neq,Nk_fc)

    # now inflated, transform back to x = (h,u,r) for saving and later plotting
    Xan[h_mask,:] = Xan_new[h_mask,:]
    Xan[hu_mask,:] = Xan_new[hu_mask,:]/Xan_new[h_mask,:]
    Xan[hr_mask,:] = Xan_new[hr_mask,:]/Xan_new[h_mask,:]



    print ' '
    print '--------- CHECK SHAPE OF MATRICES: ---------'
    print 'U_fc shape   :', np.shape(U_fc)
    print 'U_tr shape   :', np.shape(U_tr)
    print 'X_truth shape:', np.shape(X_tr), '( NOTE: should be', n_d,' by 1 )'
    print 'X shape      :', np.shape(X), '( NOTE: should be', n_d,'by', n_ens,')'
    print 'Xbar shape   :', np.shape(Xbar)
    print 'Xdev shape   :', np.shape(Xdev)
    print 'Pf shape     :', np.shape(Pf), '( NOTE: should be n by n square for n=', n_d,')'
    print 'H shape      :', np.shape(H), '( NOTE: should be', n_obs,'by', n_d,')'
    print 'K shape      :', np.shape(K), '( NOTE: should be', n_d,'by', n_obs,')'
    print 'ob_pert shape:', np.shape(obs_pert)
    print 'Y_mod shape  :', np.shape(Y_mod)
    print 'Y_obs shape  :', np.shape(Y_obs)
    print 'Xan shape    :', np.shape(Xan), '( NOTE: should be the same as X shape)'
    print 'U_an shape   :', np.shape(U_an), '( NOTE: should be the same as U_fc shape)'


    ## observational influence diagnostics
    print ' '
    print '--------- OBSERVATIONAL INFLUENCE DIAGNOSTICS:---------'
    HK = np.dot(H,K)
    HKd = np.diag(HK)
    OI = np.trace(HK)/n_obs
    OI_h = Neq*np.sum(HKd[h_obs_mask])/n_obs
    OI_hu = Neq*np.sum(HKd[hu_obs_mask])/n_obs
    OI_hr = Neq*np.sum(HKd[hr_obs_mask])/n_obs
    OI_vec = np.array([OI , OI_h , OI_hu , OI_hr])

    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    print 'trace(HK) =', np.trace(HK)
    print 'shape(HK) =', np.shape(HK), '( NOTE: should be', n_obs,'by', n_obs,')'
    print 'OI =', OI
    print 'OI check = ', np.sum(HKd)/n_obs
    print 'OI_h =', OI_h
    print 'OI_hu =', OI_hu
    print 'OI_hr =', OI_hr
    print ' '
    print ' '
    print '----------------------------------------------'
    print '------------- ANALYSIS STEP: END -------------'
    print '----------------------------------------------'
    print ' '


    return U_an, U_fc, X, X_tr, Xan, Y_obs, OI_vec

##################################################################
#'''------------------ ANALYSIS STEP edit ------------------'''
##################################################################


