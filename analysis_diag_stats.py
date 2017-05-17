##################################################################
#--------------- Error stats for saved data ---------------
#                   (T. Kent: mmtk@leeds.ac.uk)
##################################################################


## generic modules
import os
import errno
import numpy as np
import matplotlib.pyplot as plt

## custom modules

from parameters import *
from crps_calc_fun import crps_calc
##################################################################


def ave_stats(i, j, k, dirname):
    '''
    INPUT:
    ## e.g. if i,j,k... etc a coming from outer loop:
    i=1
    j=0
    k=0
    ##
    ##
    dirname = '/addinfv7_4dres'
    ##
    
    OUTPUT:
    spr, rmse, crps, OI for fc and an
    '''
    # LOAD DATA FROM GIVEN DIRECTORY
    cwd = os.getcwd()
    dirn = str(cwd+dirname+dirname+str(i+1)+str(j+1)+str(k+1))

    if os.path.exists(dirn):
        print ' '
        print 'Path: '
        print dirn
        print ' exists... calculating stats...'
        print ' '
        
        # parameters for outer loop
        o_d = [20,40]
        loc = [1.5, 2.5, 3.5, 0.]
        #inf = [1.1, 1.25, 1.5, 1.75]
        inf = [1.01, 1.05, 1.1]

        # LOAD DATA FROM GIVEN DIRECTORY
        X = np.load(str(dirn+'/X_array.npy')) # fc ensembles
        X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
        Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
        Y_obs = np.load(str(dirn+'/Y_obs_array.npy')) # obs ensembles
        OI = np.load(str(dirn+'/OI.npy')) # obs ensembles

        #np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        #print 'X_array shape (n_d,n_ens,T)      : ', np.shape(X)
        #print 'X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)
        #print 'Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)
        #print 'Y_obs_array shape (p,n_ens,T)    : ', np.shape(Y_obs)
        #print 'OI shape (Neq + 1,T)             : ', np.shape(OI)

        ##################################################################

        Neq = np.shape(OI)[0] - 1
        n_d = np.shape(X)[0]
        Nk_fc = n_d/Neq
        Kk_fc = 1./Nk_fc
        n_ens = np.shape(X)[1]
        n_obs = np.shape(Y_obs)[0]
        obs_dens = n_d/n_obs
        t_an = np.shape(X)[2]
        time_vec = range(0,t_an)

        # masks for locating model variables in state vector
        h_mask = range(0,Nk_fc)
        hu_mask = range(Nk_fc,2*Nk_fc)
        hr_mask = range(2*Nk_fc,3*Nk_fc)

        # masks for locating obs locations
        row_vec = range(obs_dens,n_d+1,obs_dens)
        obs_mask = np.array(row_vec[0:n_obs/Neq])-1
        h_obs_mask = range(0,n_obs/Neq)
        hu_obs_mask = range(n_obs/Neq,2*n_obs/Neq)
        hr_obs_mask = range(2*n_obs/Neq,3*n_obs/Neq)

        nz_index = np.where(OI[0,:])
        
        if (len(nz_index[0]) < t_an-15):
            print 'Runs crashed before Tmax...'
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        else:
            ##################################################################
            print 'Runs completed... '
            # for means and deviations
            Xbar = np.empty(np.shape(X))
            Xdev = np.empty(np.shape(X))
            Xanbar = np.empty(np.shape(X))
            Xandev = np.empty(np.shape(X))
            Xdev_tr = np.empty(np.shape(X))
            Xandev_tr = np.empty(np.shape(X))

            # for errs as at each assim time
            rmse_fc = np.empty((Neq,len(time_vec)))
            rmse_an = np.empty((Neq,len(time_vec)))
            spr_fc = np.empty((Neq,len(time_vec)))
            spr_an = np.empty((Neq,len(time_vec)))
            ame_fc = np.empty((Neq,len(time_vec)))
            ame_an = np.empty((Neq,len(time_vec)))
            crps_fc = np.empty((Neq,len(time_vec)))
            crps_an = np.empty((Neq,len(time_vec)))
            
            ONE = np.ones([n_ens,n_ens])
            ONE = ONE/n_ens # NxN array with elements equal to 1/N

            print ' *** Calculating errors from ', dirn

            for T in time_vec[1:]:
                
                plt.clf() # clear figs from previous loop
                
                Xbar[:,:,T] = np.dot(X[:,:,T],ONE) # fc mean
                Xdev[:,:,T] = X[:,:,T] - Xbar[:,:,T] # fc deviations from mean
                Xdev_tr[:,:,T] = X[:,:,T] - X_tr[:,:,T] # fc deviations from truth
                Xanbar[:,:,T] = np.dot(Xan[:,:,T],ONE) # an mean
                Xandev[:,:,T] = Xan[:,:,T] - Xanbar[:,:,T] # an deviations from mean
                Xandev_tr[:,:,T] = Xan[:,:,T] - X_tr[:,:,T] # an deviations from truth
                
                ##################################################################
                ###                       ERRORS                              ####
                ##################################################################
                
                # FORECAST: mean error
                fc_err = Xbar[:,0,T] - X_tr[:,0,T] # fc_err = ens. mean - truth
                fc_err2 = fc_err**2
                
                # ANALYSIS: mean error
                an_err = Xanbar[:,0,T] - X_tr[:,0,T] # an_err = analysis ens. mean - truth
                an_err2 = an_err**2
                
                # FORECAST: cov matrix for spread...
                Pf = np.dot(Xdev[:,:,T],np.transpose(Xdev[:,:,T]))
                Pf = Pf/(n_ens - 1) # fc covariance matrix
                var_fc = np.diag(Pf)
                
                # ... and rmse
                Pf_tr = np.dot(Xdev_tr[:,:,T],np.transpose(Xdev_tr[:,:,T]))
                Pf_tr = Pf_tr/(n_ens - 1) # fc covariance matrix w.r.t. truth
                var_fct = np.diag(Pf_tr)
                
                # ANALYSIS: cov matrix for spread...
                Pa = np.dot(Xandev[:,:,T],np.transpose(Xandev[:,:,T]))
                Pa = Pa/(n_ens - 1) # analysis covariance matrix
                var_an = np.diag(Pa)
                
                # ... and rmse
                Pa_tr = np.dot(Xandev_tr[:,:,T],np.transpose(Xandev_tr[:,:,T]))
                Pa_tr = Pa_tr/(n_ens - 1) # fc covariance matrix w.r.t truth
                var_ant = np.diag(Pa_tr)
                
                ##################################################################
                ###                       CRPS                                ####
                ##################################################################

                CRPS_fc = np.empty((Neq,Nk_fc))
                CRPS_an = np.empty((Neq,Nk_fc))

                for ii in h_mask:
                    CRPS_fc[0,ii] = crps_calc(X[ii,:,T],X_tr[ii,0,T])
                    CRPS_fc[1,ii] = crps_calc(X[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
                    CRPS_fc[2,ii] = crps_calc(X[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])
                    CRPS_an[0,ii] = crps_calc(Xan[ii,:,T],X_tr[ii,0,T])
                    CRPS_an[1,ii] = crps_calc(Xan[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
                    CRPS_an[2,ii] = crps_calc(Xan[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])


                #################################################################
              

            
                # domain-averaged errors
                ame_an[0,T] = np.mean(np.absolute(an_err[h_mask]))
                ame_fc[0,T] = np.mean(np.absolute(fc_err[h_mask]))
                spr_an[0,T] = np.sqrt(np.mean(var_an[h_mask]))
                spr_fc[0,T] = np.sqrt(np.mean(var_fc[h_mask]))
                rmse_an[0,T] = np.sqrt(np.mean(an_err2[h_mask]))
                rmse_fc[0,T] = np.sqrt(np.mean(fc_err2[h_mask]))
                crps_an[0,T] = np.mean(CRPS_an[0,:])
                crps_fc[0,T] = np.mean(CRPS_fc[0,:])

                ame_an[1,T] = np.mean(np.absolute(an_err[hu_mask]))
                ame_fc[1,T] = np.mean(np.absolute(fc_err[hu_mask]))
                spr_an[1,T] = np.sqrt(np.mean(var_an[hu_mask]))
                spr_fc[1,T] = np.sqrt(np.mean(var_fc[hu_mask]))
                rmse_an[1,T] = np.sqrt(np.mean(an_err2[hu_mask]))
                rmse_fc[1,T] = np.sqrt(np.mean(fc_err2[hu_mask]))
                crps_an[1,T] = np.mean(CRPS_an[1,:])
                crps_fc[1,T] = np.mean(CRPS_fc[1,:])

                ame_an[2,T] = np.mean(np.absolute(an_err[hr_mask]))
                ame_fc[2,T] = np.mean(np.absolute(fc_err[hr_mask]))
                spr_an[2,T] = np.sqrt(np.mean(var_an[hr_mask]))
                spr_fc[2,T] = np.sqrt(np.mean(var_fc[hr_mask]))
                rmse_an[2,T] = np.sqrt(np.mean(an_err2[hr_mask]))
                rmse_fc[2,T] = np.sqrt(np.mean(fc_err2[hr_mask]))
                crps_an[2,T] = np.mean(CRPS_an[2,:])
                crps_fc[2,T] = np.mean(CRPS_fc[2,:])
            ###########################################################################

            spr_fc_ave = spr_fc[:,nz_index].mean(axis=-1)
            err_fc_ave = ame_fc[:,nz_index].mean(axis=-1)
            rmse_fc_ave = rmse_fc[:,nz_index].mean(axis=-1)
            crps_fc_ave = crps_fc[:,nz_index].mean(axis=-1)
            
            spr_an_ave = spr_an[:,nz_index].mean(axis=-1)
            err_an_ave = ame_an[:,nz_index].mean(axis=-1)
            rmse_an_ave = rmse_an[:,nz_index].mean(axis=-1)
            crps_an_ave = crps_an[:,nz_index].mean(axis=-1)
            OI_ave = 100*OI[0,nz_index].mean(axis=-1)
            
            spr_fc_ave = spr_fc_ave.mean()
            err_fc_ave = err_fc_ave.mean()
            rmse_fc_ave = rmse_fc_ave.mean()
            crps_fc_ave = crps_fc_ave.mean()
            
            spr_an_ave = spr_an_ave.mean()
            err_an_ave = err_an_ave.mean()
            rmse_an_ave = rmse_an_ave.mean()
            crps_an_ave = crps_an_ave.mean()
            
            print 'spr_fc ave. =', spr_fc_ave
            print 'err_fc ave. =', err_fc_ave
            print 'rmse_fc ave. =', rmse_fc_ave
            print 'crps_fc ave. =', crps_fc_ave
            print 'spr_an ave. =', spr_an_ave
            print 'err_an ave. =', err_an_ave
            print 'rmse_an ave. =', rmse_an_ave
            print 'crps_an_ave. =', crps_an_ave
            print 'OI ave. =', OI_ave

            return spr_fc_ave, err_fc_ave, rmse_fc_ave, crps_fc_ave, spr_an_ave, err_an_ave, rmse_an_ave, crps_an_ave, OI_ave

    else:
        print ' '
        print ' Path:'
        print dirn
        print 'does not exist.. moving on to next one...'
        print ' '
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

