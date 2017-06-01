##################################################################
#--------------- Plotting routines for saved data ---------------
#                   (T. Kent: tkent198@gmail.com)
##################################################################
'''
    Plotting routine: <plot_func_t>
    
    Loads saved data in specific directories and produces plots as a function of time for OID, spr v err, and CRPS (i.e., domain-averaged time series). To use, specify (1) dir_name, (2) combination of parameters ijk.
    
    NOTE: Any changes to the outer loop parameters should be replicated here too.
    
    NOTEE: currently saves as .png files
    '''


## generic modules 
import os
import errno
import numpy as np
import matplotlib.pyplot as plt

## custom modules
from parameters import *
from crps_calc_fun import crps_calc

##################################################################

## 1. CHOOSE ijk. E.g., for test_enkf111/ [i,j,k] = [0,0,0]
i=0
j=0
k=0
## 2. CHOOSE directory name
dirname = '/test_enkf'
##

## parameters for outer loop
loc = [1e-10]
add_inf = [0.2]
inf = [1.01, 1.05, 1.1]

# make fig directory (if it doesn't already exist)
cwd = os.getcwd()
dirn = str(cwd+dirname+dirname+str(i+1)+str(j+1)+str(k+1))
figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

## load data
print '*** Loading saved data... '
B = np.load(str(dirn+'/B.npy')) #topography
X = np.load(str(dirn+'/X_array.npy')) # fc ensembles
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
Y_obs = np.load(str(dirn+'/Y_obs_array.npy')) # obs ensembles
OI = np.load(str(dirn+'/OI.npy')) # OI

# print shape of data arrays to terminal (sanity check)
print ' Check array shapes...'
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print 'X_array shape (n_d,n_ens,T)      : ', np.shape(X)
print 'X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)
print 'Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)
print 'Y_obs_array shape (p,n_ens,T)    : ', np.shape(Y_obs)
print ' '
##################################################################

# determine parameters from loaded arrays
Neq = np.shape(OI)[0] - 1
n_d = np.shape(X)[0]
Nk_fc = n_d/Neq
Kk_fc = 1./Nk_fc
xc = np.linspace(Kk_fc/2,L-Kk_fc/2,Nk_fc)
n_ens = np.shape(X)[1]
n_obs = np.shape(Y_obs)[0]
obs_dens = n_d/n_obs
t_an = np.shape(X)[2]
time_vec = range(0,t_an)
print 'time_vec = ', time_vec
print ' '

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

###################################################################
OI_check = 100*OI[1,1:]/3 + 100*OI[2,1:]/3 + 100*OI[3,1:]/3
OI_ave = 100*OI[0,1:-1].mean(axis=-1)

print 'OI ave. =', OI_ave

print ' ' 
print ' PLOT : OI'
print ' ' 
fig, axes = plt.subplots(1, 1, figsize=(8,5))
plt.suptitle("OI diagnostic  (N = %s): [loc, add_inf, inf] = [%s, %s, %s]" % (n_ens,loc[i], add_inf[j], inf[k]),fontsize=16)

axes.plot(time_vec[1:], 100*OI[1,1:],'r',label='$OID_h$') # rmse
axes.plot(time_vec[1:], 100*OI[2,1:],'b',label='$OID_{u}$')
axes.plot(time_vec[1:], 100*OI[3,1:],'c',label='$OID_{r}$')
axes.plot(time_vec[1:], 100*OI[0,1:],'k',linewidth=2.0,label='$OID$')
axes.set_ylabel('OID (%)',fontsize=18)
axes.legend(loc = 1, fontsize='large')
axes.set_ylim([0,50])
axes.set_xlim([time_vec[1],time_vec[-1]])
axes.set_xlabel('Assim. time $T$',fontsize=14)

name = "/OID.png"
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '

##################################################################

#nz_index = np.where(OI[0,:])

# for means and deviations
ONE = np.ones([n_ens,n_ens])
ONE = ONE/n_ens # NxN array with elements equal to 1/N
Xbar = np.empty(np.shape(X))
Xdev = np.empty(np.shape(X))
Xanbar = np.empty(np.shape(X))
Xandev = np.empty(np.shape(X))
Xdev_tr = np.empty(np.shape(X))
Xandev_tr = np.empty(np.shape(X))

# for errs etc at each assim time
rmse_fc = np.empty((Neq,len(time_vec)))
rmse_an = np.empty((Neq,len(time_vec)))
spr_fc = np.empty((Neq,len(time_vec)))
spr_an = np.empty((Neq,len(time_vec)))
ame_fc = np.empty((Neq,len(time_vec)))
ame_an = np.empty((Neq,len(time_vec)))
tote_fc = np.empty((Neq,len(time_vec)))
tote_an = np.empty((Neq,len(time_vec)))
crps_fc = np.empty((Neq,len(time_vec)))
crps_an = np.empty((Neq,len(time_vec)))


##################################################################
###               ERRORS + SPREAD                             ####
##################################################################

for T in time_vec[1:]:
    
    plt.clf() # clear figs from previous loop
    
    Xbar[:,:,T] = np.dot(X[:,:,T],ONE) # fc mean
    Xdev[:,:,T] = X[:,:,T] - Xbar[:,:,T] # fc deviations from mean
    Xdev_tr[:,:,T] = X[:,:,T] - X_tr[:,:,T] # fc deviations from truth
    Xanbar[:,:,T] = np.dot(Xan[:,:,T],ONE) # an mean
    Xandev[:,:,T] = Xan[:,:,T] - Xanbar[:,:,T] # an deviations from mean
    Xandev_tr[:,:,T] = Xan[:,:,T] - X_tr[:,:,T] # an deviations from truth    
    

    # ANALYSIS: ensemble mean error
    an_err = Xanbar[:,0,T] - X_tr[:,0,T] # an_err = analysis ens. mean - truth
    an_err2 = an_err**2

    # FORECAST: ensemble mean error
    fc_err = Xbar[:,0,T] - X_tr[:,0,T] # fc_err = ens. mean - truth
    fc_err2 = fc_err**2
    
    # analysis cov matrix
    Pa = np.dot(Xandev[:,:,T],np.transpose(Xandev[:,:,T]))
    Pa = Pa/(n_ens - 1) # analysis covariance matrix
    var_an = np.diag(Pa)

    Pa_tr = np.dot(Xandev_tr[:,:,T],np.transpose(Xandev_tr[:,:,T]))
    Pa_tr = Pa_tr/(n_ens - 1) # fc covariance matrix w.r.t truth
    var_ant = np.diag(Pa_tr)
    
    # forecast cov matrix
    Pf = np.dot(Xdev[:,:,T],np.transpose(Xdev[:,:,T]))
    Pf = Pf/(n_ens - 1) # fc covariance matrix
    var_fc = np.diag(Pf)

    Pf_tr = np.dot(Xdev_tr[:,:,T],np.transpose(Xdev_tr[:,:,T]))
    Pf_tr = Pf_tr/(n_ens - 1) # fc covariance matrix w.r.t. truth
    var_fct = np.diag(Pf_tr)
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


    # domain-averaged stats
    ame_an[0,T] = np.mean(np.absolute(an_err[h_mask]))
    ame_fc[0,T] = np.mean(np.absolute(fc_err[h_mask]))
    spr_an[0,T] = np.sqrt(np.mean(var_an[h_mask]))
    spr_fc[0,T] = np.sqrt(np.mean(var_fc[h_mask]))
    rmse_an[0,T] = np.sqrt(np.mean(an_err2[h_mask]))
    rmse_fc[0,T] = np.sqrt(np.mean(fc_err2[h_mask]))
    tote_an[0,T] = np.sqrt(np.mean(var_ant[h_mask]))
    tote_fc[0,T] = np.sqrt(np.mean(var_fct[h_mask]))
    crps_an[0,T] = np.mean(CRPS_an[0,:])
    crps_fc[0,T] = np.mean(CRPS_fc[0,:])

    ame_an[1,T] = np.mean(np.absolute(an_err[hu_mask]))
    ame_fc[1,T] = np.mean(np.absolute(fc_err[hu_mask]))
    spr_an[1,T] = np.sqrt(np.mean(var_an[hu_mask]))
    spr_fc[1,T] = np.sqrt(np.mean(var_fc[hu_mask]))
    rmse_an[1,T] = np.sqrt(np.mean(an_err2[hu_mask]))
    rmse_fc[1,T] = np.sqrt(np.mean(fc_err2[hu_mask]))
    tote_an[1,T] = np.sqrt(np.mean(var_ant[hu_mask]))
    tote_fc[1,T] = np.sqrt(np.mean(var_fct[hu_mask]))
    crps_an[1,T] = np.mean(CRPS_an[1,:])
    crps_fc[1,T] = np.mean(CRPS_fc[1,:])


    ame_an[2,T] = np.mean(np.absolute(an_err[hr_mask]))
    ame_fc[2,T] = np.mean(np.absolute(fc_err[hr_mask]))
    spr_an[2,T] = np.sqrt(np.mean(var_an[hr_mask]))
    spr_fc[2,T] = np.sqrt(np.mean(var_fc[hr_mask]))
    rmse_an[2,T] = np.sqrt(np.mean(an_err2[hr_mask]))
    rmse_fc[2,T] = np.sqrt(np.mean(fc_err2[hr_mask]))
    tote_an[2,T] = np.sqrt(np.mean(var_ant[hr_mask]))
    tote_fc[2,T] = np.sqrt(np.mean(var_fct[hr_mask]))
    crps_an[2,T] = np.mean(CRPS_an[2,:])
    crps_fc[2,T] = np.mean(CRPS_fc[2,:])
#####################################################################

###########################################################################

print ' '
print ' PLOT : RMS ERRORS vs SPREAD'
print ' '
ft=16

axlim0 = np.max(np.maximum(spr_fc[0,1:-1], rmse_fc[0,1:-1]))
axlim1 = np.max(np.maximum(spr_fc[1,1:-1], rmse_fc[1,1:-1]))
axlim2 = np.max(np.maximum(spr_fc[2,1:-1], rmse_fc[2,1:-1]))

fig, axes = plt.subplots(3, 1, figsize=(7,12))
plt.suptitle("Domain-averaged error vs spread  (N = %s): \n [loc, add_inf, inf] = [%s, %s, %s]" % (n_ens,loc[i], add_inf[j], inf[k]),fontsize=16)

axes[0].plot(time_vec[1:], spr_fc[0,1:],'r',label='fc spread') # spread
axes[0].plot(time_vec[1:], spr_an[0,1:],'b',label='an spread')
axes[0].plot(time_vec[1:], rmse_fc[0,1:], 'r--',label='fc rmse')
axes[0].plot(time_vec[1:], rmse_an[0,1:], 'b--',label='an rmse')
axes[0].set_ylabel('$h$',fontsize=18)
axes[0].text(1, 1.2*axlim0, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[0,1:].mean(axis=-1),rmse_an[0,1:].mean(axis=-1)), fontsize=ft, color='b')
axes[0].text(1, 1.1*axlim0, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[0,1:].mean(axis=-1),rmse_fc[0,1:].mean(axis=-1)), fontsize=ft, color='r')
axes[0].set_ylim([0,1.3*axlim0])
axes[0].legend(loc = 1, fontsize='small')

axes[1].plot(time_vec[1:], spr_fc[1,1:],'r',label='fc spread') # spread
axes[1].plot(time_vec[1:], spr_an[1,1:],'b',label='an spread')
axes[1].plot(time_vec[1:], rmse_fc[1,1:], 'r--',label='fc rmse')
axes[1].plot(time_vec[1:], rmse_an[1,1:], 'b--',label='an rmse')
axes[1].set_ylabel('$u$',fontsize=18)
axes[1].text(1, 1.2*axlim1, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[1,1:].mean(axis=-1),rmse_an[1,1:].mean(axis=-1)), fontsize=ft, color='b')
axes[1].text(1, 1.1*axlim1, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[1,1:].mean(axis=-1),rmse_fc[1,1:].mean(axis=-1)), fontsize=ft, color='r')
axes[1].set_ylim([0,1.3*axlim1])

axes[2].plot(time_vec[1:], spr_fc[2,1:],'r',label='fc spread') # spread
axes[2].plot(time_vec[1:], spr_an[2,1:],'b',label='an spread')
axes[2].plot(time_vec[1:], rmse_fc[2,1:], 'r--',label='fc rmse')
axes[2].plot(time_vec[1:], rmse_an[2,1:], 'b--',label='an rmse')
axes[2].set_ylabel('$r$',fontsize=18)
axes[2].set_xlabel('Assim. time $T$',fontsize=14)
axes[2].text(1, 1.2*axlim2, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[2,1:].mean(axis=-1),rmse_an[2,1:].mean(axis=-1)), fontsize=ft, color='b')
axes[2].text(1, 1.1*axlim2, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[2,1:].mean(axis=-1),rmse_fc[2,1:].mean(axis=-1)), fontsize=ft, color='r')
axes[2].set_ylim([0,1.3*axlim2])

name = "/spr_err.png"
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '

###########################################################################
###########################################################################

print ' '
print ' PLOT : CRPS'
print ' '

axlim0 = np.max(crps_fc[0,1:-1])
axlim1 = np.max(crps_fc[1,1:-1])
axlim2 = np.max(crps_fc[2,1:-1])

fig, axes = plt.subplots(3, 1, figsize=(7,12))
plt.suptitle("Domain-averaged CRPS  (N = %s): \n [loc, add_inf, inf] = [%s, %s, %s]" % (n_ens,loc[i], add_inf[j], inf[k]),fontsize=16)

axes[0].plot(time_vec[1:], crps_fc[0,1:],'r',label='fc') # spread
axes[0].plot(time_vec[1:], crps_an[0,1:],'b',label='an')
axes[0].set_ylabel('$h$',fontsize=18)
axes[0].text(1, 1.2*axlim0, '$CRPS_{an} = %.3g$' %crps_an[0,1:].mean(axis=-1), fontsize=ft, color='b')
axes[0].text(1, 1.1*axlim0, '$CRPS_{fc} = %.3g$' %crps_fc[0,1:].mean(axis=-1), fontsize=ft, color='r')
axes[0].set_ylim([0,1.3*axlim0])

axes[1].plot(time_vec[1:], crps_fc[1,1:],'r',label='fc') # spread
axes[1].plot(time_vec[1:], crps_an[1,1:],'b',label='an')
axes[1].set_ylabel('$u$',fontsize=18)
axes[1].text(1, 1.2*axlim1, '$CRPS_{an} = %.3g$' %crps_an[1,1:].mean(axis=-1), fontsize=ft, color='b')
axes[1].text(1, 1.1*axlim1, '$CRPS_{fc} = %.3g$' %crps_fc[1,1:].mean(axis=-1), fontsize=ft, color='r')
axes[1].set_ylim([0,1.3*axlim1])

axes[2].plot(time_vec[1:], crps_fc[2,1:],'r',label='fc') # spread
axes[2].plot(time_vec[1:], crps_an[2,1:],'b',label='an')
axes[2].set_ylabel('$r$',fontsize=18)
axes[2].set_xlabel('Assim. time $T$',fontsize=14)
axes[2].text(1, 1.2*axlim2, '$CRPS_{an} = %.3g$' %crps_an[2,1:].mean(axis=-1), fontsize=ft, color='b')
axes[2].text(1, 1.1*axlim2, '$CRPS_{fc} = %.3g$' %crps_fc[2,1:].mean(axis=-1), fontsize=ft, color='r')
axes[2].set_ylim([0,1.3*axlim2])

name = "/crps.png"
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '

###########################################################################
