##################################################################
#--------- EFS stats: errors and crps of ens foercasts -----------
#                   (T. Kent: amttk@leeds.ac.uk)
##################################################################
'''
    <EFS_stats>
    Computes and plots error growth, crps for the EFS data produced in <run_modRSW_EFS>.
    Also computes and saves error doubling times, to be used in <err_doub_hist>
        '''


## generic modules
import os
import errno
import numpy as np
import matplotlib.pyplot as plt

## custom modules

#from pars_modRSW import *
#from pars_enkf import *

from parameters import *
from crps_calc_fun import crps_calc

##################################################################

## e.g. if i,j,k... etc a coming from outer loop:
i=1
j=3
k=0
##
##
dirname = '/addinfv12_4dres'
##

# LOAD DATA FROM GIVEN DIRECTORY
cwd = os.getcwd()
dirn = str(cwd+dirname+dirname+str(i+1)+str(j+1)+str(k+1))
figsdir = str(dirn+'/figs')

#check if dir exixts, if not make it
try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise


T0 = 9
Tfc = 24
EFS_data = '/X_EFS_array_T'+str(T0)+'.npy'
# LOAD DATA FROM GIVEN DIRECTORY
X = np.load(str(dirn+EFS_data)) # fc ensembles
X_tr_array = np.load(str(dirn+'/X_tr_array.npy')) # truth

X_tr = X_tr_array[:,:,T0:T0+Tfc+1] # slice 24hr array from nature run
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print 'X_array shape (n_d,n_ens,T)      : ', np.shape(X)
print 'X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)


##################################################################

n_d = np.shape(X)[0]
Nk_fc = n_d/Neq
Kk_fc = 1./Nk_fc
n_ens = np.shape(X)[1]
t_an = np.shape(X)[2]
time_vec = range(0,t_an)

# masks for locating model variables in state vector
h_mask = range(0,Nk_fc)
hu_mask = range(Nk_fc,2*Nk_fc)
hr_mask = range(2*Nk_fc,3*Nk_fc)


##################################################################

# for means and deviations
Xbar = np.empty(np.shape(X))
Xdev = np.empty(np.shape(X))
Xdev_tr = np.empty(np.shape(X))

# for errs as at each assim time
rmse_fc = np.empty((Neq,len(time_vec)))
spr_fc = np.empty((Neq,len(time_vec)))
ame_fc = np.empty((Neq,len(time_vec)))
tote_fc = np.empty((Neq,len(time_vec)))
crps_fc = np.empty((Neq,len(time_vec)))
error_fc = np.empty((Neq,n_ens,len(time_vec)))

ONE = np.ones([n_ens,n_ens])
ONE = ONE/n_ens # NxN array with elements equal to 1/N

print ' *** Calculating errors...'

for T in time_vec:
    
    plt.clf() # clear figs from previous loop
    
    Xbar[:,:,T] = np.dot(X[:,:,T],ONE) # fc mean
    Xdev[:,:,T] = X[:,:,T] - Xbar[:,:,T] # fc deviations from mean
    Xdev_tr[:,:,T] = X[:,:,T] - X_tr[:,:,T] # fc deviations from truth
    
    ##################################################################
    ###               ERRORS + SPREAD                             ####
    ##################################################################
    
    
    # FORECAST: ensemble mean error
    fc_err = Xbar[:,0,T] - X_tr[:,0,T] # fc_err = ens. mean - truth
    fc_err2 = fc_err**2
    
    # forecast cov matrix
    Pf = np.dot(Xdev[:,:,T],np.transpose(Xdev[:,:,T]))
    Pf = Pf/(n_ens - 1) # fc covariance matrix
    var_fc = np.diag(Pf)
    
    Pf_tr = np.dot(Xdev_tr[:,:,T],np.transpose(Xdev_tr[:,:,T]))
    Pf_tr = Pf_tr/(n_ens - 1) # fc covariance matrix w.r.t. truth
    var_fct = np.diag(Pf_tr)
    
    err2 = Xdev_tr[:,:,T]**2
    err2_h = err2[h_mask,:].mean(axis=0)
    err2_u = err2[hu_mask,:].mean(axis=0)
    err2_r = err2[hr_mask,:].mean(axis=0)
    
    error_fc[0,:,T] = np.sqrt(err2_h)
    error_fc[1,:,T] = np.sqrt(err2_u)
    error_fc[2,:,T] = np.sqrt(err2_r)
    
    
    ##################################################################
    ###                       CRPS                                ####
    ##################################################################
    
    CRPS_fc = np.empty((Neq,Nk_fc))
    
    for ii in h_mask:
        CRPS_fc[0,ii] = crps_calc(X[ii,:,T],X_tr[ii,0,T])
        CRPS_fc[1,ii] = crps_calc(X[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
        CRPS_fc[2,ii] = crps_calc(X[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])
    
    
    #################################################################
    
    
    # domain-averaged stats
    ame_fc[0,T] = np.mean(np.absolute(fc_err[h_mask]))
    spr_fc[0,T] = np.sqrt(np.mean(var_fc[h_mask]))
    rmse_fc[0,T] = np.sqrt(np.mean(fc_err2[h_mask]))
    tote_fc[0,T] = np.sqrt(np.mean(var_fct[h_mask]))
    crps_fc[0,T] = np.mean(CRPS_fc[0,:])
    
    ame_fc[1,T] = np.mean(np.absolute(fc_err[hu_mask]))
    spr_fc[1,T] = np.sqrt(np.mean(var_fc[hu_mask]))
    rmse_fc[1,T] = np.sqrt(np.mean(fc_err2[hu_mask]))
    tote_fc[1,T] = np.sqrt(np.mean(var_fct[hu_mask]))
    crps_fc[1,T] = np.mean(CRPS_fc[1,:])

    ame_fc[2,T] = np.mean(np.absolute(fc_err[hr_mask]))
    spr_fc[2,T] = np.sqrt(np.mean(var_fc[hr_mask]))
    rmse_fc[2,T] = np.sqrt(np.mean(fc_err2[hr_mask]))
    tote_fc[2,T] = np.sqrt(np.mean(var_fct[hr_mask]))
    crps_fc[2,T] = np.mean(CRPS_fc[2,:])
#####################################################################

###########################################################################
'''
print ' '
print ' PLOT : RMS ERRORS (ensemble average)'
print ' '

axlim0 = np.max(np.max(tote_fc[0,:]))
axlim1 = np.max(np.max(tote_fc[1,:]))
axlim2 = np.max(np.max(tote_fc[2,:]))

fig, axes = plt.subplots(3, 2, figsize=(12,12))
#plt.suptitle("Domain-averaged error measures  (N = %s): \n [od, loc, inf] = [%s, %s, %s]" % (n_ens, o_d[i], loc[j], inf[k]),fontsize=16)

axes[0,0].plot(time_vec, tote_fc[0,:],'r',label='fc err')
axes[0,0].set_ylabel('$h$',fontsize=18)
#axes[0].text(1, 1.2*axlim0, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[0,nz_index].mean(axis=-1),rmse_an[0,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[0].text(1, 1.1*axlim0, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[0,nz_index].mean(axis=-1),rmse_fc[0,nz_index].mean(axis=-1)), fontsize=12, color='r')
axes[0,0].set_ylim([0,1.3*axlim0])
axes[0,0].legend(loc = 1, fontsize='small')
axes[0,0].set_title("Ensemble Forecast Error")

axes[1,0].plot(time_vec, tote_fc[1,:],'r')
axes[1,0].set_ylabel('$u$',fontsize=18)
#axes[1].text(1, 1.2*axlim1, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[1,nz_index].mean(axis=-1),rmse_an[1,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[1].text(1, 1.1*axlim1, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[1,nz_index].mean(axis=-1),rmse_fc[1,nz_index].mean(axis=-1)), fontsize=12, color='r')
axes[1,0].set_ylim([0,1.3*axlim1])

axes[2,0].plot(time_vec, tote_fc[2,:],'r') #
axes[2,0].set_ylabel('$r$',fontsize=18)
axes[2,0].set_xlabel('Assim. time $T$',fontsize=14)
#axes[2].text(1, 1.2*axlim2, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[2,nz_index].mean(axis=-1),rmse_an[2,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[2].text(1, 1.1*axlim2, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[2,nz_index].mean(axis=-1),rmse_fc[2,nz_index].mean(axis=-1)), fontsize=12, color='r')
axes[2,0].set_ylim([0,1.3*axlim2])

axes[0,1].plot(time_vec, tote_fc[0,:]/tote_fc[0,0],'r',label='fc err')
axes[0,1].set_ylabel('$h$',fontsize=18)
#axes[0].text(1, 1.2*axlim0, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[0,nz_index].mean(axis=-1),rmse_an[0,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[0].text(1, 1.1*axlim0, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[0,nz_index].mean(axis=-1),rmse_fc[0,nz_index].mean(axis=-1)), fontsize=12, color='r')
#axes[0].set_ylim([0,1.3*axlim0])
#axes[0,1].legend(loc = 1, fontsize='small')
#axes[0,1].set_title("Ensemble Forecast Error")

axes[1,1].plot(time_vec, tote_fc[1,:]/tote_fc[1,0],'r')
axes[1,1].set_ylabel('$u$',fontsize=18)
#axes[1].text(1, 1.2*axlim1, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[1,nz_index].mean(axis=-1),rmse_an[1,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[1].text(1, 1.1*axlim1, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[1,nz_index].mean(axis=-1),rmse_fc[1,nz_index].mean(axis=-1)), fontsize=12, color='r')
#axes[1].set_ylim([0,1.3*axlim1])

axes[2,1].plot(time_vec, tote_fc[2,:]/tote_fc[2,0],'r') #
axes[2,1].set_ylabel('$r$',fontsize=18)
axes[2,1].set_xlabel('Assim. time $T$',fontsize=14)
#axes[2].text(1, 1.2*axlim2, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[2,nz_index].mean(axis=-1),rmse_an[2,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[2].text(1, 1.1*axlim2, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[2,nz_index].mean(axis=-1),rmse_fc[2,nz_index].mean(axis=-1)), fontsize=12, color='r')
#axes[2].set_ylim([0,1.3*axlim2])
#
name = '/err_evol_T'+str(T0)+'.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '
'''
###########################################################################
###########################################################################

print ' '
print ' PLOT : RMS ERRORS (for each ensemble...)'
print ' '
frac = 0.15
axlim0 = np.max(np.max(error_fc[0,:,:]))
axlim1 = np.max(np.max(error_fc[1,:,:]))
axlim2 = np.max(np.max(error_fc[2,:,:]))

fig, axes = plt.subplots(3, 2, figsize=(12,12))
#plt.suptitle("Domain-averaged error measures  (N = %s): \n [od, loc, inf] = [%s, %s, %s]" % (n_ens, o_d[i], loc[j], inf[k]),fontsize=16)
for ii in range(0,n_ens):
    axes[0,0].plot(time_vec, error_fc[0,ii,:],'r',alpha=frac)
axes[0,0].plot(time_vec, error_fc[0,:,:].mean(axis=0),'k',linewidth=2.)
axes[0,0].set_ylabel('$h$',fontsize=18)
#axes[0].text(1, 1.2*axlim0, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[0,nz_index].mean(axis=-1),rmse_an[0,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[0].text(1, 1.1*axlim0, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[0,nz_index].mean(axis=-1),rmse_fc[0,nz_index].mean(axis=-1)), fontsize=12, color='r')
#axes[0,0].set_ylim([0,1.3*axlim0])
#axes[0,0].legend(loc = 1, fontsize='small')
axes[0,0].set_title("Forecast Error, E(T)")

for ii in range(0,n_ens):
    axes[1,0].plot(time_vec, error_fc[1,ii,:],'r',alpha=frac)
axes[1,0].plot(time_vec, error_fc[1,:,:].mean(axis=0),'k',linewidth=2.)
axes[1,0].set_ylabel('$u$',fontsize=18)
#axes[1].text(1, 1.2*axlim1, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[1,nz_index].mean(axis=-1),rmse_an[1,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[1].text(1, 1.1*axlim1, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[1,nz_index].mean(axis=-1),rmse_fc[1,nz_index].mean(axis=-1)), fontsize=12, color='r')
axes[1,0].set_ylim([0,1.3*axlim1])

for ii in range(0,n_ens):
    axes[2,0].plot(time_vec, error_fc[2,ii,:],'r',alpha=frac) #
axes[2,0].plot(time_vec, error_fc[2,:,:].mean(axis=0),'k',linewidth=2.)
axes[2,0].set_ylabel('$r$',fontsize=18)
axes[2,0].set_xlabel('Assim. time $T$',fontsize=14)
#axes[2].text(1, 1.2*axlim2, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[2,nz_index].mean(axis=-1),rmse_an[2,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[2].text(1, 1.1*axlim2, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[2,nz_index].mean(axis=-1),rmse_fc[2,nz_index].mean(axis=-1)), fontsize=12, color='r')
axes[2,0].set_ylim([0,1.3*axlim2])

for ii in range(0,n_ens):
    axes[0,1].plot(time_vec, error_fc[0,ii,:]/error_fc[0,ii,0],'r',alpha=frac)
ed = error_fc[0,:,:]/error_fc[0,:,0].reshape(n_ens,1)
print np.shape(ed)
axes[0,1].plot(time_vec, ed[:,:].mean(axis=0),'k',linewidth=2.)
axes[0,1].plot(time_vec,2*np.ones(len(time_vec)),'k--',linewidth=2.)
axes[0,1].set_ylabel('$h$',fontsize=18)
#axes[0].text(1, 1.2*axlim0, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[0,nz_index].mean(axis=-1),rmse_an[0,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[0].text(1, 1.1*axlim0, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[0,nz_index].mean(axis=-1),rmse_fc[0,nz_index].mean(axis=-1)), fontsize=12, color='r')
#axes[0].set_ylim([0,1.3*axlim0])
#axes[0,1].legend(loc = 1, fontsize='small')
axes[0,1].set_title("Relative Forecast Error, E(T)/E(0)")

for ii in range(0,n_ens):
    axes[1,1].plot(time_vec, error_fc[1,ii,:]/error_fc[1,ii,0],'r',alpha=frac)
ed = error_fc[1,:,:]/error_fc[1,:,0].reshape(n_ens,1)
print np.shape(ed)
axes[1,1].plot(time_vec, ed[:,:].mean(axis=0),'k',linewidth=2.)
axes[1,1].plot(time_vec,2*np.ones(len(time_vec)),'k--',linewidth=2.)
axes[1,1].set_ylabel('$u$',fontsize=18)
#axes[1].text(1, 1.2*axlim1, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[1,nz_index].mean(axis=-1),rmse_an[1,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[1].text(1, 1.1*axlim1, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[1,nz_index].mean(axis=-1),rmse_fc[1,nz_index].mean(axis=-1)), fontsize=12, color='r')
#axes[1].set_ylim([0,1.3*axlim1])

for ii in range(0,n_ens):
    axes[2,1].plot(time_vec, error_fc[2,ii,:]/error_fc[2,ii,0],'r',alpha=frac) #
ed = error_fc[2,:,:]/error_fc[2,:,0].reshape(n_ens,1)
print np.shape(ed)
axes[2,1].plot(time_vec, ed[:,:].mean(axis=0),'k',linewidth=2.)
axes[2,1].plot(time_vec,2*np.ones(len(time_vec)),'k--',linewidth=2.)
axes[2,1].set_ylabel('$r$',fontsize=18)
axes[2,1].set_xlabel('Assim. time $T$',fontsize=14)
#axes[2].text(1, 1.2*axlim2, '$(SPR,RMSE)_{an} = (%.3g,%.3g)$' %(spr_an[2,nz_index].mean(axis=-1),rmse_an[2,nz_index].mean(axis=-1)), fontsize=12, color='b')
#axes[2].text(1, 1.1*axlim2, '$(SPR,RMSE)_{fc} = (%.3g,%.3g)$' %(spr_fc[2,nz_index].mean(axis=-1),rmse_fc[2,nz_index].mean(axis=-1)), fontsize=12, color='r')
#axes[2].set_ylim([0,1.3*axlim2])
#
name = '/err_evol_ens_T'+str(T0)+'.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '

###########################################################################

print ' '
print ' PLOT : CRPS'
print ' '

axlim0 = np.max(crps_fc[0,:])
axlim1 = np.max(crps_fc[1,:])
axlim2 = np.max(crps_fc[2,:])

fig, axes = plt.subplots(3, 1, figsize=(7,12))
#plt.suptitle("Domain-averaged CRPS  (N = %s): \n [od, loc, inf] = [%s, %s, %s]" % (n_ens, o_d[i], loc[j], inf[k]),fontsize=16)

axes[0].plot(time_vec, crps_fc[0,:],'r',label='fc') # spread
axes[0].set_ylabel('$h$',fontsize=18)
#axes[0].text(1, 1.2*axlim0, '$CRPS_{an} = %.3g$' %crps_an[0,nz_index].mean(axis=-1), fontsize=12, color='b')
#axes[0].text(1, 1.1*axlim0, '$CRPS_{fc} = %.3g$' %crps_fc[0,nz_index].mean(axis=-1), fontsize=12, color='r')
axes[0].set_ylim([0,1.3*axlim0])
axes[0].legend(loc = 1, fontsize='small')
axes[0].set_title("Continuous Ranked Probability Score")

axes[1].plot(time_vec, crps_fc[1,:],'r',label='fc') # spread
axes[1].set_ylabel('$u$',fontsize=18)
#axes[1].text(1, 1.2*axlim1, '$CRPS_{an} = %.3g$' %crps_an[1,nz_index].mean(axis=-1), fontsize=12, color='b')
#axes[1].text(1, 1.1*axlim1, '$CRPS_{fc} = %.3g$' %crps_fc[1,nz_index].mean(axis=-1), fontsize=12, color='r')
axes[1].set_ylim([0,1.3*axlim1])

axes[2].plot(time_vec, crps_fc[2,:],'r',label='fc') # spread
axes[2].set_ylabel('$r$',fontsize=18)
axes[2].set_xlabel('Assim. time $T$',fontsize=14)
#axes[2].text(1, 1.2*axlim2, '$CRPS_{an} = %.3g$' %crps_an[2,nz_index].mean(axis=-1), fontsize=12, color='b')
#axes[2].text(1, 1.1*axlim2, '$CRPS_{fc} = %.3g$' %crps_fc[2,nz_index].mean(axis=-1), fontsize=12, color='r')
axes[2].set_ylim([0,1.3*axlim2])

name = '/crps_fc_T'+str(T0)+'.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '


###########################################################################
print ' '
print ' CALCULATE : Error-doubling times...'
print ' '
err_doub = np.empty((Neq,n_ens))
for jj in range(0,Neq):
    for ii in range(0,n_ens):
        try:
            err_doub[jj,ii] = np.where(error_fc[jj,ii,:]/error_fc[jj,ii,0]>2.)[0][0]
        except:
            err_doub[jj,ii] = float('nan')
print err_doub[0,:]
print err_doub[1,:]
print err_doub[2,:]
print ' '
print ' SAVE : Error-doubling times...'
print ' '
np.save(str(dirn+'/err_doub_T'+str(T0)),err_doub)

print ' '
print ' PLOT : Error-doubling time histograms'
print ' '
fig, axes = plt.subplots(3, 1, figsize=(7,12))
for kk in range(0,Neq):
    axes[kk].hist(err_doub[kk,:], bins = range(1,25))
    axes[kk].set_xlim([1,24])
name = '/err_doub_hist_T'+str(T0)+'.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '


###########################################################################

