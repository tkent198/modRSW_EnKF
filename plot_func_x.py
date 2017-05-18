##################################################################
#--------------- Plotting routines for saved data ---------------
#                   (T. Kent: mmtk@leeds.ac.uk)
##################################################################
'''
<plotting_routine>
Plotting routine: loads saved data in specific directories and produces plots at a given assimilation time. To use, specify (1) dir_name, (2) combination of parameters ijk, (3) time level T = time_vec[ii], i.e., choose ii.
'''


# generic modules 
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

##e.g. if i,j,k... etc are coming from outer loop:
i=0
j=0
k=0
##
##
dirname = '/test_enkf'
##

# LOAD DATA FROM GIVEN DIRECTORY
cwd = os.getcwd()
dirn = str(cwd+dirname+dirname+str(i+1)+str(j+1)+str(k+1))
figsdir = str(dirn+'/figs')


# parameters for outer loop
loc = [1e-10]
add_inf = [0.2]
inf = [1.01, 1.05, 1.1]

## plot at assimilation cycle ii
ii = 3
##

# make fig directory (if it doesn't already exist)
try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

# load data
B = np.load(str(dirn+'/B.npy')) #topography
X = np.load(str(dirn+'/X_array.npy')) # fc ensembles
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
Y_obs = np.load(str(dirn+'/Y_obs_array.npy')) # obs ensembles
OI = np.load(str(dirn+'/OI.npy')) # obs ensembles

# print shape of data arrays to terminal (sanity check)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print 'X_array shape (n_d,n_ens,T)      : ', np.shape(X) 
print 'X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr) 
print 'Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan) 
print 'Y_obs_array shape (p,n_ens,T)    : ', np.shape(Y_obs) 
print 'OI shape (Neq + 1,T)             : ', np.shape(OI) 
#print 'OI = ', OI
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

print 'time_vec = ', time_vec

T = time_vec[ii]

print ' *** Plotting at time T level = ', T
print ' *** Assim. time: ', assim_time[T]

# masks for locating model variables in state vector
h_mask = range(0,Nk_fc)
hu_mask = range(Nk_fc,2*Nk_fc)
hr_mask = range(2*Nk_fc,3*Nk_fc)
#hv_mask = range(3*Nk_fc,4*Nk_fc)

# masks for locating obs locations
row_vec = range(obs_dens,n_d+1,obs_dens)
obs_mask = np.array(row_vec[0:n_obs/Neq])-1
h_obs_mask = range(0,n_obs/Neq)
hu_obs_mask = range(n_obs/Neq,2*n_obs/Neq)
hr_obs_mask = range(2*n_obs/Neq,3*n_obs/Neq)
#hv_obs_mask = range(3*n_obs/Neq,4*n_obs/Neq)

##################################################################

# compute means and deviations
Xbar = np.empty(np.shape(X))
Xdev = np.empty(np.shape(X))
Xanbar = np.empty(np.shape(X))
Xandev = np.empty(np.shape(X))
Xdev_tr = np.empty(np.shape(X))
Xandev_tr = np.empty(np.shape(X))

ONE = np.ones([n_ens,n_ens])
ONE = ONE/n_ens # NxN array with elements equal to 1/N
for ii in time_vec:
    Xbar[:,:,ii] = np.dot(X[:,:,ii],ONE) # fc mean
    Xdev[:,:,ii] = X[:,:,ii] - Xbar[:,:,ii] # fc deviations from mean
    Xdev_tr[:,:,ii] = X[:,:,ii] - X_tr[:,:,ii] # fc deviations from truth
    Xanbar[:,:,ii] = np.dot(Xan[:,:,ii],ONE) # an mean
    Xandev[:,:,ii] = Xan[:,:,ii] - Xanbar[:,:,ii] # an deviations from mean
    Xandev_tr[:,:,ii] = Xan[:,:,ii] - X_tr[:,:,ii] # an deviations from truth    
##################################################################

frac = 0.15 # alpha value for translucent plotting
xc = np.linspace(Kk_fc/2,L-Kk_fc/2,Nk_fc)
'''
print ' *** PLOT: truth, ensembles and ensemble mean BEFORE analysis ***'
    
fig, axes = plt.subplots(Neq, 1, figsize=(7,12))
plt.suptitle("Trajectories BEFORE analysis  (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)
axes[0].plot(xc, X[h_mask,1:,T]+B.reshape(len(xc),1), 'b',alpha=frac)
axes[0].plot(xc, B, 'k', linewidth=2.0)
l1 = axes[0].plot(xc, X[h_mask,0,T]+B, 'b',alpha=frac,label="fc. ens.")
l2 = axes[0].plot(xc, X_tr[h_mask,0,T]+B, 'g',label="Truth",linewidth=2.0)
l3 = axes[0].plot(xc, Xbar[h_mask,0,T]+B, 'r',label="Ens. mean",linewidth=2.0)
axes[0].plot(xc,Hc*np.ones(len(xc)),'k:')
axes[0].plot(xc,Hr*np.ones(len(xc)),'k:')
axes[0].set_ylim([0,0.5+np.max(X_tr[h_mask,:,T])])
axes[0].set_ylabel('$h(x)+b(x)$',fontsize=18)
axes[0].legend(loc=0,fontsize='small')

axes[1].plot(xc, X[hu_mask,:,T], 'b',alpha=frac)
axes[1].plot(xc, X_tr[hu_mask,0,T], 'g',linewidth=2.0)
axes[1].plot(xc, Xbar[hu_mask,0,T], 'r',linewidth=2.0)
axes[1].set_ylabel('$u(x)$',fontsize=18)

axes[2].plot(xc, X[hr_mask,:,T], 'b',alpha=frac)
axes[2].plot(xc, X_tr[hr_mask,0,T], 'g',linewidth=2.0)
axes[2].plot(xc, Xbar[hr_mask,0,T], 'r',linewidth=2.0)
axes[2].set_ylabel('$r(x)$',fontsize=18)
axes[2].set_ylim([-0.1,0.5])
axes[2].set_xlabel('$x$',fontsize=18)

name_f1 = "/T%d_f1.pdf" %T
f_name_f1 = str(figsdir+name_f1)
plt.savefig(f_name_f1)
print ' *** %s at time level %d saved to %s' %(name_f1,T,figsdir)

##################################################################

### 4 panel subplot for evolution of 4 vars
print ' *** PLOT: trajectories AFTER analysis ***'
fig, axes = plt.subplots(Neq, 1, figsize=(7,12))
plt.suptitle("Trajectories AFTER analysis w/ forecast ensembles (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)
axes[0].plot(xc, X[h_mask,1:,T]+B.reshape(len(xc),1), 'b',alpha=frac)
axes[0].plot(xc, X[h_mask,0,T]+B, 'b',alpha=frac,label="fc. ens.")
axes[0].plot(xc, Xbar[h_mask,0,T]+B, 'r',label="Ens. mean")
#axes[0,0].plot(xc, X_tr[h_mask], 'g',label="Truth")
axes[0].plot(xc[obs_mask], Y_obs[h_obs_mask,0,T]+B[obs_mask], 'go',linewidth=2.0,label="Single obs.")
axes[0].plot(xc, Xanbar[h_mask,0,T]+B, 'c',linewidth=2.0,label="Analysis")
axes[0].plot(xc,Hc*np.ones(len(xc)),'k:')
axes[0].plot(xc,Hr*np.ones(len(xc)),'k:')
axes[0].plot(xc, B, 'k', linewidth=2.0)
axes[0].set_ylim([0,0.5+np.max(X_tr[h_mask,:,T])])
axes[0].set_ylabel('$h(x)+b(x)$',fontsize=18)
axes[0].legend(loc = 1, fontsize='small')

axes[1].plot(xc, X[hu_mask,:,T], 'b',alpha=frac)
axes[1].plot(xc, Xbar[hu_mask,0,T], 'r')
#axes[0,1].plot(xc, X_tr[hu_mask], 'g')
axes[1].plot(xc[obs_mask], Y_obs[hu_obs_mask,0,T], 'go',linewidth=2.0)
axes[1].plot(xc, Xanbar[hu_mask,0,T], 'c',linewidth=2.0)
axes[1].set_ylabel('$u(x)$',fontsize=18)

axes[2].plot(xc, X[hr_mask,:,T], 'b',alpha=frac)
axes[2].plot(xc, Xbar[hr_mask,0,T], 'r')
#axes[1,1].plot(xc, X_tr[hr_mask], 'g')
axes[2].plot(xc[obs_mask], Y_obs[hr_obs_mask,0,T], 'go',linewidth=2.0)
axes[2].plot(xc, Xanbar[hr_mask,0,T], 'c',linewidth=2.0)
axes[2].set_ylabel('$r(x)$',fontsize=18)
axes[2].set_ylim([-0.1,0.5])
axes[2].set_xlabel('$x$',fontsize=18)
axes[2].text(0.025, 0.4, 'OI (all obs.) = %s' %float('%.3g' % OI[0,T]),fontsize=12, color='k')

name_f2 = "/T%d_f2.pdf" %T
f_name_f2 = str(figsdir+name_f2)
plt.savefig(f_name_f2)
print ' *** %s at time level %d saved to %s' %(name_f2,T,figsdir)

##################################################################

### 4 panel subplot for evolution of 4 vars incl. analysis ensembles
print ' *** PLOT: trajectories AFTER analysis ***'
fig, axes = plt.subplots(Neq, 1, figsize=(7,12))
plt.suptitle("Trajectories AFTER analysis w/ analysis ensembles (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)

#axes[0,0].plot(xc, X[h_mask,1:], 'b',alpha=frac)
#axes[0,0].plot(xc, X[h_mask,0], 'b',alpha=frac,label="fc. ens.")
axes[0].plot(xc, Xbar[h_mask,0,T]+B, 'r',label="Ens. mean")
axes[0].plot(xc, X_tr[h_mask,0,T]+B, 'g',label="Truth")
axes[0].plot(xc[obs_mask], Y_obs[h_obs_mask,0,T]+B[obs_mask], 'go',linewidth=2.0,label="Single obs.")
axes[0].plot(xc, Xan[h_mask,1:,T]+B.reshape(len(xc),1), 'b',alpha=frac)
axes[0].plot(xc, Xan[h_mask,0,T]+B, 'b',alpha=frac,label="an. ens.")
axes[0].plot(xc, Xanbar[h_mask,0,T]+B, 'c',linewidth=2.0,label="Analysis")
axes[0].plot(xc,Hc*np.ones(len(xc)),'k:')
axes[0].plot(xc,Hr*np.ones(len(xc)),'k:')
axes[0].plot(xc, B, 'k', linewidth=2.0)
axes[0].set_ylim([0,0.5+np.max(X_tr[h_mask,:,T])])
axes[0].set_ylabel('$h(x)+b(x)$',fontsize=18)
axes[0].legend(loc = 1, fontsize='small')

#axes[0,1].plot(xc, X[hu_mask,:], 'b',alpha=frac)
axes[1].plot(xc, Xbar[hu_mask,0,T], 'r')
axes[1].plot(xc, X_tr[hu_mask,:,T], 'g')
axes[1].plot(xc[obs_mask], Y_obs[hu_obs_mask,0,T], 'go',linewidth=2.0)
axes[1].plot(xc, Xan[hu_mask,:,T], 'b',alpha=frac)
axes[1].plot(xc, Xanbar[hu_mask,0,T], 'c',linewidth=2.0)
axes[1].set_ylabel('$u(x)$',fontsize=18)

#axes[1,1].plot(xc, X[hr_mask,:], 'b',alpha=frac)
axes[2].plot(xc, Xbar[hr_mask,0,T], 'r')
axes[2].plot(xc, X_tr[hr_mask,:,T], 'g')
axes[2].plot(xc[obs_mask], Y_obs[hr_obs_mask,0,T], 'go',linewidth=2.0)
axes[2].plot(xc, Xan[hr_mask,:,T], 'b',alpha=frac)
axes[2].plot(xc, Xanbar[hr_mask,0,T], 'c',linewidth=2.0)
axes[2].set_ylabel('$r(x)$',fontsize=18)
axes[2].set_ylim([-0.1,0.5])
axes[2].set_xlabel('$x$',fontsize=18)
axes[2].text(0.025, 0.4, 'OI (all obs.) = %s' %float('%.3g' % OI[0,T]),fontsize=12, color='k')

name_f3 = "/T%d_f3.pdf" %T
f_name_f3 = str(figsdir+name_f3)
plt.savefig(f_name_f3)
print ' *** %s at time level %d saved to %s' %(name_f3,T,figsdir)

'''
##################################################################

#print ' CHECK:', xc[obs_mask], Y_obs[h_obs_mask,:,T].mean(axis=1)+B[obs_mask]

### 6 panel subplot for evolution of 3 vars: fc and an
print ' *** PLOT: trajectories BEFORE AND AFTER analysis ***'
fig, axes = plt.subplots(Neq, 2, figsize=(15,10))
#plt.suptitle("Ensemble trajectories (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)

axes[0,0].plot(xc, X[h_mask,1:,T]+B.reshape(len(xc),1), 'b',alpha=frac)
axes[0,0].plot(xc, X[h_mask,0,T]+B, 'b',alpha=frac,label="fc. ens.")
axes[0,0].plot(xc, Xbar[h_mask,0,T]+B, 'r',label="Ens. mean")
axes[0,0].plot(xc, X_tr[h_mask,0,T]+B, 'g',label="Truth")
#axes[0,0].plot(xc[obs_mask], Y_obs[h_obs_mask,:,T].mean(axis=1)+B[obs_mask], 'go',linewidth=2.0,label="Obs.")
axes[0,0].errorbar(xc[obs_mask], Y_obs[h_obs_mask,:,T].mean(axis=1)+B[obs_mask], ob_noise[0], fmt='go',linewidth=2.0,label="Obs.")
#axes[0,0].errorbar(xc[obs_mask], Y_obs[h_obs_mask,:,T].mean(axis=1)+B[obs_mask], [Y_obs[h_obs_mask,:,T].min(axis=1), Y_obs[h_obs_mask,:,T].max(axis=1)], fmt='go', linewidth=2.0, label="Obs.")
axes[0,0].plot(xc,Hc*np.ones(len(xc)),'k:')
axes[0,0].plot(xc,Hr*np.ones(len(xc)),'k:')
axes[0,0].plot(xc, B, 'k', linewidth=2.0)
axes[0,0].set_ylim([0,0.1+np.max(X_tr[h_mask,:,T]+B)])
axes[0,0].set_ylabel('$h(x)+b(x)$',fontsize=18)
#axes[0,0].legend(loc = 1, fontsize='small')
axes[0,0].legend(loc = 1)


axes[0,1].plot(xc, Xan[h_mask,1:,T]+B.reshape(len(xc),1), 'b',alpha=frac)
axes[0,1].plot(xc, Xan[h_mask,0,T]+B, 'b',alpha=frac,label="an. ens.")
axes[0,1].plot(xc, Xanbar[h_mask,0,T]+B, 'c',linewidth=2.0,label="Analysis")
axes[0,1].plot(xc, X_tr[h_mask,0,T]+B, 'g',label="Truth")
#axes[0,1].plot(xc[obs_mask], Y_obs[h_obs_mask,:,T].mean(axis=1)+B[obs_mask], 'go',linewidth=2.0,label="Obs.")
axes[0,1].errorbar(xc[obs_mask], Y_obs[h_obs_mask,:,T].mean(axis=1)+B[obs_mask], ob_noise[0], fmt='go',linewidth=2.0,label="Obs.")
#axes[0,1].errorbar(xc[obs_mask], Y_obs[h_obs_mask,:,T].mean(axis=1)+B[obs_mask], [Y_obs[h_obs_mask,:,T].min(axis=1), Y_obs[h_obs_mask,:,T].max(axis=1)], fmt='go', linewidth=2.0, label="Obs.")
axes[0,1].plot(xc,Hc*np.ones(len(xc)),'k:')
axes[0,1].plot(xc,Hr*np.ones(len(xc)),'k:')
axes[0,1].plot(xc, B, 'k', linewidth=2.0)
axes[0,1].set_ylim([0,0.1+np.max(X_tr[h_mask,:,T]+B)])
#axes[0,1].set_ylabel('$h(x)+b(x)$',fontsize=18)
#axes[0,1].legend(loc = 1, fontsize='small')
axes[0,1].legend(loc = 1)

axes[1,0].plot(xc, X[hu_mask,:,T], 'b',alpha=frac)
axes[1,0].plot(xc, Xbar[hu_mask,0,T], 'r')
axes[1,0].plot(xc, X_tr[hu_mask,:,T], 'g')
axes[1,0].plot(xc[obs_mask], Y_obs[hu_obs_mask,:,T].mean(axis=1), 'go',linewidth=2.0)
axes[1,0].errorbar(xc[obs_mask], Y_obs[hu_obs_mask,:,T].mean(axis=1), ob_noise[1], fmt='go',linewidth=2.0)
#axes[1,0].errorbar(xc[obs_mask], Y_obs[hu_obs_mask,:,T].mean(axis=1), [Y_obs[hu_obs_mask,:,T].min(axis=1), Y_obs[hu_obs_mask,:,T].max(axis=1)], fmt='go', linewidth=2.0)
axes[1,0].set_ylabel('$u(x)$',fontsize=18)

axes[1,1].plot(xc, Xan[hu_mask,:,T], 'b',alpha=frac)
axes[1,1].plot(xc, Xanbar[hu_mask,0,T], 'c',linewidth=2.0)
axes[1,1].plot(xc, X_tr[hu_mask,:,T], 'g')
#axes[1,1].plot(xc[obs_mask], Y_obs[hu_obs_mask,:,T].mean(axis=1), 'go',linewidth=2.0)
axes[1,1].errorbar(xc[obs_mask], Y_obs[hu_obs_mask,:,T].mean(axis=1), ob_noise[1], fmt='go',linewidth=2.0)
#axes[1,1].errorbar(xc[obs_mask], Y_obs[hu_obs_mask,:,T].mean(axis=1), [Y_obs[hu_obs_mask,:,T].min(axis=1), Y_obs[hu_obs_mask,:,T].max(axis=1)], fmt='go', linewidth=2.0)
#axes[1,1].set_ylabel('$u(x)$',fontsize=18)

axes[2,0].plot(xc, X[hr_mask,:,T], 'b',alpha=frac)
axes[2,0].plot(xc, Xbar[hr_mask,0,T], 'r')
axes[2,0].plot(xc, X_tr[hr_mask,:,T], 'g')
#axes[2,0].plot(xc[obs_mask], Y_obs[hr_obs_mask,:,T].mean(axis=1), 'go',linewidth=2.0)
axes[2,0].errorbar(xc[obs_mask], Y_obs[hr_obs_mask,:,T].mean(axis=1), ob_noise[2],fmt='go',linewidth=2.0)
#axes[2,0].errorbar(xc[obs_mask], Y_obs[hr_obs_mask,:,T].mean(axis=1), [Y_obs[hr_obs_mask,:,T].min(axis=1), Y_obs[hr_obs_mask,:,T].max(axis=1)], fmt='go', linewidth=2.0)
axes[2,0].plot(xc,np.zeros(len(xc)),'k')
axes[2,0].set_ylabel('$r(x)$',fontsize=18)
axes[2,0].set_ylim([-0.025,0.02+np.max(X_tr[hr_mask,0,T])])
axes[2,0].set_xlabel('$x$',fontsize=18)

axes[2,1].plot(xc, Xan[hr_mask,:,T], 'b',alpha=frac)
axes[2,1].plot(xc, Xanbar[hr_mask,0,T], 'c',linewidth=2.0)
axes[2,1].plot(xc, X_tr[hr_mask,:,T], 'g')
#axes[2,1].plot(xc[obs_mask], Y_obs[hr_obs_mask,:,T].mean(axis=1), 'go',linewidth=2.0)
axes[2,1].errorbar(xc[obs_mask], Y_obs[hr_obs_mask,:,T].mean(axis=1), ob_noise[2], fmt='go',linewidth=2.0)
#axes[2,1].errorbar(xc[obs_mask], Y_obs[hr_obs_mask,:,T].mean(axis=1), [Y_obs[hr_obs_mask,:,T].min(axis=1), Y_obs[hr_obs_mask,:,T].max(axis=1)], fmt='go', linewidth=2.0)
axes[2,1].plot(xc,np.zeros(len(xc)),'k')
#axes[2,1].set_ylabel('$r(x)$',fontsize=18)
axes[2,1].set_ylim([-0.025,0.02+np.max(X_tr[hr_mask,0,T])])
axes[2,1].set_xlabel('$x$',fontsize=18)

name_f6 = "/T%d_f6.pdf" %T
f_name_f6 = str(figsdir+name_f6)
plt.savefig(f_name_f6)
print ' *** %s at time level %d saved to %s' %(name_f6,T,figsdir)

##################################################################
###                       ERRORS                              ####
##################################################################

# ANALYSIS
an_err = Xanbar[:,0,T] - X_tr[:,0,T] # an_err = analysis ens. mean - truth
an_err2 = an_err**2
# domain-averaged mean errors
an_ME_h = an_err[h_mask].mean() 
an_ME_hu = an_err[hu_mask].mean()
an_ME_hr = an_err[hr_mask].mean()
# domain-averaged absolute errors
an_absME_h = np.absolute(an_err[h_mask])
an_absME_hu = np.absolute(an_err[hu_mask])
an_absME_hr = np.absolute(an_err[hr_mask])

#an_rmse = [an_rmse_h, an_rmse_hu, an_rmse_hv, an_rmse_hr]

Pa = np.dot(Xandev[:,:,T],np.transpose(Xandev[:,:,T]))
Pa = Pa/(n_ens - 1) # analysis covariance matrix
#Ca = np.corrcoef(Xandev) # analysis correlation matrix
var_an = np.diag(Pa)

Pa_tr = np.dot(Xandev_tr[:,:,T],np.transpose(Xandev_tr[:,:,T]))
Pa_tr = Pa_tr/(n_ens - 1) # fc covariance matrix w.r.t truth
#Cf = np.corrcoef(Xdev) # fc correlation matrix w.r.t truth
var_ant = np.diag(Pa_tr)

## FORECAST
fc_err = Xbar[:,0,T] - X_tr[:,0,T] # fc_err = ens. mean - truth
fc_err2 = fc_err**2
# domain-averaged mean errors
fc_ME_h = fc_err[h_mask].mean()
fc_ME_hu = fc_err[hu_mask].mean()
fc_ME_hr = fc_err[hr_mask].mean()
# domain-averaged absolute errors
fc_absME_h = np.absolute(fc_err[h_mask])
fc_absME_hu = np.absolute(fc_err[hu_mask])
fc_absME_hr = np.absolute(fc_err[hr_mask])

# as a vector
#fc_rmse = [fc_rmse_h, fc_rmse_hu, fc_rmse_hv, fc_rmse_hr]

# cov matrix
Pf = np.dot(Xdev[:,:,T],np.transpose(Xdev[:,:,T]))
Pf = Pf/(n_ens - 1) # fc covariance matrix
#Cf = np.corrcoef(Xdev) # fc correlation matrix
var_fc = np.diag(Pf)

Pf_tr = np.dot(Xdev_tr[:,:,T],np.transpose(Xdev_tr[:,:,T]))
Pf_tr = Pf_tr/(n_ens - 1) # fc covariance matrix w.r.t. truth
#Cf = np.corrcoef(Xdev_tr) # fc correlation matrix w.r.t truth
var_fct = np.diag(Pf_tr)

# fc/an
ME_ratio_h = np.sqrt(fc_err2[h_mask])/np.sqrt(an_err2[h_mask])
ME_ratio_hu = np.sqrt(fc_err2[hu_mask])/np.sqrt(an_err2[hu_mask])
ME_ratio_hr = np.sqrt(fc_err2[hr_mask])/np.sqrt(an_err2[hr_mask])
# fc - an
ME_diff_h = np.sqrt(fc_err2[h_mask])-np.sqrt(an_err2[h_mask])
ME_diff_hu = np.sqrt(fc_err2[hu_mask])-np.sqrt(an_err2[hu_mask])
ME_diff_hr = np.sqrt(fc_err2[hr_mask])-np.sqrt(an_err2[hr_mask])

##################################################################

print ' ' 
print '--------- ERRORS: ---------'
print ' ' 
np.set_printoptions(formatter={'float': '{: 0.3g}'.format})

#print 'Mean error:'
#print 'ME for h: an_err =', an_ME_h, '; fc_err =', fc_ME_h
#print 'ME for hu: an_err =', an_ME_hu, '; fc_err =', fc_ME_hu
#print 'ME for hv: an_err =', an_ME_hv, '; fc_err =', fc_ME_hv
#print 'ME for hr: an_err =', an_ME_hr, '; fc_err =', fc_ME_hr

print 'Abs. mean error:'
print ' ' 
print 'Abs. ME for h: an_ame =', np.mean(an_absME_h), '; fc_ame =', np.mean(fc_absME_h)
print 'Abs. ME for hu: an_ame =', np.mean(an_absME_hu), '; fc_ame =', np.mean(fc_absME_hu)
print 'Abs. ME for hr: an_ame =', np.mean(an_absME_hr), '; fc_ame =', np.mean(fc_absME_hr)
print ' ' 
print ' *** PLOT: error of forecast and analysis ***'
print ' ' 
# position text on plot
pl_h = np.max([fc_absME_h,an_absME_h])
pl_hu = np.max([fc_absME_hu,an_absME_hu])
pl_hr = np.max([fc_absME_hr,an_absME_hr])

fig, axes = plt.subplots(3, 2, figsize=(12,12))
#plt.suptitle("Abs. mean error  (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)

axes[0,0].plot(xc, an_absME_h,'b')
axes[0,0].plot(xc, fc_absME_h,'r')
axes[0,0].set_ylabel('$h(x)$',fontsize=18)
axes[0,0].text(0.025, 1.2*pl_h, '$ME_{an} = %.3g$' %an_absME_h.mean(),fontsize=12, color='b')
axes[0,0].text(0.025, 1.1*pl_h, '$ME_{fc} = %.3g$' %fc_absME_h.mean(),fontsize=12, color='r')
axes[0,0].set_ylim([0,1.3*pl_h])
axes[0,0].set_title("Error: forecast and analysis mean")

axes[1,0].plot(xc, an_absME_hu, 'b')
axes[1,0].plot(xc, fc_absME_hu, 'r')
axes[1,0].set_ylabel('$u(x)$',fontsize=18)
axes[1,0].text(0.025, 1.2*pl_hu, '$ME_{an} = %.3g$' %an_absME_hu.mean(),fontsize=12, color='b')
axes[1,0].text(0.025, 1.1*pl_hu, '$ME_{fc} = %.3g$' %fc_absME_hu.mean(),fontsize=12, color='r')
axes[1,0].set_ylim([0,1.3*pl_hu])

axes[2,0].plot(xc, an_absME_hr, 'b')
axes[2,0].plot(xc, fc_absME_hr, 'r')
axes[2,0].set_ylabel('$r(x)$',fontsize=18)
axes[2,0].set_xlabel('$x$',fontsize=18)
axes[2,0].text(0.025, 1.2*pl_hr, '$ME_{an} = %.3g$' % an_absME_hr.mean(),fontsize=12, color='b')
axes[2,0].text(0.025, 1.1*pl_hr, '$ME_{fc} = %.3g$' % fc_absME_hr.mean(),fontsize=12, color='r')
axes[2,0].set_ylim([0,1.3*pl_hr])

axes[0,1].plot(xc, ME_diff_h, 'g')
axes[0,1].plot(xc,np.zeros(len(xc)),'k:')
#axes[0,1].legend(loc=0,fontsize='small')
axes[0,1].set_title("$\Delta$ ME = ME$_{fc}$ - ME$_{an}$")

axes[1,1].plot(xc, ME_diff_hu, 'g')
axes[1,1].plot(xc,np.zeros(len(xc)),'k:')

axes[2,1].plot(xc, ME_diff_hr, 'g')
axes[2,1].plot(xc,np.zeros(len(xc)),'k:')
axes[2,1].set_xlabel('$x$',fontsize=18)

name_f4 = "/T%d_f4.pdf" %T
f_name_f4 = str(figsdir+name_f4)
plt.savefig(f_name_f4)
print ' *** %s at time level %d saved to %s' %(name_f4,T,figsdir)
ft = 16
# position text on plot
pl_h = np.max([np.sqrt(var_fc[h_mask]),fc_absME_h])
pl_hu = np.max([np.sqrt(var_fc[hu_mask]),fc_absME_hu])
pl_hr = np.max([np.sqrt(var_fc[hr_mask]),fc_absME_hr])

# domain-averaged errors
an_spr_h = np.mean(np.sqrt(var_an[h_mask]))
an_rmse_h = np.mean(np.sqrt(var_ant[h_mask]))
fc_spr_h = np.mean(np.sqrt(var_fc[h_mask]))
fc_rmse_h = np.mean(np.sqrt(var_fct[h_mask]))

an_spr_hu = np.mean(np.sqrt(var_an[hu_mask]))
an_rmse_hu = np.mean(np.sqrt(var_ant[hu_mask]))
fc_spr_hu = np.mean(np.sqrt(var_fc[hu_mask]))
fc_rmse_hu = np.mean(np.sqrt(var_fct[hu_mask]))

an_spr_hr = np.mean(np.sqrt(var_an[hr_mask]))
an_rmse_hr = np.mean(np.sqrt(var_ant[hr_mask]))
fc_spr_hr = np.mean(np.sqrt(var_fc[hr_mask]))
fc_rmse_hr = np.mean(np.sqrt(var_fct[hr_mask]))


fig, axes = plt.subplots(3, 2, figsize=(12,12))
#plt.suptitle("Ensemble spread and error  (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)

axes[0,0].plot(xc, np.sqrt(var_fc[h_mask]),'r',label='fc spread') # spread
axes[0,0].plot(xc, fc_absME_h,'r--',label='fc err') # rmse
axes[0,0].plot(xc, np.sqrt(var_an[h_mask]),'b',label='an spread')
axes[0,0].plot(xc, an_absME_h,'b--',label='an err')
axes[0,0].set_ylabel('$h(x)$',fontsize=18)
axes[0,0].text(0.025, 1.2*pl_h, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_h,np.mean(an_absME_h)), fontsize=ft, color='b')
axes[0,0].text(0.025, 1.1*pl_h, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_h,np.mean(fc_absME_h)), fontsize=ft, color='r')
axes[0,0].set_ylim([0,1.3*pl_h])
#axes[0,0].legend(loc = 1)
#axes[0,0].set_title("Spread and RMSE: fc and an")

axes[1,0].plot(xc, np.sqrt(var_fc[hu_mask]), 'r')
axes[1,0].plot(xc, fc_absME_hu, 'r--')
axes[1,0].plot(xc, np.sqrt(var_an[hu_mask]), 'b')
axes[1,0].plot(xc, an_absME_hu , 'b--')
axes[1,0].set_ylabel('$u(x)$',fontsize=18)
axes[1,0].text(0.025, 1.2*pl_hu, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_hu,an_absME_hu.mean()), fontsize=ft, color='b')
axes[1,0].text(0.025, 1.1*pl_hu, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_hu,fc_absME_hu.mean()), fontsize=ft, color='r')
axes[1,0].set_ylim([0,1.3*pl_hu])

axes[2,0].plot(xc, np.sqrt(var_fc[hr_mask]), 'r')
axes[2,0].plot(xc, fc_absME_hr , 'r--')
axes[2,0].plot(xc, np.sqrt(var_an[hr_mask]), 'b')
axes[2,0].plot(xc, an_absME_hr , 'b--')
axes[2,0].set_ylabel('$r(x)$',fontsize=18)
axes[2,0].set_xlabel('$x$',fontsize=18)
axes[2,0].text(0.025, 1.2*pl_hr, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_hr,an_absME_hr.mean() ), fontsize=ft, color='b')
axes[2,0].text(0.025, 1.1*pl_hr, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_hr,fc_absME_hr.mean() ), fontsize=ft, color='r')
axes[2,0].set_ylim([0,1.3*pl_hr])

axes[0,1].plot(xc, fc_absME_h - np.sqrt(var_fc[h_mask]), 'r',label='fc: err  - spr')
axes[0,1].plot(xc, an_absME_h - np.sqrt(var_an[h_mask]), 'b',label='an: err - spr')
axes[0,1].plot(xc,np.zeros(len(xc)),'k:')
axes[0,1].legend(loc=0)

axes[1,1].plot(xc, fc_absME_hu - np.sqrt(var_fc[hu_mask]), 'r')
axes[1,1].plot(xc, an_absME_hu - np.sqrt(var_an[hu_mask]), 'b')
axes[1,1].plot(xc, np.zeros(len(xc)),'k:')

axes[2,1].plot(xc, fc_absME_hr - np.sqrt(var_fc[hr_mask]), 'r')
axes[2,1].plot(xc, an_absME_hr - np.sqrt(var_an[hr_mask]), 'b')
axes[2,1].plot(xc, np.zeros(len(xc)),'k:')
axes[2,1].set_xlabel('$x$',fontsize=18)

name_f5 = "/T%d_f5.pdf" %T
f_name_f5 = str(figsdir+name_f5)
plt.savefig(f_name_f5)
print ' *** %s at time level %d saved to %s' %(name_f5,T,figsdir)




###################################################################


CRPS_fc = np.empty((Neq,Nk_fc))
CRPS_an = np.empty((Neq,Nk_fc))

for ii in h_mask:
    CRPS_fc[0,ii] = crps_calc(X[ii,:,T],X_tr[ii,0,T])
    CRPS_fc[1,ii] = crps_calc(X[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
    CRPS_fc[2,ii] = crps_calc(X[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])
    CRPS_an[0,ii] = crps_calc(Xan[ii,:,T],X_tr[ii,0,T])
    CRPS_an[1,ii] = crps_calc(Xan[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
    CRPS_an[2,ii] = crps_calc(Xan[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])

print ' '
print ' PLOT : CRPS'
print ' '

lw = 1. # linewidth
axlim0 = np.max(CRPS_fc[0,:])
axlim1 = np.max(CRPS_fc[1,:])
axlim2 = np.max(CRPS_fc[2,:])
ft = 16
xl = 0.65

fig, axes = plt.subplots(3, 1, figsize=(7,12))
#plt.suptitle("CRPS  (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)

axes[0].plot(xc, CRPS_fc[0,:],'r',linewidth=lw,label='fc')
axes[0].plot(xc, CRPS_an[0,:],'b',linewidth=lw,label='an')
axes[0].set_ylabel('$h(x)$',fontsize=18)
#axes[0].legend(loc = 1)
axes[0].text(xl, 1.2*axlim0, '$CRPS_{an} = %.3g$' %CRPS_an[0,:].mean(axis=-1), fontsize=ft, color='b')
axes[0].text(xl, 1.1*axlim0, '$CRPS_{fc} = %.3g$' %CRPS_fc[0,:].mean(axis=-1), fontsize=ft, color='r')
axes[0].set_ylim([0,1.3*axlim0])
#axes[0].set_xlabel('$x$',fontsize=18)

axes[1].plot(xc, CRPS_fc[1,:],'r',linewidth=lw)
axes[1].plot(xc, CRPS_an[1,:],'b',linewidth=lw)
axes[1].set_ylabel('$u(x)$',fontsize=18)
axes[1].text(xl, 1.2*axlim1, '$CRPS_{an} = %.3g$' %CRPS_an[1,:].mean(axis=-1), fontsize=ft, color='b')
axes[1].text(xl, 1.1*axlim1, '$CRPS_{fc} = %.3g$' %CRPS_fc[1,:].mean(axis=-1), fontsize=ft, color='r')
axes[1].set_ylim([0,1.3*axlim1])
#axes[1].set_xlabel('$x$',fontsize=18)

axes[2].plot(xc, CRPS_fc[2,:],'r',linewidth=lw)
axes[2].plot(xc, CRPS_an[2,:],'b',linewidth=lw)
axes[2].set_ylabel('$r(x)$',fontsize=18)
axes[2].text(xl, 1.2*axlim2, '$CRPS_{an} = %.3g$' %CRPS_an[2,:].mean(axis=-1), fontsize=ft, color='b')
axes[2].text(xl, 1.1*axlim2, '$CRPS_{fc} = %.3g$' %CRPS_fc[2,:].mean(axis=-1), fontsize=ft, color='r')
axes[2].set_ylim([0,1.3*axlim2])
axes[2].set_xlabel('$x$',fontsize=18)


name = "/T%d_f_crps.pdf" %T
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' *** %s at time level %d saved to %s' %(name,T,figsdir)
