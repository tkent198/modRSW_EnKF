#######################################################################
# Investigating computation and structure of model error and candidate Q matrices
# (T. Kent: amttk@leeds.ac.uk)
#######################################################################
'''
    5.9.16
'''
import math as m
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import os
import errno


##################################################################
# LOCALISATION PROPERTIES
################################################################

# LOAD DATA FROM GIVEN DIRECTORY
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

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

# outer loop pars
o_d = [20,40]
loc = [1.5, 2.5, 3.5, 1e-10] # loc = 1e-10 = no localistaion
inf = [1.01, 1.05, 1.1]

X = np.load(str(dirn+'/X_array.npy')) # fc ensembles
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
Y_obs = np.load(str(dirn+'/Y_obs_array.npy')) # obs ensembles
OI = np.load(str(dirn+'/OI.npy')) # obs ensembles

print 'X_array shape (n_d,n_ens,T)      : ', np.shape(X)
print 'X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)
print 'Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)
print 'Y_obs_array shape (p,n_ens,T)    : ', np.shape(Y_obs)
print 'OI shape (Neq + 1,T)             : ', np.shape(OI)

##################################################################

Neq = Neq = np.shape(OI)[0] - 1
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
hv_mask = range(3*Nk_fc,4*Nk_fc)

# masks for locating obs locations
row_vec = range(obs_dens,n_d+1,obs_dens)
obs_mask = np.array(row_vec[0:n_obs/Neq])-1
h_obs_mask = range(0,n_obs/Neq)
hu_obs_mask = range(n_obs/Neq,2*n_obs/Neq)
hr_obs_mask = range(2*n_obs/Neq,3*n_obs/Neq)
hv_obs_mask = range(3*n_obs/Neq,4*n_obs/Neq)

##################################################################

# compute means and deviations
Xbar = np.empty(np.shape(X))
Xdev = np.empty(np.shape(X))
Xanbar = np.empty(np.shape(X))
Xandev = np.empty(np.shape(X))
Xdev_tr = np.empty(np.shape(X))
Xandev_tr = np.empty(np.shape(X))
Q_array = np.empty((n_d,n_d,t_an))

ONE = np.ones([n_ens,n_ens])
ONE = ONE/n_ens # NxN array with elements equal to 1/N
for T in time_vec[1:]:
    Xbar[:,:,T] = np.dot(X[:,:,T],ONE) # fc mean
    Xdev[:,:,T] = X[:,:,T] - Xbar[:,:,T] # fc deviations from mean
    Xdev_tr[:,:,T] = X[:,:,T] - X_tr[:,:,T] # fc deviations from truth
    Xanbar[:,:,T] = np.dot(Xan[:,:,T],ONE) # an mean
    Xandev[:,:,T] = Xan[:,:,T] - Xanbar[:,:,T] # an deviations from mean
    Xandev_tr[:,:,T] = Xan[:,:,T] - X_tr[:,:,T] # an deviations from truth
    Q_array[:,:,T] = np.dot(Xdev_tr[:,:,T], Xdev_tr[:,:,T].T)/(n_ens-1)


T = time_vec[48]

## covariance matrix
Pf = np.dot(Xdev[:,:,T], Xdev[:,:,T].T)
Pf = Pf/(n_ens - 1)
print 'max Pf value: ', np.max(Pf)
print 'min Pf value: ', np.min(Pf)
print ' '
## true covariance matrix
Pf_tr = np.dot(Xdev_tr[:,:,T], Xdev_tr[:,:,T].T)
Pf_tr = Pf_tr/(n_ens - 1)
print 'max Pf_tr value: ', np.max(Pf_tr)
print 'min Pf_tr value: ', np.min(Pf_tr)

## correlation matrix
Cf = np.corrcoef(Xdev[:,:,T])
Cf_tr = np.corrcoef(Xdev_tr[:,:,T])

## Q matrix
QT = Q_array[:,:,T]
print np.shape(QT)
#Q_ave = Q_array[:,:,0:2*T].mean(axis=2)
Q_ave = Q_array[:,:,:].mean(axis=2)
print np.shape(Q_ave)
Q = Q_ave

q = np.random.multivariate_normal(np.zeros(600), Q, n_ens)
q = q.T
q2 = np.random.multivariate_normal(np.zeros(600), QT, n_ens)
q2 = q2.T
print 'q shape: ', np.shape(q)
print ' '
print 'max Q ave value: ', np.max(Q)
print 'max Q t-dep. value: ', np.max(QT)
print ''
print 'min Q ave value: ', np.min(Q)
print 'min Q t-dep. value: ', np.min(QT)

alph=0.2

print ' *** PLOT: noise sampled from Q ***'
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.plot(range(0,Nk_fc), q[0:200,0],'g',label='h',alpha=alph)
axes.plot(range(0,Nk_fc), q[200:400,0],'r',label='u',alpha=alph)
axes.plot(range(0,Nk_fc), q[400:600,0],'b',label='r',alpha=alph)
axes.plot(range(0,Nk_fc), q[0:200,1:],'g',alpha=alph)
axes.plot(range(0,Nk_fc), q[200:400,1:],'r',alpha=alph)
axes.plot(range(0,Nk_fc), q[400:600,1:],'b',alpha=alph)
#axes.plot(range(0,Nk_fc), q2[0:200,1],'g',linewidth=2)
#axes.plot(range(0,Nk_fc), q2[200:400,1],'r',linewidth=2)
#axes.plot(range(0,Nk_fc), q2[400:600,1],'b',linewidth=2)
axes.set_ylim([-1.5,1.5])
axes.set_xlabel('x')
axes.set_title('model error noise')
axes.legend(loc = 1)

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.plot(range(0,Nk_fc), q2[0:200,0],'g',label='h',alpha=alph)
axes.plot(range(0,Nk_fc), q2[200:400,0],'r',label='u',alpha=alph)
axes.plot(range(0,Nk_fc), q2[400:600,0],'b',label='r',alpha=alph)
axes.plot(range(0,Nk_fc), q2[0:200,1:],'g',alpha=alph)
axes.plot(range(0,Nk_fc), q2[200:400,1:],'r',alpha=alph)
axes.plot(range(0,Nk_fc), q2[400:600,1:],'b',alpha=alph)
#axes.plot(range(0,Nk_fc), q2[0:200,1],'g',linewidth=2)
#axes.plot(range(0,Nk_fc), q2[200:400,1],'r',linewidth=2)
#axes.plot(range(0,Nk_fc), q2[400:600,1],'b',linewidth=2)
axes.set_ylim([-1.5,1.5])
axes.set_xlabel('x')
axes.set_title('model error noise')
axes.legend(loc = 1)

print ' *** PLOT: Q matrix ***'
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
im = axes.imshow(QT,cmap=plt.cm.RdBu,vmin=-0.05, vmax=0.05)
fig.colorbar(im)

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
im = axes.imshow(Q,cmap=plt.cm.RdBu,vmin=-0.05, vmax=0.05)
fig.colorbar(im)
plt.show()

'''
print ' *** PLOT: comparing Pf with true Pf *** '
print 'Max Difference Pf = ', np.max(Pf - Pf_tr)
cpar = 0.05
tick_loc = [0.5*Nk_fc,1.5*Nk_fc,2.5*Nk_fc]
tick_lab = [r'$h$',r'$hu$',r'$hr$']

fig, axes = plt.subplots(1, 3, figsize=(20,10))

im=axes[0].imshow(Pf,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
axes[0].set_xticks(tick_loc)
axes[0].set_yticks(tick_loc)
axes[0].set_xticklabels(tick_lab,fontsize=14)
axes[0].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(axes[1,0].get_xticklines(),visible=False) # invisible ticks
#plt.setp(axes[1,0].get_yticklines(),visible=False)
axes[0].set_title(r'$P^f_e$')

im=axes[1].imshow(Pf_tr,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
axes[1].set_xticks(tick_loc)
axes[1].set_yticks(tick_loc)
axes[1].set_xticklabels(tick_lab,fontsize=14)
axes[1].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(fig.axes[0,1].get_xticklines(),visible=False) # invisible ticks
#plt.setp(fig.axes[0,1].get_yticklines(),visible=False)
axes[1].set_title(r'$P^f_e$ truth (i.e., Q)')

im=axes[2].imshow(Pf - Pf_tr,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
axes[2].set_xticks(tick_loc)
axes[2].set_yticks(tick_loc)
axes[2].set_xticklabels(tick_lab,fontsize=14)
axes[2].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(fig.axes[0,1].get_xticklines(),visible=False) # invisible ticks
#plt.setp(fig.axes[0,1].get_yticklines(),visible=False)
axes[2].set_title('Difference')

im.set_clim(-cpar,cpar)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()

print ' *** PLOT: comparing Cf with true Cf *** '
print 'Max Difference Cf = ', np.max(Cf - Cf_tr)
cpar = 1.0
tick_loc = [0.5*Nk_fc,1.5*Nk_fc,2.5*Nk_fc]
tick_lab = [r'$h$',r'$hu$',r'$hr$']

fig, axes = plt.subplots(1, 3, figsize=(20,10))

im=axes[0].imshow(Cf,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
axes[0].set_xticks(tick_loc)
axes[0].set_yticks(tick_loc)
axes[0].set_xticklabels(tick_lab,fontsize=14)
axes[0].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(axes[1,0].get_xticklines(),visible=False) # invisible ticks
#plt.setp(axes[1,0].get_yticklines(),visible=False)
axes[0].set_title(r'$C^f_e$')

im=axes[1].imshow(Cf_tr,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
axes[1].set_xticks(tick_loc)
axes[1].set_yticks(tick_loc)
axes[1].set_xticklabels(tick_lab,fontsize=14)
axes[1].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(fig.axes[0,1].get_xticklines(),visible=False) # invisible ticks
#plt.setp(fig.axes[0,1].get_yticklines(),visible=False)
axes[1].set_title(r'$C^f_e$ truth')

im=axes[2].imshow(Cf - Cf_tr,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
axes[2].set_xticks(tick_loc)
axes[2].set_yticks(tick_loc)
axes[2].set_xticklabels(tick_lab,fontsize=14)
axes[2].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(fig.axes[0,1].get_xticklines(),visible=False) # invisible ticks
#plt.setp(fig.axes[0,1].get_yticklines(),visible=False)
axes[2].set_title('Difference')

im.set_clim(-cpar,cpar)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
'''
