##################################################################
#- Compute the CRPS at a given time as a function of space       -
#                   (T. Kent: mmtk@leeds.ac.uk)
##################################################################
'''

    '''


# generic modules
import os
import errno
import numpy as np
import matplotlib.pyplot as plt

## custom modules
from parameters import *
from crps_calc_fun import crps_calc

##################################################################

##e.g. if i,j,k... etc are coming from outer loop:
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

# parameters for outer loop
o_d = [20,40]
loc = [1.5, 2.5, 3.5, 0.]
#inf = [1.1, 1.25, 1.5, 1.75]
inf = [1.01, 1.05, 1.1]

## plot at assimilation cycle ii
ii = 36
##

## make fig directory (if it doesn't already exist)
#try:
#    os.makedirs(figsdir)
#except OSError as exception:
#    if exception.errno != errno.EEXIST:
#        raise

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
xc = np.linspace(Kk_fc/2,L-Kk_fc/2,Nk_fc)


print 'time_vec = ', time_vec

T = time_vec[ii]

# masks for locating model variables in state vector
h_mask = range(0,Nk_fc)
hu_mask = range(Nk_fc,2*Nk_fc)
hr_mask = range(2*Nk_fc,3*Nk_fc)

## masks for locating obs locations
#row_vec = range(obs_dens,n_d+1,obs_dens)
#obs_mask = np.array(row_vec[0:n_obs/Neq])-1
#h_obs_mask = range(0,n_obs/Neq)
#hu_obs_mask = range(n_obs/Neq,2*n_obs/Neq)
#hr_obs_mask = range(2*n_obs/Neq,3*n_obs/Neq)

h = X[h_mask,:,T]
ht = X_tr[h_mask,0,T]
print np.shape(h)
print np.shape(ht)

CRPSh = np.empty(len(h_mask))
CRPSu = np.empty(len(h_mask))
CRPSr = np.empty(len(h_mask))

for ii in h_mask:
    CRPSh[ii] = crps_calc(X[ii,:,T],X_tr[ii,0,T])
    CRPSu[ii] = crps_calc(X[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
    CRPSr[ii] = crps_calc(X[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])


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
fig, axes = plt.subplots(3, 1, figsize=(10,7))
plt.suptitle("CRPS  (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)

axes[0].plot(xc, CRPS_fc[0,:],'r',linewidth=2.0)
axes[0].plot(xc, CRPS_an[0,:],'b',linewidth=2.0)
axes[0].set_ylabel('CRPS',fontsize=18)
axes[0].set_ylim([0,np.max(CRPSh)])
axes[0].set_xlim([xc[0],xc[-1]])
#axes.set_xlabel('Assim. time $T$',fontsize=14)

axes[1].plot(xc, CRPS_fc[1,:],'r',linewidth=2.0)
axes[1].plot(xc, CRPS_an[1,:],'b',linewidth=2.0)
axes[1].set_ylabel('CRPS',fontsize=18)
axes[1].set_ylim([0,np.max(CRPSu)])
axes[1].set_xlim([xc[0],xc[-1]])

axes[2].plot(xc, CRPS_fc[2,:],'r',linewidth=2.0)
axes[2].plot(xc, CRPS_an[2,:],'b',linewidth=2.0)
axes[2].set_ylabel('CRPS',fontsize=18)
axes[2].set_ylim([0,np.max(CRPSr)])
axes[2].set_xlim([xc[0],xc[-1]])

plt.show()

print np.mean(CRPSh)
print np.mean(CRPS_fc[0,:])


