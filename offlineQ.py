#######################################################################
# Investigating computation and structure of model error and candidate Q matrices
# (T. Kent: mmtk@leeds.ac.uk)
#######################################################################
'''
    
    
    '''
import math as m
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import os
import errno


##################################################################


# LOAD DATA FROM GIVEN DIRECTORY
## e.g. if i,j,k... etc a coming from outer loop:
i=1
j=3
k=0
##
##
dirname = '/addinfv12_4dres'
##
cwd = os.getcwd()
dirn = str(cwd+dirname+dirname+str(i+1)+str(j+1)+str(k+1))
figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise


X = np.load(str(dirn+'/X_array.npy')) # fc ensembles
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles


print 'X_array shape (n_d,n_ens,T)      : ', np.shape(X)
print 'X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)
print 'Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)

##################################################################

n_d = np.shape(X)[0]
Nk_fc = n_d/Neq
Kk_fc = 1./Nk_fc
n_ens = np.shape(X)[1]
t_an = np.shape(X)[2]
time_vec = range(0,t_an)


##################################################################

# compute means and deviations
Xbar = np.empty(np.shape(X))
Xdev_tr = np.empty(np.shape(X))
Q_array = np.empty((n_d,n_d,t_an))

ONE = np.ones([n_ens,n_ens])
ONE = ONE/n_ens # NxN array with elements equal to 1/N

for T in time_vec[1:]:
    Xbar[:,:,T] = np.dot(X[:,:,T],ONE) # fc mean
    Xdev_tr[:,:,T] = X[:,:,T] - X_tr[:,:,T] # fc deviations from truth
    Q_array[:,:,T] = np.dot(Xdev_tr[:,:,T], Xdev_tr[:,:,T].T)/(n_ens-1)

T = time_vec[30]

## time-dep Q matrix (at time T)
QT = Q_array[:,:,T]
print np.shape(QT)

## static Q matrix (averaged over all times T)
Q_ave = Q_array[:,:,:].mean(axis=2)
print np.shape(Q_ave)
Q = Q_ave

q = np.random.multivariate_normal(np.zeros(600), Q, n_ens)
q = q.T

qt = np.random.multivariate_normal(np.zeros(600), QT, n_ens)
qt = qt.T

print 'q shape: ', np.shape(q)
print ' '
print 'max Q static value: ', np.max(Q)
print 'max Q t-dep. value: ', np.max(QT)
print ''
print 'min Q static value: ', np.min(Q)
print 'min Q t-dep. value: ', np.min(QT)

alph=0.2

fig, axes = plt.subplots(2, 2, figsize=(10,10))

axes[0,0].plot(range(0,Nk_fc), q[0:200,0],'g',label='h',alpha=alph)
axes[0,0].plot(range(0,Nk_fc), q[200:400,0],'r',label='u',alpha=alph)
axes[0,0].plot(range(0,Nk_fc), q[400:600,0],'b',label='r',alpha=alph)
axes[0,0].plot(range(0,Nk_fc), q[0:200,1:],'g',alpha=alph)
axes[0,0].plot(range(0,Nk_fc), q[200:400,1:],'r',alpha=alph)
axes[0,0].plot(range(0,Nk_fc), q[400:600,1:],'b',alpha=alph)
axes[0,0].set_ylim([-1.5,1.5])
axes[0,0].set_xlabel('x')
axes[0,0].set_title('samples from static Q')
axes[0,0].legend(loc = 1)

axes[1,0].plot(range(0,Nk_fc), qt[0:200,0],'g',label='h',alpha=alph)
axes[1,0].plot(range(0,Nk_fc), qt[200:400,0],'r',label='u',alpha=alph)
axes[1,0].plot(range(0,Nk_fc), qt[400:600,0],'b',label='r',alpha=alph)
axes[1,0].plot(range(0,Nk_fc), qt[0:200,1:],'g',alpha=alph)
axes[1,0].plot(range(0,Nk_fc), qt[200:400,1:],'r',alpha=alph)
axes[1,0].plot(range(0,Nk_fc), qt[400:600,1:],'b',alpha=alph)
axes[1,0].set_ylim([-1.5,1.5])
axes[1,0].set_xlabel('x')
axes[1,0].set_title('samples from time-dep. Q')
axes[1,0].legend(loc = 1)

axes[0,1].imshow(Q,cmap=plt.cm.RdBu,vmin=-0.05, vmax=0.05)
axes[0,1].set_title('static Q')

axes[1,1].imshow(QT,cmap=plt.cm.RdBu,vmin=-0.05, vmax=0.05)
axes[1,1].set_title('time-dep. Q')

plt.show()

np.save(str(cwd+'/Q_offline'),Q)
