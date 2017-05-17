##################################################################
#--------- Plot err doub time histogram from EFS stats -----------
#                   (T. Kent: mmtk@leeds.ac.uk)
##################################################################
'''
    Plots error doubling time histograms from saved data <err_doub_Tn.npy>
    
    '''


## generic modules
import os
import errno
import numpy as np
import matplotlib.pyplot as plt

## custom modules
from parameters import *

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

#T0 = [12,13,14,15,16,17,18,19,20,21,22,23,24]
T0 = range(9,25)
err_doub = np.empty((Neq,len(T0)*n_ens))
#
for i in range(0,len(T0)):
    err_doub[:,i*n_ens:(i+1)*n_ens] = np.load(str(dirn+'/err_doub_T'+str(T0[i]))+'.npy')
#



###############################################################################
print ' '
print ' PLOT : Error-doubling time histograms...'
print ' '

# bin width = 1hr
print '... with bin width = 1hr '

fig, axes = plt.subplots(3, 1, figsize=(7,12))
plt.suptitle('Error-doubling time: histogram',fontsize=18)
for kk in range(0,Neq):
    hist, bins = np.histogram(err_doub[kk,:], bins = np.linspace(1,25,25))
    axes[kk].hist(err_doub[kk,:], bins = np.linspace(1,24,24))
    axes[kk].set_xlim([1,25])
    axes[kk].set_ylabel('Count',fontsize=14)
    axes[kk].text(18, 0.8*np.max(hist), '$Mean = %.3g$' %np.nanmean(err_doub[kk,:]), fontsize=16, color='b')
    axes[kk].text(18, 0.7*np.max(hist), '$Median = %.3g$' %np.median(err_doub[kk,:][~np.isnan(err_doub[kk,:])]), fontsize=16, color='b')
    axes[kk].text(18, 0.9*np.max(hist), '$Total = %.3g / %.3g$' %(np.sum(hist), len(T0)*n_ens), fontsize=16, color='b')
axes[2].set_xlabel('Time (hrs)',fontsize=14)

name = '/err_doub_hist_ave.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '


# bin width = 2hr
print '... with bin width = 2hr '

fig, axes = plt.subplots(3, 1, figsize=(7,12))
plt.suptitle('Error-doubling time: histogram',fontsize=18)
for kk in range(0,Neq):
    hist, bins = np.histogram(err_doub[kk,:], bins = np.linspace(1,25,13))
    axes[kk].hist(err_doub[kk,:], bins = np.linspace(1,25,13))
    axes[kk].set_xlim([1,25])
    axes[kk].set_ylabel('Count',fontsize=14)
    axes[kk].text(18, 0.8*np.max(hist), '$Mean = %.3g$' %np.nanmean(err_doub[kk,:]), fontsize=16, color='b')
    axes[kk].text(18, 0.7*np.max(hist), '$Median = %.3g$' %np.median(err_doub[kk,:][~np.isnan(err_doub[kk,:])]), fontsize=16, color='b')
    axes[kk].text(18, 0.9*np.max(hist), '$Total = %.3g / %.3g$' %(np.sum(hist), len(T0)*n_ens), fontsize=16, color='b')
axes[2].set_xlabel('Time (hrs)',fontsize=14)

name = '/err_doub_hist_ave2.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '

#np.isnan(X[0,:]).sum()
#print np.sum(hist)
###########################################################################