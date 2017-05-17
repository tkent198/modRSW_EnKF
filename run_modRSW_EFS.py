#######################################################################
# Ensemble forecasts for 1.5D SWEs with rain variable and topography
#               (T. Kent: mmtk@leeds.ac.uk)
#######################################################################

'''
    25.09.2016
    
    Script runs ensemble forecasts, initialised from analysed ensembles at a given time.
    
    Specify: directory, ijk, T0, Tfc
    
    '''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt


##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from parameters import * # module storing fixed parameters
from f_modRSW import make_grid, step_forward_topog, time_step, ens_forecast, ens_forecast_topog
from f_enkf_modRSW import analysis_step_enkf
from create_readme import create_readme


i=1
j=3
k=0

#################################################################
# directory for loading/saving
#################################################################
cwd = os.getcwd()
dirname = '/addinfv12_4dres'
dirn = str(cwd+dirname+dirname+str(i+1)+str(j+1)+str(k+1))


##################################################################
# Mesh generation for forecasts
##################################################################

fc_grid =  make_grid(Nk_fc,L) # forecast

Kk_fc = fc_grid[0]
x_fc = fc_grid[1]
xc_fc = fc_grid[2]

# masks for locating model variables in state vector
h_mask = range(0,Nk_fc)
hu_mask = range(Nk_fc,2*Nk_fc)
hr_mask = range(2*Nk_fc,3*Nk_fc)

##LOAD SAVED DATA
U = np.load(str(dirn+'/U.npy'))
X_array = np.load(str(dirn+'/Xan_array.npy')) # load ANALYSIS ensembles to sample ICs
X_tr = np.load(str(dirn+'/X_tr_array.npy'))
B = np.load(str(dirn+'/B.npy'))
#X_EFS = np.load(str(dirn+'/X_EFS_array_T24.npy'))

T = 9 # to locate IC from saved data
Tfc = 24 # length of forecast (hrs)
tmax = Tfc*tmeasure

X0 = X_array[:,:,T]
n_d = np.shape(X0)[0]
X_fc_array = np.empty([n_d,n_ens,Tfc+1])
X_fc_array[:,:,0] = X0


#X0_tr = X_tr[:,0,T]
#print np.shape(X0_tr)
#X0_tr = X0_tr.reshape(n_d,1)
#print np.shape(X0_tr)


##################################################################
#  Apply initial conditions
##################################################################
print ' '
print '---------------------------------------------------'
print '---------      ICs: load from saved data     ---------'
print '---------------------------------------------------'
print ' '

X0[hu_mask,:] = X0[hu_mask,:]*X0[h_mask,:]
X0[hr_mask,:] = X0[hr_mask,:]*X0[h_mask,:]

#X0_tr[hu_mask,0] = X0_tr[hu_mask,0]*X0_tr[h_mask,0]
#X0_tr[hr_mask,0] = X0_tr[hr_mask,0]*X0_tr[h_mask,0]
#U0_tr = X0_tr[:,0].reshape(Neq,Nk_fc)
#print np.shape(U0_tr)

U0ens = np.empty([Neq,Nk_fc,n_ens])
for N in range(0,n_ens):
    U0ens[:,:,N] = X0[:,N].reshape(Neq,Nk_fc)

print ' '
print 'Check ICs...'
print ' '

fig, axes = plt.subplots(3, 1, figsize=(8,8))
axes[0].plot(xc_fc, U0ens[0,:,:]+B.reshape(len(xc_fc),1), 'b')
#axes[0].plot(xc_fc, U0_tr[0,:]+B.reshape(len(xc_fc),1), 'g')
axes[0].plot(xc_fc, B, 'k', linewidth=2.)
axes[0].set_ylabel('$h_0(x)$',fontsize=18)
#axes[0].set_ylim([0,np.max(U0ens[0,:,:])])
axes[1].plot(xc_fc, U0ens[1,:,:], 'b')
#axes[1].plot(xc_fc, U0_tr[1,:], 'g')
axes[1].set_ylabel('$hu_0(x)$',fontsize=18)
axes[2].plot(xc_fc, U0ens[2,:,:], 'b')
#axes[2].plot(xc_fc, U0_tr[2,:], 'g')
axes[2].set_ylabel('$hr_0(x)$',fontsize=18)
axes[2].set_xlabel('$x$',fontsize=18)
#plt.interactive(True)
plt.show() # use block=False?

##################################################################
#  Integrate ensembles forward in time until obs. is available   #
##################################################################
print ' '
print '-------------------------------------------------'
print '     ------ ENSEMBLE FORECAST SYSTEM ------      '
print '-------------------------------------------------'
print ' '

##if from start...
U = U0ens
index=0
print 'tmax =', tmax
print 'dtmeasure =', dtmeasure
print 'tmeasure = ', tmeasure

while tmeasure-dtmeasure <= tmax:
    print ' '
    print '----------------------------------------------'
    print '---------- ENSEMBLE FORECAST: START ----------'
    print '----------------------------------------------'
    print ' '
    
    num_cores_tot = mp.cpu_count()
    num_cores_use = num_cores_tot-1
    print 'Number of cores available:', num_cores_tot
    print 'Number of cores used:', num_cores_use
    
    print 'Starting ensemble integrations from time =', index,' to', index+1
    
    print  ' *** Started: ', str(datetime.now())
    
    print np.shape(U)
    
    pool = mp.Pool(processes=num_cores_use)
    
    mp_out = [pool.apply_async(ens_forecast_topog, args=(N, U, B, Nk_fc, Kk_fc, assim_time, index, tmeasure)) for N in range(0,n_ens)]
    
    UU = [p.get() for p in mp_out]
    
    pool.close()
    
    print ' All ensembles integrated forward from time =', index,' to', index+1
    print ' *** Ended: ', str(datetime.now())
    print np.shape(UU)
    
    UU =np.swapaxes(UU,0,1)
    UU =np.swapaxes(UU,1,2)
    
    print np.shape(UU)
    
    print ' '
    print '----------------------------------------------'
    print '------------- FORECAST STEP: END -------------'
    print '----------------------------------------------'
    print ' '
    
    ##################################################################
    #  calculate crps and error at this time then integrate forward  #
    ##################################################################
    
    
    # transform to X for saving
    UU[1:,:,:] = UU[1:,:,:]/UU[0,:,:]
    for N in range(0,n_ens):
        X_fc_array[:,N,index+1] = UU[:,:,N].flatten()
    print ' '
    print ' Saving DATA at time =', index+1
    print ' '
    np.save(str(dirn+'/X_EFS_array_T'+str(T)),X_fc_array)

    UU[1:,:,:] = UU[1:,:,:]*UU[0,:,:]
    
    U = UU # update U for next integration

    # on to next assim_time
    index = index + 1
    tmeasure = tmeasure + dtmeasure




##################################################################
#                       END OF PROGRAM                           #
##################################################################


