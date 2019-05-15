##################################################################
#------------- Plotting routine for saved test data -------------
#                   (T. Kent:  amttk@leeds.ac.uk)
##################################################################
'''
Uses simulation data generated in run_modRSW.py test cases. Goal: compare simulations with different Nk.
    '''

# generic modules
import os
import errno
import numpy as np
import matplotlib.pyplot as plt

# custom module
from parameters import *
from f_modRSW import make_grid

# LOAD DATA FROM GIVEN DIRECTORY
cwd = os.getcwd()
dirname = str('/test_model')
dirn = str(cwd+dirname)

U_fc = np.load(str(dirn+'/U_array_Nk200.npy')) # Nk = 200
B_fc = np.load(str(dirn+'/B_Nk200.npy')) # Nk = 200

U_tr = np.load(str(dirn+'/U_array_Nk800.npy')) # Nk = 800
B_tr = np.load(str(dirn+'/B_Nk800.npy')) # Nk = 800

Nk_fc = np.shape(U_fc)[1]
Nk_tr = np.shape(U_tr)[1]

grid_fc =  make_grid(Nk_fc,L)
xc_fc = grid_fc[2]
grid_tr =  make_grid(Nk_tr,L)
xc_tr = grid_tr[2]

time = np.shape(U_fc)[2]

for T in range(0,time):

    ### 6 panel subplot
    print ' *** PLOT: trajectories for Nk = 200, 800 ***'
    fig, axes = plt.subplots(Neq, 2, figsize=(15,10))
    plt.suptitle("Comparing trajectories for Nk = 200 (left) and Nk = 800 (right)",fontsize=18)

    axes[0,0].plot(xc_fc, U_fc[0,:,T]+B_fc, 'b')
    axes[0,0].plot(xc_fc,Hc*np.ones(len(xc_fc)),'r:')
    axes[0,0].plot(xc_fc,Hr*np.ones(len(xc_fc)),'r:')
    axes[0,0].plot(xc_fc, B_fc, 'k', linewidth=2.0)
    axes[0,0].set_ylim([0,3])
    axes[0,0].set_ylabel('$h(x)+b(x)$',fontsize=18)

    axes[0,1].plot(xc_tr, U_tr[0,:,T]+B_tr, 'b')
    axes[0,1].plot(xc_tr,Hc*np.ones(len(xc_tr)),'r:')
    axes[0,1].plot(xc_tr,Hr*np.ones(len(xc_tr)),'r:')
    axes[0,1].plot(xc_tr, B_tr, 'k', linewidth=2.0)
    axes[0,1].set_ylim([0,3])

    axes[1,0].plot(xc_fc, U_fc[1,:,T]/U_fc[0,:,T], 'b')
    axes[1,0].set_ylim([0,2])
    axes[1,0].set_ylabel('$u(x)$',fontsize=18)

    axes[1,1].plot(xc_tr, U_tr[1,:,T]/U_tr[0,:,T], 'b')
    axes[1,1].set_ylim([0,2])

    axes[2,0].plot(xc_fc, U_fc[2,:,T]/U_fc[0,:,T], 'b')
    axes[2,0].set_ylim([-0.05,0.15])
    axes[2,0].plot(xc_fc,np.zeros(len(xc_fc)),'k--')
    axes[2,0].set_ylabel('$r(x)$',fontsize=18)
    axes[2,0].set_xlabel('$x$',fontsize=18)

    axes[2,1].plot(xc_tr, U_tr[2,:,T]/U_tr[0,:,T], 'b')
    axes[2,1].set_ylim([-0.05,0.15])
    axes[2,1].plot(xc_tr,np.zeros(len(xc_tr)),'k--')
    axes[2,1].set_xlabel('$x$',fontsize=18)

    fname = "/t%d_compare.png" %T
    f_name = str(dirn+fname)
    plt.savefig(f_name)
    print ' *** %s at time level %d saved to %s' %(f_name,T,dirn)
