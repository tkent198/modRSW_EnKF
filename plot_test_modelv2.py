##################################################################
#------------- Plotting routine for saved test data -------------
#                   (T. Kent:  amttk@leeds.ac.uk)
##################################################################
'''
Uses simulation data generated in run_modRSW.py test cases.
Goal: compare simulations with different Nk.
    '''

# generic modules
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# custom module
from parameters import *
from f_modRSW import make_grid

# LOAD DATA FROM GIVEN DIRECTORY
cwd = os.getcwd()
dirname = str('/test_model')
dirn = str(cwd+dirname)

U_fc1 = np.load(str(dirn+'/U_array_Nk200.npy')) # Nk = 200
B_fc1 = np.load(str(dirn+'/B_Nk200.npy')) # Nk = 200

U_fc2 = np.load(str(dirn+'/U_array_Nk400.npy')) # Nk = 400
B_fc2 = np.load(str(dirn+'/B_Nk400.npy')) # Nk = 400

U_tr = np.load(str(dirn+'/U_array_Nk800.npy')) # Nk = 800
B_tr = np.load(str(dirn+'/B_Nk800.npy')) # Nk = 800

Nk_fc1 = np.shape(U_fc1)[1] #200
Nk_fc2 = np.shape(U_fc2)[1] #400
Nk_tr = np.shape(U_tr)[1] #800

grid_fc1 =  make_grid(Nk_fc1,L)
xc_fc1 = grid_fc1[2]
grid_fc2 =  make_grid(Nk_fc2,L)
xc_fc2 = grid_fc2[2]
grid_tr =  make_grid(Nk_tr,L)
xc_tr = grid_tr[2]

time = np.shape(U_tr)[2]

for T in range(0,time):

    ### 6 panel subplot
    # print ' *** PLOT: trajectories for Nk = 200, 800 ***'
    fig, axes = plt.subplots(Neq, 1, figsize=(8,15))
    # plt.suptitle("Comparing trajectories for Nk = 200 (left) and Nk = 800 (right)",fontsize=18)

    axes[0].plot(xc_tr, U_tr[0,:,T]+B_tr, color='c')
    axes[0].plot(xc_fc2, U_fc2[0,:,T]+B_fc2, color='g')
    axes[0].plot(xc_fc1, U_fc1[0,:,T]+B_fc1, color='b')
    axes[0].plot(xc_fc1,Hc*np.ones(len(xc_fc1)),'k:')
    axes[0].plot(xc_fc1,Hr*np.ones(len(xc_fc1)),'k:')
    axes[0].plot(xc_fc1, B_fc1, 'k', linewidth=2.0)
    axes[0].set_ylim([0,4])
    axes[0].set_ylabel('$h(x)+b(x)$',fontsize=18)
    axins = zoomed_inset_axes(axes[0], 2.2, loc=2)#, bbox_to_anchor=(0.05, 0.1))
    axins.plot(xc_tr, U_tr[0,:,T]+B_tr, color='c')
    axins.plot(xc_fc2, U_fc2[0,:,T]+B_fc2, color='g')
    axins.plot(xc_fc1, U_fc1[0,:,T]+B_fc1, color='b')
    axins.plot(xc_fc1,Hc*np.ones(len(xc_fc1)),'k:')
    axins.plot(xc_fc1,Hr*np.ones(len(xc_fc1)),'k:')
    axins.plot(xc_fc1, B_fc1, 'k', linewidth=2.0)
    x1, x2, y1, y2 = 0.75, 1., 0.95, 2.1 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    mark_inset(axes[0], axins, loc1=1, loc2=3, fc="none", ec="0.5")
    # axes[0,1].plot(xc_tr, U_tr[0,:,T]+B_tr, 'b')
    # axes[0,1].plot(xc_tr,Hc*np.ones(len(xc_tr)),'r:')
    # axes[0,1].plot(xc_tr,Hr*np.ones(len(xc_tr)),'r:')
    # axes[0,1].plot(xc_tr, B_tr, 'k', linewidth=2.0)
    # axes[0,1].set_ylim([0,3])

    axes[1].plot(xc_tr, U_tr[1,:,T]/U_tr[0,:,T], 'c')
    axes[1].plot(xc_fc2, U_fc2[1,:,T]/U_fc2[0,:,T], 'g')
    axes[1].plot(xc_fc1, U_fc1[1,:,T]/U_fc1[0,:,T], 'b')
    axes[1].set_ylim([0.5,2])
    axes[1].set_ylabel('$u(x)$',fontsize=18)

    # axes[1,1].plot(xc_tr, U_tr[1,:,T]/U_tr[0,:,T], 'b')
    # axes[1,1].set_ylim([0,2])

    axes[2].plot(xc_tr, U_tr[2,:,T]/U_tr[0,:,T], 'c')
    axes[2].plot(xc_fc2, U_fc2[2,:,T]/U_fc2[0,:,T], 'g')
    axes[2].plot(xc_fc1, U_fc1[2,:,T]/U_fc1[0,:,T], 'b')
    axes[2].set_ylim([-0.05,0.15])
    axes[2].plot(xc_fc1,np.zeros(len(xc_fc1)),'k--')
    axes[2].set_ylabel('$r(x)$',fontsize=18)
    axes[2].set_xlabel('$x$',fontsize=18)
    axins = zoomed_inset_axes(axes[2], 2.2, loc=2)#, bbox_to_anchor=(0.05, 0.1))
    axins.plot(xc_tr, U_tr[2,:,T]/U_tr[0,:,T], 'c')
    axins.plot(xc_fc2, U_fc2[2,:,T]/U_fc2[0,:,T], 'g')
    axins.plot(xc_fc1, U_fc1[2,:,T]/U_fc1[0,:,T], 'b')
    x1, x2, y1, y2 = 0.75, 1., -0.01, 0.05 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    mark_inset(axes[2], axins, loc1=1, loc2=3, fc="none", ec="0.5")

    # axes[2,1].plot(xc_tr, U_tr[2,:,T]/U_tr[0,:,T], 'b')
    # axes[2,1].set_ylim([-0.05,0.15])
    # axes[2,1].plot(xc_tr,np.zeros(len(xc_tr)),'k--')
    # axes[2,1].set_xlabel('$x$',fontsize=18)

    fname = "/t%d_comparev2.png" %T
    f_name = str(dirn+fname)
    plt.savefig(f_name,dpi=300)
    # print ' *** %s at time level %d saved to %s' %(f_name,T,dirn)
