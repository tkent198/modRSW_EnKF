#######################################################################
# modRSW with topography (T. Kent: tkent198@gmail.com)
#######################################################################
'''
Given mesh, IC, time paramters, integrates modRSW and plots evolution. 
Useful first check of simulations before use in the EnKF.
'''

#%%%----- Modules used -----%%%'''
#from math import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

##################################################################
#%%%----- Set up -----%%%'''
##################################################################

from parameters import * # module storing fixed parameters
from f_modRSW import step_forward_topog, time_step, make_grid, make_grid_2
from init_cond_modRSW import init_cond_topog_cos, init_cond_topog

##################################################################    
# Mesh generation for forecast and truth resolutions
##################################################################
Nk = 200

grid =  make_grid(Nk,L) # forecast
Kk = grid[0]
x = grid[1]
xc = grid[2]

##################################################################    
#'''%%%----- Apply initial conditions -----%%%'''
##################################################################

# CHOOSE INITIAL PROFILE FROM init_cond_modRSW:
U0, B = init_cond_topog_cos(x,Nk,Neq,H0,L,A,V)

### 4 panel subplot for initial state of 4 vars
fig, axes = plt.subplots(3, 1, figsize=(8,8))
axes[0].plot(xc, U0[0,:]+B, 'b')
axes[0].plot(xc, B, 'k', linewidth=2.)
axes[0].set_ylabel('$h_0(x)$',fontsize=18)
axes[0].set_ylim([0,2*H0])
axes[1].plot(xc, U0[1,:], 'b')
axes[1].set_ylim([-2,2])
axes[1].set_ylabel('$hu_0(x)$',fontsize=18)
axes[2].plot(xc, U0[2,:], 'b')
axes[2].set_ylabel('$hr_0(x)$',fontsize=18)
axes[2].set_ylim([0,0.5])
axes[2].set_xlabel('$x$',fontsize=18)
#plt.interactive(True)
plt.show() # use block=False?
print('Done initial conditions')

##################################################################
#'''%%%----- Define system arrays and time parameters------%%%'''
##################################################################
U = np.empty([Neq,Nk])
U = U0
tn = 0
hour = 0.144
Nmeas = 10.
tmax = Nmeas*hour
dtmeasure = tmax/Nmeas
tmeasure = dtmeasure

##################################################################
#'''%%%----- integrate forward in time until tmax ------%%%'''
##################################################################
print 'Integrating forward...'
while tn < tmax:
    
    dt = time_step(U,Kk,cfl_fc)
    tn = tn + dt

    if tn > tmeasure:
	dt = dt - (tn - tmeasure) + 1e-12
        tn = tmeasure + 1e-12

    U = step_forward_topog(U,B,dt,tn,Nk,Kk) # U(n+1) = M(U(n))
    print 't =',tn

    if tn > tmeasure:
	
        print 'Plotting at time =',tmeasure

        tmeasure = tmeasure + dtmeasure

        fig, axes = plt.subplots(3, 1, figsize=(8,8))
        axes[0].plot(xc, U[0,:]+B, 'b')
        axes[0].plot(xc, B, 'k', linewidth=2.0)
        axes[0].plot(xc,Hc*np.ones(len(xc)),'r:')
        axes[0].plot(xc,Hr*np.ones(len(xc)),'r:')
        axes[0].set_ylim([0,4*H0])
        axes[0].set_ylabel('$h(x)$',fontsize=18)
        axes[1].plot(xc, U[1,:], 'b')
        axes[1].set_ylim([-2,2])
        axes[1].set_ylabel('$hu(x)$',fontsize=18)
        axes[2].plot(xc, U[2,:], 'b')
        axes[2].set_ylabel('$hr(x)$',fontsize=18)
        axes[2].set_ylim([0,0.1])
        axes[2].set_xlabel('$x$',fontsize=18)
        #plt.interactive(True)
        plt.show() # use block=False?


print '***** DONE: end of simulation at time:', tn
print ' '   
print ' -------------- SUMMARY: ------------- '  
print ' ' 
print 'Dynamics:'
print 'Ro =', Ro  
print 'Fr = ', Fr
print '(H_0 , H_c , H_r) =', [H0, Hc, Hr]  
print ' resolution (number of gridcells) =', Nk
print ' '   
print ' ----------- END OF SUMMARY: ---------- '  
print ' '  


##################################################################
#			END OF PROGRAM				 #
##################################################################


