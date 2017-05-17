##################################################################
#----------------- Initial conditions for modRSW -----------------
#                   (T. Kent: mmtk@leeds.ac.uk)
##################################################################

'''
Functions generate different initial conditions described below for modRSW model with
and without bottom topography...

INPUT ARGS:
# x: mesh coords
# Neq: number of equations (variables) - 4 w/o topography, 5 w/ topography
# Nk: no. of cells in mesh
# H0: reference (scaled) height 
# L: length of domain
# A: amplitude
# V: velocity scale

OUTPUT:
# U0: array of initial data, size (Neq,Nk)

##################################################################
DESCRIPTIONS:

Rotation, no topography:

<init_cond_1>
--- sinusiodal waves in h and u, zero v and r.

<init_cond_2>
--- Rossby adj with disturbed height profile:
--- Exact step in h, zero u, v, and r.

<init_cond_3>
--- Rossby adj with disturbed height profile:
--- Smoothed step in h, zero u, v, and r.

<init_cond_4>
--- Rossby adj with disturbed v-velocity profile:
--- Single jet in v, flat h profile, zero u and r.

<init_cond_5>
--- Rossby adj with disturbed v-velocity profile:
--- Double jet in v, flat h profile, zero u and r.

<init_cond_6>
--- Rossby adj with disturbed v-velocity profile:
--- Quadrupel jet in v, flat h profile, zero u and r.

<init_cond_6_1>
--- Rossby adj with disturbed v-velocity profile:
--- Quadrupel jet in v, flat h=1 profile, u = constant \ne 0, and zero r.

Topography, no rotation:

<init_cond_topog>
--- single parabolic ridge

<init_cond_topog4>
--- 4 parabolic ridges

<init_cond_topog_cos>
--- superposition of sinusoids, as used in thesis chapter 6
'''

###############################################################

import numpy as np
 
###############################################################

def init_cond_1(x,Nk,Neq,H0,L,A,V):

    k = 2*np.pi # for sinusoidal waves

    ic1 = H0 + A*np.sin(2*k*x/L)
    ic2 = A*np.sin(1*k*x/L)
    #ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))

    # Define array and fill with first-order FV (piecewise constant) initial data  
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    return U0

###############################################################

def init_cond_2(x,Nk,Neq,H0,L,A,V):
    from f_modRSW import heaviside
# for disturbed height (top-hat) Rossby adj. set up.
# Exact step:
    f1 = heaviside(x-0.25*L)
    f2 = heaviside(x-0.75*L)

    ic1 = H0 + A*(0.5*f1 - 0.5*f2)
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    return U0

###############################################################

def init_cond_3(x,Nk,Neq,H0,L,A,V):

# for disturbed height (top-hat) Rossby adj. set up
# Smoothed step:
    gam = 100 
    f1 = 1-np.tanh(gam*(x-0.75*L))
    f2 = 1-np.tanh(gam*(x-0.25*L))

    ic1 = H0 + A*(0.5*f1 - 0.5*f2)
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    return U0

###############################################################

def init_cond_4(x,Nk,Neq,H0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = H0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
# single jet
    Lj = 0.1*L
    ic4 = V*(1+np.tanh(4*(x-0.5*L)/Lj + 2))*(1-np.tanh(4*(x-0.5*L)/Lj - 2))/4 
    #ic4 = V*(1+np.tanh(4*(x)/Lj + 2))*(1-np.tanh(4*(x)/Lj - 2))/4 
    
# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    return U0

###############################################################

def init_cond_5(x,Nk,Neq,H0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = H0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
 
## double jet
    Lj = 0.1*L
    f1 = V*(1+np.tanh(4*(x-0.75*L)/Lj + 2))*(1-np.tanh(4*(x-0.75*L)/Lj - 2))/4
    f2 = V*(1+np.tanh(4*(x-0.25*L)/Lj + 2))*(1-np.tanh(4*(x-0.25*L)/Lj - 2))/4
    ic4 = f1-f2
    
# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    return U0
    
###############################################################

def init_cond_5_1(x,Nk,Neq,H0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = H0*np.ones(len(x))
    ic2 = 0.5*np.ones(len(x))
    ic3 = np.zeros(len(x))
 
## double jet
    Lj = 0.1*L
    f1 = V*(1+np.tanh(4*(x-0.75*L)/Lj + 2))*(1-np.tanh(4*(x-0.75*L)/Lj - 2))/4
    f2 = V*(1+np.tanh(4*(x-0.25*L)/Lj + 2))*(1-np.tanh(4*(x-0.25*L)/Lj - 2))/4
    ic4 = f1-f2
    
# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    return U0

###############################################################

def init_cond_6(x,Nk,Neq,H0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = H0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
## multiple (>2) jets
    Lj = 0.05
    f3 = (1+np.tanh(4*(x-0.8)/Lj + 2))*(1-np.tanh(4*(x-0.8)/Lj - 2))/4
    f4 = (1+np.tanh(4*(x-0.2)/Lj + 2))*(1-np.tanh(4*(x-0.2)/Lj - 2))/4
    f5 = (1+np.tanh(4*(x-0.6)/Lj + 2))*(1-np.tanh(4*(x-0.6)/Lj - 2))/4
    f6 = (1+np.tanh(4*(x-0.4)/Lj + 2))*(1-np.tanh(4*(x-0.4)/Lj - 2))/4
    #ic4 = V*(f3+f4-f5-f6)
    ic4 = V*(f3-f4+f5-f6)

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    return U0

###############################################################

def init_cond_6_1(x,Nk,Neq,H0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = H0*np.ones(len(x))
    ic2 = 0.5*np.ones(len(x))
    ic3 = np.zeros(len(x))
## multiple (>2) jets
    Lj = 0.05
    f3 = (1+np.tanh(4*(x-0.8)/Lj + 2))*(1-np.tanh(4*(x-0.8)/Lj - 2))/4
    f4 = (1+np.tanh(4*(x-0.2)/Lj + 2))*(1-np.tanh(4*(x-0.2)/Lj - 2))/4
    f5 = (1+np.tanh(4*(x-0.6)/Lj + 2))*(1-np.tanh(4*(x-0.6)/Lj - 2))/4
    f6 = (1+np.tanh(4*(x-0.4)/Lj + 2))*(1-np.tanh(4*(x-0.4)/Lj - 2))/4
    #ic4 = V*(f3+f4-f5-f6)
    ic4 = V*(f3-f4+f5-f6)

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    return U0

###############################################################

def init_cond_topog(x,Nk,Neq,H0,L,A,V):
    # for a single parabolic ridge
    ic1 = H0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic2= 1./ic1 # for hu = 1:
    ic3 = np.zeros(len(x))
    
    # single hill
    bc = 0.5
    xp = 0.1
    a = 0.05*L
    B = np.maximum(0, bc*(1 - ((x - L*xp)**2)/a**2))
    B = np.maximum(0,B)

    U0 = np.zeros((Neq,Nk))
    B = 0.5*(B[0:Nk] + B[1:Nk+1]); # b
    U0[0,:] = np.maximum(0, 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) - B) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr

    return U0, B

###############################################################

def init_cond_topog4(x,Nk,Neq,H0,L,A,V):
    # for 4 parabolic ridges
    ic1 = H0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic2=1/ic1 # for hu = 1:
    ic3 = np.zeros(len(x))
    
    # 4 hills
    bc = 0.4
    xp = 0.5
    a = 0.025*L
    B = np.maximum(bc*(1 - ((x - L*0.25*xp)**2)/a**2), bc*(1 - ((x - L*0.45*xp)**2)/a**2))
    B = np.maximum(B, bc*(1 - ((x - L*0.65*xp)**2)/a**2))
    B = np.maximum(B, bc*(1 - ((x - L*0.85*xp)**2)/a**2))
    B = np.maximum(0,B)
    
    U0 = np.zeros((Neq,Nk))
    B = 0.5*(B[0:Nk] + B[1:Nk+1]); # b
    U0[0,:] = np.maximum(0, 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) - B) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    
    return U0, B

###############################################################


def init_cond_topog_cos(x,Nk,Neq,H0,L,A,V):
#    superposition of cosines
    ic1 = H0*np.ones(len(x))
    ic2=1/ic1 # for hu = 1:
    ic3 = np.zeros(len(x))

    k = 2*np.pi
    xp = 0.1
    waven = [2,4,6]
    A = [0.2, 0.1, 0.2]

    B = A[0]*(1+np.cos(k*(waven[0]*(x-xp)-0.5)))+ A[1]*(1+np.cos(k*(waven[1]*(x-xp)-0.5)))+    A[2]*(1+np.cos(k*(waven[2]*(x-xp)-0.5)))
    B = 0.5*B
    
    index = np.where(B<=np.min(B)+1e-10)
    index = index[0]
    B[:index[0]] = 0
    B[index[-1]:] = 0

    U0 = np.zeros((Neq,Nk))
    B = 0.5*(B[0:Nk] + B[1:Nk+1]); # b
    U0[0,:] = np.maximum(0, 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) - B) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    
    return U0, B

###############################################################
