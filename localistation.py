#######################################################################
# Investigating effect of localisation lengthscale
# (T. Kent: amttk@leeds.ac.uk)
#######################################################################
'''
5.2.16
'''
import math as m
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import os
import errno

##################################################################
# GASPARI-COHN TAPER FUNCTION FOR COV LOCALISATION
################################################################
def gaspcohn(r):
    # Gaspari-Cohn taper function.
    # very close to exp(-(r/c)**2), where c = sqrt(0.15)
    # r should be >0 and normalized so taper = 0 at r = 1
    rr = 2.0*r
    rr += 1.e-13 # avoid divide by zero warnings from numpy
    taper = np.where(r<=0.5, \
                     ( ( ( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2 + 1.0,\
                     np.zeros(r.shape,r.dtype))

    taper = np.where(np.logical_and(r>0.5,r<1.), \
                    ( ( ( ( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
                    + 4.0 - 2.0 / (3.0 * rr), taper)
    return taper

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

loc = [4.,2.5,1.,1e-10]

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

# masks for locating obs locations
row_vec = range(obs_dens,n_d+1,obs_dens)
obs_mask = np.array(row_vec[0:n_obs/Neq])-1
h_obs_mask = range(0,n_obs/Neq)
hu_obs_mask = range(n_obs/Neq,2*n_obs/Neq)
hr_obs_mask = range(2*n_obs/Neq,3*n_obs/Neq)

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
for T in time_vec[1:]:
    Xbar[:,:,T] = np.dot(X[:,:,T],ONE) # fc mean
    Xdev[:,:,T] = X[:,:,T] - Xbar[:,:,T] # fc deviations from mean
    Xdev_tr[:,:,T] = X[:,:,T] - X_tr[:,:,T] # fc deviations from truth
    Xanbar[:,:,T] = np.dot(Xan[:,:,T],ONE) # an mean
    Xandev[:,:,T] = Xan[:,:,T] - Xanbar[:,:,T] # an deviations from mean
    Xandev_tr[:,:,T] = Xan[:,:,T] - X_tr[:,:,T] # an deviations from truth

T = time_vec[36]


# covariance matrix
Pf = np.dot(Xdev[:,:,T], Xdev[:,:,T].T)
Pf = Pf/(n_ens - 1)
print 'max Pf value: ', np.max(Pf)
print 'min Pf value: ', np.min(Pf)

# correlation matrix
Cf = np.corrcoef(Xdev[:,:,T])

#taper functions
taper = np.zeros([Nk_fc,len(loc)])
for ii in range(0,len(loc)):
    loc_rho = loc[ii] # loc_rho is form of lengthscale.
    rr = np.arange(0,loc_rho,loc_rho/Nk_fc)
    taper[:,ii] = gaspcohn(rr)

## print ' *** PLOT: localisation taper function ***'
fig, axes = plt.subplots(1, 1, figsize=(5,5))
#axes.plot(range(0,Nk_fc), taper[:,0],label='loc = %s' %loc[0])
#axes.plot(range(0,Nk_fc), taper[:,1],label='loc = %s' %loc[1])
#axes.plot(range(0,Nk_fc), taper[:,2],label='loc = %s' %loc[2])
#axes.plot(range(0,Nk_fc), taper[:,3],label='loc = %s' %loc[3])
axes.plot(range(0,Nk_fc), taper[:,0],label='$L_{loc} = 50$')
axes.plot(range(0,Nk_fc), taper[:,1],label='$L_{loc} = 80$')
axes.plot(range(0,Nk_fc), taper[:,2],label='$L_{loc} = 200$')
axes.plot(range(0,Nk_fc), taper[:,3],label='$L_{loc} = \infty$')
axes.set_ylim(0,1.02)
axes.set_xlabel('x')
#axes.set_title('Taper functions')
axes.legend(loc = 7)
axes.set_aspect(1./axes.get_data_ratio())

name_f = "/loc.pdf"
f_name_f = str(figsdir+name_f)
plt.savefig(f_name_f)
print ' *** %s at time level %d saved to %s' %(name_f,T,figsdir)


vec = taper[:,j]
rho = np.zeros((Nk_fc,Nk_fc))
for ii in range(Nk_fc):
    for jj in range(Nk_fc):
        rho[ii,jj] = vec[np.min([abs(ii-jj),abs(ii+Nk_fc-jj),abs(ii-Nk_fc-jj)])]
rho = np.tile(rho, (Neq,Neq))
print 'loc matrix rho shape: ', np.shape(rho)

Pf_loc = rho*Pf
Cf_loc = rho*Cf


print ' *** PLOT: forecast correlation matrix ***'
cpar = 0.3
tick_loc = [0.5*Nk_fc,1.5*Nk_fc,2.5*Nk_fc]
tick_lab = [r'$h$',r'$hu$',r'$hr$']
fig, axes = plt.subplots(2, 2, figsize=(10,10))

axes[0,0].plot(range(0,Nk_fc), taper[:,0],'k-',label='loc = %s' %loc[0])
axes[0,0].plot(range(0,Nk_fc), taper[:,1],'k--',label='loc = %s' %loc[1])
axes[0,0].plot(range(0,Nk_fc), taper[:,2],'k:',label='loc = %s' %loc[2])
axes[0,0].plot(range(0,Nk_fc), taper[:,3],'r',label='loc = %s' %loc[3])
axes[0,0].set_xlabel('Gridpoints')
axes[0,0].set_title('Taper functions')
axes[0,0].legend(loc = 1)
axes[0,0].set_aspect(1./axes[0,0].get_data_ratio())

im=axes[1,0].imshow(Pf,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
axes[1,0].set_xticks(tick_loc)
axes[1,0].set_yticks(tick_loc)
axes[1,0].set_xticklabels(tick_lab,fontsize=14)
axes[1,0].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(axes[1,0].get_xticklines(),visible=False) # invisible ticks
#plt.setp(axes[1,0].get_yticklines(),visible=False)
axes[1,0].set_title(r'$C^f_e$')

im=axes[0,1].imshow(rho,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
axes[0,1].set_xticks(tick_loc)
axes[0,1].set_yticks(tick_loc)
axes[0,1].set_xticklabels(tick_lab,fontsize=14)
axes[0,1].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(fig.axes[0,1].get_xticklines(),visible=False) # invisible ticks
#plt.setp(fig.axes[0,1].get_yticklines(),visible=False)
axes[0,1].set_title(r'$\rho_{loc}$ with $loc = %s$' %loc[j])

im=axes[1,1].imshow(Pf_loc,cmap=plt.cm.RdBu,vmin=-cpar, vmax=cpar)
#fig.title('Ensemble forecast error correlation matrix, $C^f_e$', fontsize = 20)
axes[1,1].set_xticks(tick_loc)
axes[1,1].set_yticks(tick_loc)
axes[1,1].set_xticklabels(tick_lab,fontsize=14)
axes[1,1].set_yticklabels(tick_lab,fontsize=14)
#plt.setp(fig.axes[1,1].get_xticklines(),visible=False) # invisible ticks
#plt.setp(fig.axes[1,1].get_yticklines(),visible=False)
axes[1,1].set_title(r'$\rho_{loc} \circ C^f_e$')

#plt.axis('equal')
im.set_clim(-cpar,cpar)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()

#name_f = "/T%d_loc.pdf" %T
#f_name_f = str(figsdir+name_f)
#plt.savefig(f_name_f)
#print ' *** %s at time level %d saved to %s' %(name_f,T,figsdir)
