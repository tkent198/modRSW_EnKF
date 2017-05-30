##################################################################
# Summary diagnostics of idealised enkf experiments
# inc. summary plots a la Poterjoy and Zhang
##################################################################

'''
    Each directory has i*j*k experiments with different parameter combinations. This script produces summary plots for comparison. 
    
    Specify: dirname
    
    If DIAGS.npy exists, straight to plotting. If not, calculate statistics and save before plotting.
    '''

## generic modules
import os
import errno
import numpy as np
import matplotlib.pyplot as plt

## custom modules
from parameters import *
from analysis_diag_stats import ave_stats
##################################################################

dirname = '/test_enkf'

# LOAD DATA FROM GIVEN DIRECTORY
cwd = os.getcwd()
dirn = str(cwd+dirname)
figsdir = str(dirn+'/figs')

#check if dir exixts, if not make it
try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

#TEST CASE: parameters for outer loop
loc = [1e-10]
add_inf = [0.2]
inf = [1.01, 1.05, 1.1]

loc = ['inf']
ii=0
##################################################################

# if LOAD:
try:
    DIAGS = np.load(str(dirn+'/DIAGS.npy'))
    DIAGS = np.roll(DIAGS,1,2)
    spr_fc = DIAGS[0,:,:,:]
    spr_an = DIAGS[1,:,:,:]
    err_fc = DIAGS[2,:,:,:]
    err_an = DIAGS[3,:,:,:]
    rmse_fc = DIAGS[4,:,:,:]
    rmse_an = DIAGS[5,:,:,:]
    crps_fc = DIAGS[6,:,:,:]
    crps_an = DIAGS[7,:,:,:]
    OI = DIAGS[8,:,:,:]

##################################################################

# if NOT:
except:
    spr_fc = np.empty([len(loc),len(add_inf),len(inf)])
    spr_an = np.empty([len(loc),len(add_inf),len(inf)])
    err_fc = np.empty([len(loc),len(add_inf),len(inf)])
    err_an = np.empty([len(loc),len(add_inf),len(inf)])
    rmse_fc = np.empty([len(loc),len(add_inf),len(inf)])
    rmse_an = np.empty([len(loc),len(add_inf),len(inf)])
    crps_fc = np.empty([len(loc),len(add_inf),len(inf)])
    crps_an = np.empty([len(loc),len(add_inf),len(inf)])
    OI = np.empty([len(loc),len(add_inf),len(inf)])

    for i in range(0,len(loc)):
        for j in range(0,len(add_inf)):
            for k in range(0,len(inf)):
                spr_fc[i,j,k], err_fc[i,j,k], rmse_fc[i,j,k], crps_fc[i,j,k], spr_an[i,j,k], err_an[i,j,k], rmse_an[i,j,k], crps_an[i,j,k], OI[i,j,k] = ave_stats(i, j, k, dirname)

    DIAGS = np.empty([9,len(loc),len(add_inf),len(inf)])
    DIAGS[0,:,:,:] = spr_fc
    DIAGS[1,:,:,:] = spr_an
    DIAGS[2,:,:,:] = err_fc
    DIAGS[3,:,:,:] = err_an
    DIAGS[4,:,:,:] = rmse_fc
    DIAGS[5,:,:,:] = rmse_an
    DIAGS[6,:,:,:] = crps_fc
    DIAGS[7,:,:,:] = crps_an
    DIAGS[8,:,:,:] = OI

    np.save(str(dirn+'/DIAGS'),DIAGS)
    print ' '
    print ' *** Summary diagnostics saved in :', dirn
    print ' '

    DIAGS = np.roll(DIAGS,1,2)
    spr_fc = DIAGS[0,:,:,:]
    spr_an = DIAGS[1,:,:,:]
    err_fc = DIAGS[2,:,:,:]
    err_an = DIAGS[3,:,:,:]
    rmse_fc = DIAGS[4,:,:,:]
    rmse_an = DIAGS[5,:,:,:]
    crps_fc = DIAGS[6,:,:,:]
    crps_an = DIAGS[7,:,:,:]
    OI = DIAGS[8,:,:,:]


##################################################################
fs = 14
cpar = 0.06
tick_loc_y = [0]
tick_loc_x = [0,1,2]
tick_lab_y = np.roll(add_inf,1)
tick_lab_x = inf



##################################################################
print ' *** PLOT: STATS matrix with AME ***'
##################################################################

fig, axes = plt.subplots(2, 2, figsize=(10,10))

im=axes[0,0].matshow(err_fc[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = err_fc[ii,y_val,x_val]
    if c == np.nanmin(err_an[ii,:,:]):
        axes[0,0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[0,0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
axes[0,0].set_xticks(tick_loc_x)
axes[0,0].set_yticks(tick_loc_y)
axes[0,0].set_xticklabels(tick_lab_x,fontsize=14)
axes[0,0].set_yticklabels(tick_lab_y,fontsize=14)
axes[0,0].set_title('err_fc')
#axes[0,0].grid()

im=axes[1,0].matshow(spr_fc[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = spr_fc[ii,y_val,x_val]
    if c == np.nanmin(spr_fc[ii,:,:]):
        axes[1,0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[1,0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)

axes[1,0].set_xticks(tick_loc_x)
axes[1,0].set_yticks(tick_loc_y)
axes[1,0].set_xticklabels(tick_lab_x,fontsize=14)
axes[1,0].set_yticklabels(tick_lab_y,fontsize=14)
axes[1,0].set_title('spr_fc')
#axes[1,0].grid()

im=axes[0,1].matshow(err_an[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = err_an[ii,y_val,x_val]
    if c == np.nanmin(err_an[ii,:,:]):
        axes[0,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[0,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
axes[0,1].set_xticks(tick_loc_x)
axes[0,1].set_yticks(tick_loc_y)
axes[0,1].set_xticklabels(tick_lab_x,fontsize=14)
axes[0,1].set_yticklabels(tick_lab_y,fontsize=14)
axes[0,1].set_title('err_an')
#axes[0,1].grid()

im=axes[1,1].matshow(spr_an[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = spr_an[ii,y_val,x_val]
    if c == np.nanmin(spr_an[ii,:,:]):
        axes[1,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[1,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
axes[1,1].set_xticks(tick_loc_x)
axes[1,1].set_yticks(tick_loc_y)
axes[1,1].set_xticklabels(tick_lab_x,fontsize=14)
axes[1,1].set_yticklabels(tick_lab_y,fontsize=14)
axes[1,1].set_title('spr_an')
#axes[1,1].grid()

im.set_clim(0,cpar)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
##################################################################
name = "/spr_mae_summary%d.pdf" %(ii+1)
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '
##################################################################
print ' *** PLOT: STATS matrix with RMSE ***'
##################################################################

cpar = np.nanmax(np.maximum(rmse_fc[ii,:,:],spr_fc[ii,:,:]))
cpar = np.round(cpar+0.025,2)
print cpar

fig, axes = plt.subplots(2, 2, figsize=(12,10))

im=axes[0,0].matshow(rmse_fc[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c =rmse_fc[ii,y_val,x_val]
    if c == np.nanmin(rmse_fc[ii,:,:]):
        axes[0,0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[0,0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
axes[0,0].set_xticks(tick_loc_x)
axes[0,0].set_yticks(tick_loc_y)
axes[0,0].set_xticklabels(tick_lab_x,fontsize=14)
axes[0,0].set_yticklabels(tick_lab_y,fontsize=14)
axes[0,0].set_title('rmse_fc')
#axes[0,0].grid()

im=axes[1,0].matshow(spr_fc[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = spr_fc[ii,y_val,x_val]
    if c == np.nanmin(spr_fc[ii,:,:]):
        axes[1,0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[1,0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
axes[1,0].set_xticks(tick_loc_x)
axes[1,0].set_yticks(tick_loc_y)
axes[1,0].set_xticklabels(tick_lab_x,fontsize=14)
axes[1,0].set_yticklabels(tick_lab_y,fontsize=14)
axes[1,0].set_title('spr_fc')
#axes[1,0].grid()

im=axes[0,1].matshow(rmse_an[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = rmse_an[ii,y_val,x_val]
    if c == np.nanmin(rmse_an[ii,:,:]):
        axes[0,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[0,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)

axes[0,1].set_xticks(tick_loc_x)
axes[0,1].set_yticks(tick_loc_y)
axes[0,1].set_xticklabels(tick_lab_x,fontsize=14)
axes[0,1].set_yticklabels(tick_lab_y,fontsize=14)
axes[0,1].set_title('rmse_an')
#axes[0,1].grid()

im=axes[1,1].matshow(spr_an[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = spr_an[ii,y_val,x_val]
    if c == np.nanmin(spr_an[ii,:,:]):
        axes[1,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[1,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
axes[1,1].set_xticks(tick_loc_x)
axes[1,1].set_yticks(tick_loc_y)
axes[1,1].set_xticklabels(tick_lab_x,fontsize=14)
axes[1,1].set_yticklabels(tick_lab_y,fontsize=14)
axes[1,1].set_title('spr_an')
#axes[1,1].grid()

im.set_clim(0,cpar)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

##################################################################
name = "/spr_rmse_summary%d.pdf" %(ii+1)
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '
##################################################################
print ' *** PLOT: STATS matrix with CRPS ***'
##################################################################

cpar = np.nanmax(np.maximum(crps_fc[ii,:,:],crps_fc[ii,:,:]))
cpar = np.round(cpar+0.005,2)
print cpar

fig, axes = plt.subplots(1, 2, figsize=(12,7))

im=axes[0].matshow(crps_fc[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c =crps_fc[ii,y_val,x_val]
    if c == np.nanmin(crps_fc[ii,:,:]):
        axes[0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[0].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
axes[0].set_xticks(tick_loc_x)
axes[0].set_yticks(tick_loc_y)
axes[0].set_xticklabels(tick_lab_x,fontsize=14)
axes[0].set_yticklabels(tick_lab_y,fontsize=14)
axes[0].set_title('crps_fc')
#axes[0,0].grid()

im=axes[1].matshow(crps_an[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = crps_an[ii,y_val,x_val]
    if c == np.nanmin(crps_an[ii,:,:]):
        axes[1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes[1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
axes[1].set_xticks(tick_loc_x)
axes[1].set_yticks(tick_loc_y)
axes[1].set_xticklabels(tick_lab_x,fontsize=14)
axes[1].set_yticklabels(tick_lab_y,fontsize=14)
axes[1].set_title('crps_an')
#axes[1,0].grid()

#im=axes[0,1].matshow(rmse_an[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
#y, x = np.meshgrid(tick_loc_y,tick_loc_x)
#for x_val, y_val in zip(x.flatten(), y.flatten()):
#    c = rmse_an[ii,y_val,x_val]
#    if c == np.min(rmse_an[ii,:,:]):
#        axes[0,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
#    else:
#        axes[0,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
#axes[0,1].set_xticks(tick_loc_x)
#axes[0,1].set_yticks(tick_loc_y)
#axes[0,1].set_xticklabels(tick_lab_x,fontsize=14)
#axes[0,1].set_yticklabels(tick_lab_y,fontsize=14)
#axes[0,1].set_title('rmse_an')
##axes[0,1].grid()
#
#im=axes[1,1].matshow(spr_an[ii,:,:],cmap='hot_r',vmin=0,vmax=cpar)
#y, x = np.meshgrid(tick_loc_y,tick_loc_x)
#for x_val, y_val in zip(x.flatten(), y.flatten()):
#    c = spr_an[ii,y_val,x_val]
#    if c == np.min(spr_an[ii,:,:]):
#        axes[1,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs,weight='bold')
#    else:
#        axes[1,1].text(x_val, y_val, '%.3g'%c, va='center', ha='center',fontsize=fs)
#axes[1,1].set_xticks(tick_loc_x)
#axes[1,1].set_yticks(tick_loc_y)
#axes[1,1].set_xticklabels(tick_lab_x,fontsize=14)
#axes[1,1].set_yticklabels(tick_lab_y,fontsize=14)
#axes[1,1].set_title('spr_an')
##axes[1,1].grid()

im.set_clim(0,cpar)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

##################################################################
name = "/crps_summary%d.pdf" %(ii+1)
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '


##################################################################
print ' *** PLOT: OI matrix ***'
##################################################################
cpar = np.nanmax(OI[ii,:,:])
cpar = np.round(cpar+5,-1)
print cpar
fig = plt.figure(figsize=(7,7))
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
im = axes.matshow(OI[ii,:,:],cmap='hot_r',vmin=0, vmax=cpar)
y, x = np.meshgrid(tick_loc_y,tick_loc_x)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = OI[ii,y_val,x_val]
    if c == np.nanmax(OI[ii,:,:]):
        axes.text(x_val, y_val, '%.1f'%c, va='center', ha='center',fontsize=fs,weight='bold')
    else:
        axes.text(x_val, y_val, '%.1f'%c, va='center', ha='center',fontsize=fs)

axes.set_xticks(tick_loc_x)
axes.set_yticks(tick_loc_y)
axes.set_xticklabels(tick_lab_x,fontsize=14)
axes.set_yticklabels(tick_lab_y,fontsize=14)
fig.colorbar(im)
##################################################################
name = "/OI_summary%d.pdf" %(ii+1)
f_name = str(figsdir+name)
plt.savefig(f_name)
print ' '
print ' *** %s saved to %s' %(name,figsdir)
print ' '
##################################################################

#plt.show()

'''

fig, ax = plt.subplots()

min_val, max_val, diff = 0., 5., 1.

#imshow portion
N_points = (max_val - min_val) / diff
print 'N_points =', N_points
imshow_data = np.random.rand(N_points, N_points)
ax.imshow(imshow_data, interpolation='nearest')

#text portion
ind_array = np.arange(min_val, max_val, diff)
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = imshow_data[x_val,y_val]
    ax.text(x_val, y_val, c, va='center', ha='center')

#set tick marks for grid
ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(min_val-diff/2, max_val-diff/2)
ax.set_ylim(min_val-diff/2, max_val-diff/2)
ax.grid()
plt.show()
'''