#!/usr/bin/env python
# coding: utf-8

# ## Remapping an ensemble of OTM solutions in Eulerian coordinates in 3D

# In[1]:


# Import the OTM function from the WM_Methods package
#from WM_Methods import OTM, Remapping
## Module to track runtime of cells and loops
#import time
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#import pandas as pd
import xarray as xr
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import dask.array as da
import os

## Load a text file that includes the list of strings pointing to the relevant data. 
## This ensures that pull requests don't continuously overwrite hardcoded file paths.

filename = '/home/users/nmackay/MTM/WM_Methods/folders_obs.txt'
with open(filename) as f:
    mylist = f.read().splitlines()
    


# In[2]:


# Define run parameters

basedir_ensemble = '../Outputs/18012025/'
DIC=False

avg_period = [2000,2010,6]

dataset = 'ECCO'
cstardef = 'nitrate'
tree_depth=6 # 64 water masses per basin

fluxopts = ('JENA','SOMFFN','CMEMS','CSIR','JMA','NIES')
flux_priors = ('Cflux3a','Cflux3b','Cflux3c','Cflux3d','Cflux3e','Cflux3f')
outcrop_weights = (2,4,8)

Basins = xr.open_mfdataset('/home/users/nmackay/MTM/BSP_obs/Ehmen_NN/ERA5/nitrate_Cstar_components/BSP_ERA5_NN_*.nc').Basin


# In[3]:


# Load runs and extract OTM terms for each ensemble member run

dC_adj_all = np.zeros((9,64,len(fluxopts),15,len(outcrop_weights)))
dC_mix_all = np.zeros((9,64,len(fluxopts),15,len(outcrop_weights)))
dC_prior_all = np.zeros((9,64,len(fluxopts),15,len(outcrop_weights)))
dC_change_all = np.zeros((9,64,len(fluxopts),15,len(outcrop_weights)))
dC_OTM_all = np.zeros((9,64,len(fluxopts),15,len(outcrop_weights)))

for pf in tqdm(np.arange(len(fluxopts))):

    for member in tqdm(np.arange(15)):

        for ow in np.arange(len(outcrop_weights)):       

            fluxes = fluxopts[pf]
    
            exec('runname = \'' + basedir_ensemble + dataset + '_' + str(avg_period[0]) + '_' + str(avg_period[1]) + '_' + str(avg_period[2]) + '_' + fluxes + '_' + cstardef + '_outcropweight_' + str(outcrop_weights[ow]) + '_member' + str(member+1) + '.nc\'' + '')
            if DIC:
                exec('runname = \'' + basedir_ensemble + dataset + '_' + str(avg_period[0]) + '_' + str(avg_period[1]) + '_' + str(avg_period[2]) + '_' + fluxes + '_DIC_member' + str(member+1) + '.nc\'' + '')
              
    
            if os.path.isfile(runname):
    
                print(runname)
    
                BSP_basedir = str('/home/users/nmackay/MTM/BSP_obs/Ehmen_NN/' + dataset + '/' + cstardef +'_Cstar_components/')
                BSP_files = (BSP_basedir + 'member_' + str(member+1) + '/BSP_' + dataset + '_NN*.nc')
    
                print(BSP_files)
    
                if dataset=='ECCO':
                    yr_init = 1992
                elif dataset == 'ERA5' or dataset == 'JRA55':
                    yr_init = 1990
    
                obs_BSP_data = xr.open_mfdataset(BSP_files)
                OTM_result = xr.open_mfdataset(runname)
    
                month_init_early=(OTM_result.init_early-yr_init)*12
                month_init_late=(OTM_result.init_late-yr_init)*12
                Early_period = (np.array([month_init_early,month_init_early+OTM_result.dyrs*12]))
                Late_period = (np.array([month_init_late,month_init_late+OTM_result.dyrs*12]))
    
                C_early = obs_BSP_data.Cstar_mean_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
                C_late = obs_BSP_data.Cstar_mean_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
                if DIC:
                    C_early = obs_BSP_data.DIC_mean_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
                    C_late = obs_BSP_data.DIC_mean_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
                    
                V_early = obs_BSP_data.V_sum_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
                V_late = obs_BSP_data.V_sum_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
                A_late = obs_BSP_data.A_sum_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
    
                dC_adj=OTM_result.dC_adjustment.values.reshape(Basins.size,2**tree_depth) # mmol-C/m^3
                dC_prior=OTM_result.dC_Cflux.values.reshape(Basins.size,2**tree_depth) # mmol-C/m^3
                dC_change = C_late.values - C_early.values # mmol-C/m^3
    
                dC_adj[~np.isfinite(dC_change)]=0
    #            dC_adj[~np.isfinite(dC_adj)]=0
                dC_prior[~np.isfinite(dC_prior)]=0
                dC_change[~np.isfinite(dC_change)]=0
    
                # Create an adjusted OTM flux for outcropping water masses only
    
                dC_OTM = dC_prior+dC_adj
                dC_OTM[A_late==0]=0
                
                ##
                
                C0 = np.nanmedian(obs_BSP_data.Cstar_mean_hist.isel(Time=slice(Early_period[0],Late_period[1])).values.flatten())
    
                X = C_early.values.flatten()-C0+dC_prior.flatten()
                X[np.isnan(X)]=0
                dC_mix=np.matmul(X,OTM_result.gij.values)/V_late.values.flatten() - (C_early.values.flatten()-C0+dC_prior.flatten())
    
                dC_mix=dC_mix.reshape(Basins.size,2**tree_depth)
                dC_mix[~np.isfinite(dC_mix)]=0
                
    
                dC_adj_all[:,:,pf,member,ow] = dC_adj
                dC_prior_all[:,:,pf,member,ow] = dC_prior
                dC_change_all[:,:,pf,member,ow] = dC_change
                dC_mix_all[:,:,pf,member,ow] = dC_mix
                dC_OTM_all[:,:,pf,member,ow] = dC_OTM 


# In[4]:


## Calculate OTM components ensemble mean

dC_adj = np.nanmean(dC_adj_all,axis=(2,3,4))
dC_prior = np.nanmean(dC_prior_all,axis=(2,3,4))
dC_change = np.nanmean(dC_change_all,axis=(2,3,4))
dC_mix = np.nanmean(dC_mix_all,axis=(2,3,4))
dC_OTM = np.nanmean(dC_OTM_all,axis=(2,3,4))


# In[5]:


plt.plot((dC_prior.flatten()+dC_mix.flatten()+dC_adj.flatten()))
plt.plot(dC_change.flatten())


# In[6]:


## Load remapping mask

chunks = {'Time': 1, 'k': 2, 'tree_depth':1}
chunks2 = {'k': 5, 'Basins': 1}

if dataset=='ECCO':
    Remapping_mask=xr.open_mfdataset('/home/users/nmackay/unicorns/MTM/Remapping_mask/ECCO1deg/Remapping_mask_ECCO1deg_1992_2018.nc', chunks = chunks) # ECCO
elif dataset=='ERA5' or dataset=='JRA55':
    Remapping_mask=xr.open_mfdataset('/home/users/nmackay/unicorns/MTM/Remapping_mask/EN4/Remapping_mask_EN4_1990_2019.nc', chunks = chunks) # EN4

# Load basin mask

EN4_mask = xr.open_mfdataset(mylist[5] ,decode_times=True, chunks = chunks2).astype('float32')


# In[7]:


## Allocate values to basins using basin and Eulerian masks

Basin_mask=EN4_mask.mask_EN4

dC_adj_tot = 0
dC_mix_tot = 0
dC_prior_tot = 0
dC_change_tot = 0
dC_OTM_tot = 0

for i in tqdm(range(2**tree_depth)):
    tmp_dCadj = 0
    tmp_dCmix = 0
    tmp_prior = 0
    tmp_dCflux = 0
    tmp_change = 0
    tmp_OTM = 0
    
    for j in range(Basins.size):
        
        tmp = dC_adj[j,i]*Basin_mask[j,:,:,:].values * Remapping_mask.Eulerian_mask[i,:,:,:].values
        tmp=np.nan_to_num(tmp,nan=0)
        tmp_dCadj = tmp+tmp_dCadj
        
        tmp = dC_mix[j,i]*Basin_mask[j,:,:,:].values * Remapping_mask.Eulerian_mask[i,:,:,:].values
        tmp=np.nan_to_num(tmp,nan=0)
        tmp_dCmix = tmp+tmp_dCmix
        
        tmp = dC_prior[j,i]*Basin_mask[j,:,:,:].values * Remapping_mask.Eulerian_mask[i,:,:,:].values
        tmp=np.nan_to_num(tmp,nan=0)
        tmp_prior = tmp+tmp_prior
               
        tmp = dC_change[j,i]*Basin_mask[j,:,:,:].values * Remapping_mask.Eulerian_mask[i,:,:,:].values
        tmp=np.nan_to_num(tmp,nan=0)
        tmp_change = tmp+tmp_change

        tmp = dC_OTM[j,i]*Basin_mask[j,:,:,:].values * Remapping_mask.Eulerian_mask[i,:,:,:].values
        tmp=np.nan_to_num(tmp,nan=0)
        tmp_OTM = tmp+tmp_OTM

                                                             
        
    dC_adj_tot = tmp_dCadj+dC_adj_tot
    dC_mix_tot = tmp_dCmix+dC_mix_tot
    dC_prior_tot = tmp_prior+dC_prior_tot
    dC_change_tot = tmp_change+dC_change_tot
    dC_OTM_tot = tmp_OTM+dC_OTM_tot

# In[8]:


## Save remapped outputs to netcdf

if DIC:
    OTMrun = dataset + '_' + str(avg_period[0]) + '_' + str(avg_period[1]) + '_' + str(avg_period[2]) + '_DIC'
else:
    OTMrun = dataset + '_' + str(avg_period[0]) + '_' + str(avg_period[1]) + '_' + str(avg_period[2]) + '_' + cstardef

EN4_data = xr.open_mfdataset('/home/users/nmackay/unicorns/machine_learning/EN4_data/analysis/EN.4.2.2.analyses.g10.1990/EN.4.2.2.f.analysis.g10.1990??.nc', decode_times=True, decode_cf=True).isel(time=0)
land_mask = xr.open_mfdataset('land_mask_EN4.nc')
area = xr.open_mfdataset(mylist[7]).rename_dims({'latitude':'lat','longitude':'lon'}).areas*land_mask.isel(depth=0)
EN4_dz = (EN4_data.depth_bnds[:,1]-EN4_data.depth_bnds[:,0])

da_dC_adj_tot = xr.DataArray(data = dC_adj_tot.astype('float32'), dims = ["depth","lat","lon"])
da_dC_mix_tot = xr.DataArray(data = dC_mix_tot.astype('float32'), dims = ["depth","lat","lon"])
da_dC_prior_tot = xr.DataArray(data = dC_prior_tot.astype('float32'), dims = ["depth","lat","lon"])
da_dC_change_tot = xr.DataArray(data = dC_change_tot.astype('float32'), dims = ["depth","lat","lon"])
da_dC_OTM_tot = xr.DataArray(data = dC_OTM_tot.astype('float32'), dims = ["depth","lat","lon"])

coords=Remapping_mask.Eulerian_mask[0,:,:,:].reset_coords(('tree_depth','Basin'))

obs_OTM_remapped=xr.Dataset(coords=coords.coords).astype('float32')
obs_OTM_remapped = obs_OTM_remapped.assign_coords(area = area.reset_coords(names=("depth","time")).__xarray_dataarray_variable__)
obs_OTM_remapped = obs_OTM_remapped.assign_coords(dz = EN4_dz.reset_coords(names="time").depth_bnds)

obs_OTM_remapped['dC_adj_remapped'] = da_dC_adj_tot
obs_OTM_remapped['dC_mix_remapped'] = da_dC_mix_tot
obs_OTM_remapped['dC_prior_remapped'] = da_dC_prior_tot
obs_OTM_remapped['dC_change_remapped'] = da_dC_change_tot
obs_OTM_remapped['dC_OTM_remapped'] = da_dC_OTM_tot
obs_OTM_remapped['init_early'] = OTM_result.init_early
obs_OTM_remapped['init_late'] = OTM_result.init_late
obs_OTM_remapped['dyrs'] = OTM_result.dyrs

obs_OTM_remapped.to_netcdf('/home/users/nmackay/MTM/Outputs/' + basedir_ensemble[11:19] + '/' + OTMrun + '_ensemble_remapped.nc')


# In[ ]:




