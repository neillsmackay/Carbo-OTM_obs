#!/usr/bin/env python
# coding: utf-8

## Process the results of an OTM ensemble and save
# This version handles ensembles with varying weight factors for non-outcropping water masses

# In[16]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from scipy.stats import sem
from tqdm import tqdm

# Define run parameters

basedir_ensemble = '../Outputs/18012025/'
avg_periods = np.array([[1990,2000,6],[2000,2010,6]])
cstardef='nitrate'
DIC=False

datasets = ('ERA5','JRA55','ECCO')
#datasets = ('ERA5','')
fluxopts = ('JENA','SOMFFN','CMEMS','CSIR','JMA','NIES')
flux_priors = ('Cflux3a','Cflux3b','Cflux3c','Cflux3d','Cflux3e','Cflux3f')
outcrop_weights = (2,4,8)

Basins = xr.open_mfdataset('/home/users/nmackay/MTM/BSP_obs/Ehmen_NN/ERA5/nitrate_Cstar_components/BSP_ERA5_NN_*.nc').Basin


# In[18]:


# Load runs and extract OTM flux and budget terms for each ensemble member run

dC_OTM_all = np.zeros((10,avg_periods.shape[0],len(datasets),len(fluxopts),15,len(outcrop_weights)))
dC_prod_all = np.zeros((10,avg_periods.shape[0],len(datasets),len(fluxopts),15,len(outcrop_weights)))
Cint_change_all = np.zeros((10,avg_periods.shape[0],len(datasets),len(fluxopts),15,len(outcrop_weights)))
net_carbon_trans_all = np.zeros((10,avg_periods.shape[0],len(datasets),len(fluxopts),15,len(outcrop_weights)))
res_all = np.zeros((10,avg_periods.shape[0],len(datasets),len(fluxopts),15,len(outcrop_weights)))
dC_OTM_perm2_all = np.zeros((10,avg_periods.shape[0],len(datasets),len(fluxopts),15,len(outcrop_weights)))
dC_prod_perm2_all = np.zeros((10,avg_periods.shape[0],len(datasets),len(fluxopts),15,len(outcrop_weights)))


yr2sec = 365.25*24*60*60

for ds in tqdm(np.arange(len(datasets))): 
    
    for ap in tqdm(np.arange(avg_periods.shape[0])):
        
        for pf in tqdm(np.arange(len(fluxopts))):

            for member in tqdm(np.arange(15)):

                for ow in np.arange(len(outcrop_weights)):

                    fluxes = fluxopts[pf]
    
                    if DIC==True:
                        exec('runname = \'' + basedir_ensemble + datasets[ds] + '_' + str(avg_periods[ap,0]) + '_' + str(avg_periods[ap,1]) + '_' + str(avg_periods[ap,2]) + '_' + fluxes  + '_DIC_member' + str(member+1) + '.nc\'' + '')
                    else:
                        exec('runname = \'' + basedir_ensemble + datasets[ds] + '_' + str(avg_periods[ap,0]) + '_' + str(avg_periods[ap,1]) + '_' + str(avg_periods[ap,2]) + '_' + fluxes  + '_' + cstardef + '_outcropweight_' + str(outcrop_weights[ow]) + '_member' + str(member+1) + '.nc\'' + '')
    
                    if os.path.isfile(runname):
                        
                        print(runname)
    
                        BSP_basedir = str('/home/users/nmackay/MTM/BSP_obs/Ehmen_NN/' + datasets[ds] + '/' + cstardef + '_Cstar_components/')
    
    
                        BSP_files = (BSP_basedir + 'member_' + str(member+1) + '/BSP_' + datasets[ds] + '_NN*.nc')
    
                        print(BSP_files)
        
                        if datasets[ds]=='ECCO':
                            yr_init = 1992
                        elif datasets[ds]=='ERA5' or datasets[ds]=='JRA55':
                            yr_init = 1990
    
                        obs_BSP_data = xr.open_mfdataset(BSP_files)
    
                        exec('Cflux_prior = obs_BSP_data.' + flux_priors[pf] + '_sum_hist')   
    
                        d_ij = np.zeros((Basins.size,Basins.size))
    
                        d_ij[0,:] = [1, -1, 0, 0, 0, 0, 0, -1, 0]
                        d_ij[1,:] = [1, 1, -1, 0, 0, 0, 0, 0, 0]
                        d_ij[2,:] = [0, 1, 1, -1, 0, 0, 0, 0, 0]
                        d_ij[3,:] = [0, 0, 1, 1, 0, 0, 0, 0, -1]
                        d_ij[4,:] = [0, 0, 0, 0, 1, 0, 1, 0, -1]
                        d_ij[5,:] = [0, 0, 0, 0, 0, 1, 1, 0, -1]
                        d_ij[6,:] = [0, 0, 0, 0, -1, -1, 1, 1, 0]
                        d_ij[7,:] = [1, 0, 0, 0, 0, 0, -1, 1, 0]
                        d_ij[8,:] = [0, 0, 0, 1, 1, 1, 0, 0, 1]
        
                        basin_connex = np.array([[0,1],
                                                 [1,2],
                                                 [2,3],
                                                 [3,8],
                                                 [4,8],
                                                 [4,6],
                                                 [5,8],
                                                 [6,5],
                                                 [7,6],
                                                 [0,7]]).astype(int)
    
                        init_early = avg_periods[ap,0]
                        init_late = avg_periods[ap,1]
                        avg_yrs = avg_periods[ap,2]
        
                        month_init_early=(init_early-yr_init)*12
                        month_init_late=(init_late-yr_init)*12
                        Early_period = (np.array([month_init_early,month_init_early+avg_yrs*12]))
                        Late_period = (np.array([month_init_late,month_init_late+avg_yrs*12]))
    
                        dyrs = (Late_period.mean()-Early_period.mean())/12
        
                        V_early = obs_BSP_data.V_sum_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
                        V_late = obs_BSP_data.V_sum_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
                        C_early = obs_BSP_data.Cstar_mean_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
                        C_late = obs_BSP_data.Cstar_mean_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
                        A_early = obs_BSP_data.A_sum_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
                        A_late = obs_BSP_data.A_sum_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
     
                        if DIC==True:
                            C_early = obs_BSP_data.DIC_mean_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
                            C_late = obs_BSP_data.DIC_mean_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
        
                        t1=(str(yr_init)+"-01-01")
                        t2 = "2019-12-31"
    
                        ERA5_time=xr.open_mfdataset('ERA5_time.nc', decode_times=True, decode_cf=True).sel(time=slice(t1,t2)).time
                        ERA5_dt = (np.hstack((np.diff(ERA5_time.values)/10**9,31*24*3600))).astype('float64') # ERA5 time vector finishes 1st December
                        obs_dt = ERA5_dt[np.newaxis,0:obs_BSP_data.Time.size,np.newaxis]
    
                        Cint_early = ((obs_BSP_data.Cstar_mean_hist)*obs_BSP_data.V_sum_hist).isel(Time=slice(Early_period[0],Early_period[1])).mean('Time').sum('Depth') # mmol-C
                        Cint_late = ((obs_BSP_data.Cstar_mean_hist)*obs_BSP_data.V_sum_hist).isel(Time=slice(Late_period[0],Late_period[1])).mean('Time').sum('Depth') # mmol-C
    
                        Cint_early_OTM = (C_early*V_early).sum('Depth')
                        Cint_late_OTM = (C_late*V_late).sum('Depth')
    
                        Cint_change = ((Cint_late_OTM-Cint_early_OTM)/(dyrs*10**18/12)).values # Use same fields as OTM
        
                        Vol_1 = V_early.values.flatten()
                        Vol_2 = V_late.values.flatten()
                        A_2 = A_late.values.flatten()
        
                        Basin_1 = np.zeros_like(V_early)
                        Basin_2 = np.zeros_like(V_early)
                        Basin_names = []
                        for i in range(np.array(Basins).size):
                            Basin_1[i,:] = i
                            Basin_2[i,:] = i
                            for j in range(V_early.shape[-1]):
                                #... and for basin name
                                Basin_names.append(Basins[i])
    
                        Basin_1_inds = Basin_1.flatten()
                        Basin_2_inds = Basin_2.flatten()
    
                
                        Cflux_early = (Cflux_prior*obs_dt).isel(Time=slice(Early_period[0],Late_period[1])).cumsum('Time').sel(Time=slice(Early_period[0],Early_period[1])).mean('Time').sum('Depth') # mmol-C
                        Cflux_late = (Cflux_prior*obs_dt).isel(Time=slice(Early_period[0],Late_period[1])).cumsum('Time').sel(Time=slice(Late_period[0],Late_period[1])).mean('Time').sum('Depth') # mmol-C
    
                        Cflux_input = ((Cflux_late-Cflux_early)/(dyrs*(10**18/12))).values # units: PgC/yr
                        Cflux_input_perm2 = ((Cflux_late-Cflux_early)/(A_early.sum('Depth')*dyrs*(10**3))).values # Basin average flux (mol/m^2/yr)
                        Cflux_input_perm2_tot = ((Cflux_late-Cflux_early).sum('Basin')/(A_early.sum(('Depth','Basin'))*dyrs*(10**3))).values  # Global average flux (mol/m^2/yr)
                        #Cint_change = ((Cint_late-Cint_early)/(dyrs*10**18/12)).values # units: PgC/yr
    
    
                        Opt_result = xr.open_mfdataset(runname)
        
                        tree_depth=int(math.log2(Opt_result.tree_depth))
    
                        dC_Cflux = Opt_result.dC_Cflux.values
                        dC_Cflux[np.isnan(dC_Cflux)] = 0
                        dC_Cflux[~np.isfinite(dC_Cflux)] = 0
    
                        # Remove surface flux where water masses don't outcrop
    
                        Cflux_OTM = (dC_Cflux*Vol_1+Opt_result.dC_adjustment.values*Vol_2)
                        Cflux_OTM[A_2==0] = 0
                        Cflux_OTM_int = np.reshape(Cflux_OTM,(Basins.size,2**tree_depth)).sum(axis=1)/(dyrs*(10**18/12))
    
                        Cflux_OTM_perm2 = (np.reshape(Cflux_OTM,(Basins.size,2**tree_depth))).sum(axis=1)/(A_early.sum('Depth')*dyrs*(10**3)).values # Basin average flux (mol/m^2/yr)
                        Cflux_OTM_perm2_tot = (Cflux_OTM).sum()/(A_early.sum()*dyrs*10**3) # Global average flux (mol/m^2/yr)
    
                        ##
    
                        dC_Cflux_int = (np.reshape(dC_Cflux,(Basins.size,2**tree_depth))*V_early).sum(axis=1)/(dyrs*(10**18/12))
                        dC_Cflux_int_perm2 = (np.reshape(dC_Cflux,(Basins.size,2**tree_depth))*V_early).sum(axis=1)/(dyrs*(10**3))/A_early.sum('Depth').values # Basin average flux (mol/m^2/yr)
                        dC_Cflux_int_perm2_tot = (dC_Cflux*V_early.values.flatten()).sum()/(A_early.sum()*dyrs*10**3) # Global average flux (mol/m^2/yr)
    
                        dC_adj_int = ((Opt_result.dC_adjustment.values.reshape(Basins.size,2**tree_depth)*V_late).sum(axis=1)/(dyrs*(10**18/12))).values # Note Taimoor says we should be multiplying by V_late
                        dC_adj_int_perm2 = ((Opt_result.dC_adjustment.values.reshape(Basins.size,2**tree_depth)*V_late).sum(axis=1)/(dyrs*(10**3))).values/A_late.sum('Depth').values # Basin average flux (mol/m^2/yr)
                        dC_adj_int_perm2_tot = (Opt_result.dC_adjustment.values*V_late.values.flatten()).sum()/(A_late.sum()*dyrs*10**3) # Global average flux (mol/m^2/yr)
    
                        # Inter-basin transports
    
                        dt = Opt_result.dt.values
    
                        C_1 = C_early.values.flatten() + dC_Cflux    
                        C_1[np.isnan(C_1)] = 0
        
                        g_ij = Opt_result.gij.values
            
                        section_trans = np.zeros((Vol_1.size,Vol_1.size))
    
                        for i in (range(Vol_1.size)):
                            for j in range(Vol_2.size):
                                if d_ij[int(Basin_1_inds[i]), int(Basin_2_inds[j])]!=0:
                                    if Basin_names[i] != Basin_names[j]:
                                        section_trans[i,j] = g_ij[i,j]/(dt*10**6)*d_ij[int(Basin_1_inds[i]), int(Basin_2_inds[j])] # Sv
    
                        int_section_trans = np.zeros((Basins.size, Basins.size))
                        int_section_trans_C = np.zeros((Basins.size, Basins.size))
                        for i in (range(Basins.size)):
                            for j in range(Basins.size):
                                int_section_trans[i,j] = np.nansum(section_trans[V_early.shape[1]*i:(V_early.shape[1]*(i+1)),V_early.shape[1]*j:(V_early.shape[1]*(j+1))])
    
                        section_trans_temp = np.zeros((Vol_1.size, Basins.size))
                        for j in range(Basins.size):
                            section_trans_temp[:,j] = np.nansum(section_trans[:,V_early.shape[1]*j:(V_early.shape[1]*(j+1))],axis=-1)
    
                        C_joined = np.zeros((int(np.size(basin_connex)/2), int(V_early.shape[1]*2)))
                        section_joined = np.zeros((int(np.size(basin_connex)/2), int(V_early.shape[1]*2)))
    
                        for i in range(int(np.size(basin_connex)/2)):
                            C_joined[i,:] = np.concatenate([C_1[V_early.shape[1]*basin_connex[i][0]:(V_early.shape[1]*(basin_connex[i][0]+1))],C_1[V_early.shape[1]*basin_connex[i][1]:(V_early.shape[1]*(basin_connex[i][1]+1))]])
                            section_joined[i,:] = np.concatenate([section_trans_temp[V_early.shape[1]*basin_connex[i][0]:(V_early.shape[1]*(basin_connex[i][0]+1)),basin_connex[i][1]],\
                                section_trans_temp[V_early.shape[1]*basin_connex[i][1]:(V_early.shape[1]*(basin_connex[i][1]+1)),basin_connex[i][0]]])
    
                        C_trans = C_joined*section_joined*10**6*yr2sec/(10**18/12) # PgC/yr
                        CF_section_tot = np.nansum(C_trans, axis=-1)
    
                        net_carbon_trans_Basins = [CF_section_tot[0]+CF_section_tot[9],\
                        CF_section_tot[1]-CF_section_tot[0],\
                            CF_section_tot[2]-CF_section_tot[1],\
                                CF_section_tot[3]-CF_section_tot[2],\
                                    CF_section_tot[4]-CF_section_tot[5],\
                                        CF_section_tot[6]-CF_section_tot[7],\
                                            CF_section_tot[7]-CF_section_tot[8]+CF_section_tot[5],\
                                                CF_section_tot[8]-CF_section_tot[9],\
                                                -CF_section_tot[4]-CF_section_tot[6]-CF_section_tot[3]]
    
    
                        net_carbon_trans_all[:,ap,ds,pf,member,ow] = np.append(net_carbon_trans_Basins, np.nansum(net_carbon_trans_Basins)) # Net carbon transport from OTM solution
    #                    dC_OTM_all[:,ap,ds,pf,member,ow] = np.append(dC_Cflux_int + dC_adj_int, np.nansum(dC_Cflux_int + dC_adj_int)) # Carbon flux from OTM solution
    #                    dC_OTM_perm2_all[:,ap,ds,pf,member,ow] = np.append(dC_Cflux_int_perm2 + dC_adj_int_perm2, dC_Cflux_int_perm2_tot + dC_adj_int_perm2_tot) # Carbon flux density from OTM solution
                        dC_OTM_all[:,ap,ds,pf,member,ow] = np.append(Cflux_OTM_int, np.nansum(Cflux_OTM_int)) # Carbon flux from OTM solution
                        dC_OTM_perm2_all[:,ap,ds,pf,member,ow] = np.append(Cflux_OTM_perm2, Cflux_OTM_perm2_tot) # Carbon flux density from OTM solution
                        dC_prod_all[:,ap,ds,pf,member,ow] = np.append(Cflux_input, np.nansum(Cflux_input)) # pCO2 product
                        dC_prod_perm2_all[:,ap,ds,pf,member,ow] = np.append(Cflux_input_perm2, Cflux_input_perm2_tot) # pCO2 product flux density
    #                    Cint_change_all[:,ap,ds,pf,member,ow] = np.append(Cint_change, np.nansum(Cint_change)) # Inventory change from BSP files
                        Cint_change_all[:,ap,ds,pf,member,ow] = np.append(Cflux_OTM_int + net_carbon_trans_Basins, np.nansum(Cflux_OTM_int + net_carbon_trans_Basins)) # Inventory change from BSP files
                        res_all[:,ap,ds,pf,member,ow] = np.append(Cint_change - dC_Cflux_int - dC_adj_int - net_carbon_trans_Basins, np.nansum(Cint_change - dC_Cflux_int - dC_adj_int - net_carbon_trans_Basins))

dC_OTM_all[dC_OTM_all==0] = np.nan
dC_OTM_perm2_all[dC_OTM_perm2_all==0] = np.nan
dC_prod_all[dC_prod_all==0] = np.nan
dC_prod_perm2_all[dC_prod_perm2_all==0] = np.nan
net_carbon_trans_all[net_carbon_trans_all==0] = np.nan
Cint_change_all[Cint_change_all==0] = np.nan


# In[19]:


# Save ensemble results

da_dC_OTM_all = xr.DataArray(data = dC_OTM_all, dims = ["Basin","Init_early","Forcing","Prior","Member","Outcrop_weight"],
                             coords=dict(Basin = np.append(Basins, 'Global'),  Init_early = avg_periods[:,0], Forcing = np.array(datasets), Prior = np.array(fluxopts), Member = np.arange(15)+1, Outcrop_weight = np.array(outcrop_weights)))
da_dC_OTM_perm2_all = xr.DataArray(data = dC_OTM_perm2_all, dims = ["Basin","Init_early","Forcing","Prior","Member","Outcrop_weight"],
                             coords=dict(Basin = np.append(Basins, 'Global'),  Init_early = avg_periods[:,0], Forcing = np.array(datasets), Prior = np.array(fluxopts), Member = np.arange(15)+1, Outcrop_weight = np.array(outcrop_weights)))
da_net_carbon_trans_all = xr.DataArray(data = net_carbon_trans_all, dims = ["Basin","Init_early","Forcing","Prior","Member","Outcrop_weight"],
                             coords=dict(Basin = np.append(Basins, 'Global'),  Init_early = avg_periods[:,0], Forcing = np.array(datasets), Prior = np.array(fluxopts), Member = np.arange(15)+1, Outcrop_weight = np.array(outcrop_weights)))
da_dC_prod_all = xr.DataArray(data = dC_prod_all, dims = ["Basin","Init_early","Forcing","Prior","Member","Outcrop_weight"],
                             coords=dict(Basin = np.append(Basins, 'Global'),  Init_early = avg_periods[:,0], Forcing = np.array(datasets), Prior = np.array(fluxopts), Member = np.arange(15)+1, Outcrop_weight = np.array(outcrop_weights)))
da_dC_prod_perm2_all = xr.DataArray(data = dC_prod_perm2_all, dims = ["Basin","Init_early","Forcing","Prior","Member","Outcrop_weight"],
                             coords=dict(Basin = np.append(Basins, 'Global'),  Init_early = avg_periods[:,0], Forcing = np.array(datasets), Prior = np.array(fluxopts), Member = np.arange(15)+1, Outcrop_weight = np.array(outcrop_weights)))
da_Cint_change_all = xr.DataArray(data = Cint_change_all, dims = ["Basin","Init_early","Forcing","Prior","Member","Outcrop_weight"],
                             coords=dict(Basin = np.append(Basins, 'Global'),  Init_early = avg_periods[:,0], Forcing = np.array(datasets), Prior = np.array(fluxopts), Member = np.arange(15)+1, Outcrop_weight = np.array(outcrop_weights)))
da_res_all = xr.DataArray(data = res_all, dims = ["Basin","Init_early","Forcing","Prior","Member","Outcrop_weight"],
                             coords=dict(Basin = np.append(Basins, 'Global'),  Init_early = avg_periods[:,0], Forcing = np.array(datasets), Prior = np.array(fluxopts), Member = np.arange(15)+1, Outcrop_weight = np.array(outcrop_weights)))

da_ensemble = xr.Dataset()

da_ensemble['OTM_flux'] = da_dC_OTM_all
da_ensemble['OTM_flux_dens'] = da_dC_OTM_perm2_all
da_ensemble['net_carbon_trans'] = da_net_carbon_trans_all
da_ensemble['prior_flux'] = da_dC_prod_all
da_ensemble['prior_flux_dens'] = da_dC_prod_perm2_all
da_ensemble['Cint_change'] = da_Cint_change_all
da_ensemble['residual'] = da_res_all

if DIC==True:
    da_ensemble.to_netcdf(basedir_ensemble + 'ensemble_budget_DIC.nc')
else:
    da_ensemble.to_netcdf(basedir_ensemble + 'ensemble_budget_' + cstardef + '.nc')

