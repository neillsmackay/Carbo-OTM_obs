#!/usr/bin/env python
# coding: utf-8

# # Water-mass Methods Package applied to observations
# 
# ### This version minimises dT_adj, dS_adj and dC_adj simultaneously

# In[31]:


# Import the MTM function from the WM_Methods package
from WM_Methods import MTM_3
## Module to track runtime of cells and loops
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import xarray as xr
from tqdm import tqdm
import math
## Suppress warnings related to division by zero
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os

def OTM_obs_NN(expt_ID, BSP_files, yr_init, dyrs, init_early, init_late, fluxes, weight_multiplier, use_DIC):

    import os

    ''' Author: Neill Mackay (2024)
    A version of the MTM_obs_NN.ipynb Jupyter script for running the Optimum Transformation Method applied to observations using Ehmen neural network interior carbon reconstruction that takes various inputs
    expt_ID: ID to create filename to save the OTM run outputs
    BSP_files: Folder containing BSP-binned data
    yr_init: Year that BSP data starts (1992 for ECCO data, 1990 for EN4 data)
    dyrs: The length in years of the early and late time averaging periods
    init_early: The year of the start of the early averaging period
    init_late: The year of the start of the late averaging period
    use_DIC: Boolean flag, whether to use DIC data instead of C*
    fluxes: Which air-sea flux prior to use
    '''

    # Load optional flags

    DIC=use_DIC
    
    ## Load a text file that includes the list of strings pointing to the relevant data. 
    ## This ensures that pull requests don't continuously overwrite hardcoded file paths.

    filename = 'folders_obs.txt'
    with open(filename) as f:
        mylist = f.read().splitlines() 

    ## Set run ID

    print('Your Experiment ID is '+expt_ID)
    date = datetime.today().strftime('%d%m%Y')
    dir = '../Outputs/'+ date +'/'
    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    file_path = dir + expt_ID + '.nc'


    ### Define key parameters

    obs_BSP_data = xr.open_mfdataset(BSP_files)

    Cflux_prior = getattr(obs_BSP_data, fluxes + '_sum_hist')
    print(Cflux_prior.attrs['description'])
   
    # Range of years of which 'early' and 'late' are defined

    month_init_early=(init_early-yr_init)*12
    month_init_late=(init_late-yr_init)*12
    Early_period = (np.array([month_init_early,month_init_early+dyrs*12]))
    Late_period = (np.array([month_init_late,month_init_late+dyrs*12]))
    range_yrs = init_late-init_early+1

    print(Early_period[0])
    print(Early_period[1])
    print(Late_period[0])
    print(Late_period[1])

    # Establish basic constants and TS grid
    Cp=4000
    rho=1029
    #S0=35
    #T0=273.15
    #T0=0
    #C0=850 # For C*
    #if DIC:
    #    C0=2200 # For DIC

    # Thermal expansion, haline contraction and true scale factor
    alph = 1.7657*10**-4
    bet = 7.5544*10**-4
    ST_scale=bet/alph

    areanorming = 10**12 #normalising coeffcients
    volnorming = 10**15 #normalising coeffcients


    saveplots=False

    # If Surface fluxes are available

    ### Load Data

    # Calculate tracer offsets from median of BSP values

    T0 = np.nanmedian(obs_BSP_data.T_mean_hist.isel(Time=slice(Early_period[0],Late_period[1])).values.flatten())
    S0 = np.nanmedian(obs_BSP_data.S_mean_hist.isel(Time=slice(Early_period[0],Late_period[1])).values.flatten())
    C0 = np.nanmedian(obs_BSP_data.Cstar_mean_hist.isel(Time=slice(Early_period[0],Late_period[1])).values.flatten())
    if DIC:
        C0 = np.nanmedian(obs_BSP_data.DIC_mean_hist.isel(Time=slice(Early_period[0],Late_period[1])).values.flatten())


    prior_opt = False # If using the result of an earlier optimisation as a prior 

    ## Early Period
    Part_early = obs_BSP_data.Partitions_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
    SA_early =  obs_BSP_data.S_mean_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')-S0
    CT_early = obs_BSP_data.T_mean_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')-T0
    C_early = obs_BSP_data.Cstar_mean_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')-C0
    if DIC:
        C_early = obs_BSP_data.DIC_mean_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')-C0
    V_early = obs_BSP_data.V_sum_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')
    A_early = obs_BSP_data.A_sum_hist.isel(Time=slice(Early_period[0],Early_period[1])).mean('Time')

    ## Late Period
    Part_late = obs_BSP_data.Partitions_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
    SA_late =  obs_BSP_data.S_mean_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')-S0
    CT_late = obs_BSP_data.T_mean_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')-T0
    C_late = obs_BSP_data.Cstar_mean_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')-C0
    if DIC:
        C_late = obs_BSP_data.DIC_mean_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')-C0
    V_late = obs_BSP_data.V_sum_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')
    A_late = obs_BSP_data.A_sum_hist.isel(Time=slice(Late_period[0],Late_period[1])).mean('Time')

    Basins = obs_BSP_data.Basin.values
    tree_depth=obs_BSP_data.Depth.size

    # Time variable

    t1=(str(yr_init)+"-01-01")
    t2 = "2019-12-01"

    #ERA5_time=xr.open_mfdataset(mylist[1], decode_times=True, decode_cf=True).sel(time=slice(t1,t2)).time
    ERA5_time=xr.open_mfdataset('ERA5_time.nc', decode_times=True, decode_cf=True).sel(time=slice(t1,t2)).time

    ERA5_dt = (np.hstack((np.diff(ERA5_time.values)/10**9,31*24*3600))).astype('float64') # ERA5 time vector finishes 1st December
    ERA5_dt = ERA5_dt[np.newaxis,0:obs_BSP_data.Time.size,np.newaxis]

    ##

    obs_dt = ERA5_dt # Landschutzer time vector has timestamps mid-month so we use ERA5_dt


    ##### Calculate the area weight scale (1/std(T)), 1/std(S))

    #T_nonan_std = obs_BSP_data.T_mean_hist.std(skipna=True).values
    #S_nonan_std = obs_BSP_data.S_mean_hist.std(skipna=True).values

    #T_scale = 1/T_nonan_std
    #S_scale = 1/S_nonan_std

    # Do carbon using BSP binned data

    #C_nonan_std = obs_BSP_data.Cstar_mean_hist.std(skipna=True).values
    #if DIC:
    #    C_nonan_std = obs_BSP_data.DIC_mean_hist.std(skipna=True).values

    #C_scale = 1/C_nonan_std

    ## Load prior flux data

    ## Calculate the cumulative time integrated surface fluxes

    if obs_BSP_data.Time.size == Late_period[-1]:

        HFDS_cumsum = ((obs_BSP_data.hfds_sum_hist*obs_dt).cumsum('Time').isel(Time=-1))\
            -((obs_BSP_data.hfds_sum_hist*obs_dt).cumsum('Time').isel(Time=Early_period[0])) # units: J
        WFO_cumsum = ((obs_BSP_data.wfo_sum_hist*obs_dt).cumsum('Time').isel(Time=-1))\
             -((obs_BSP_data.wfo_sum_hist*obs_dt).cumsum('Time').isel(Time=Early_period[0]))# units: kg
        Cflux_cumsum = ((Cflux_prior*obs_dt).cumsum('Time').isel(Time=-1))\
                -((Cflux_prior*obs_dt).cumsum('Time').isel(Time=Early_period[0])) # units: mmol-C
    else:

        HFDS_cumsum = ((obs_BSP_data.hfds_sum_hist*obs_dt).cumsum('Time').isel(Time=Late_period[-1]))\
            -((obs_BSP_data.hfds_sum_hist*obs_dt).cumsum('Time').isel(Time=Early_period[0])) # units: J
        WFO_cumsum = ((obs_BSP_data.wfo_sum_hist*obs_dt).cumsum('Time').isel(Time=Late_period[-1]))\
             -((obs_BSP_data.wfo_sum_hist*obs_dt).cumsum('Time').isel(Time=Early_period[0]))# units: kg
        Cflux_cumsum = ((Cflux_prior*obs_dt).cumsum('Time').isel(Time=Late_period[-1]))\
                -((Cflux_prior*obs_dt).cumsum('Time').isel(Time=Early_period[0])) # units: mmol-C
    
    dt_cumsum = obs_dt[0,Early_period[0]:Late_period[-1],0].sum() # units: seconds
    
    
    ## The final dflux value is then scaled by the time between the middle of t1 and the middle of t2
    dhfds = HFDS_cumsum*((np.mean(Late_period)-np.mean(Early_period))/(Late_period[-1]-Early_period[0]))
    dwfo = WFO_cumsum*((np.mean(Late_period)-np.mean(Early_period))/(Late_period[-1]-Early_period[0]))
    
    dCflux = Cflux_cumsum*((np.mean(Late_period)-np.mean(Early_period))/(Late_period[-1]-Early_period[0]))
    
    dt = dt_cumsum*((np.mean(Late_period)-np.mean(Early_period))/(Late_period[-1]-Early_period[0]))

    ## Convert dflux to equivalent T or S change

    dT_hfds = dhfds/(Cp*rho*V_early) # units: C
    dS_wfo = -dwfo*S0/(rho*V_early) # units: g/kg
    dC_Cflux = dCflux/V_early # units: mmol-C/m^3


    # Flatten the early and late variables to a 1D array
    Vol_1 = V_early.values.flatten()
    Vol_2 = V_late.values.flatten()
    S_1 = SA_early.values.flatten()
    S_2 = SA_late.values.flatten()
    T_1 = CT_early.values.flatten()
    T_2 = CT_late.values.flatten()
    C_1 = C_early.values.flatten()
    C_2 = C_late.values.flatten()
    A_1 = A_early.values.flatten()
    A_2 = A_late.values.flatten()


    # Do the same for basin index
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

    #... and for the edges of the BSP bins
    ## Here we calculate the mean TS edges averaged over both early and late times
    S_start = (0.5*(Part_early.values[:,:,0]+Part_late.values[:,:,0])).flatten()
    S_end = (0.5*(Part_early.values[:,:,1]+Part_late.values[:,:,1])).flatten()
    T_start = (0.5*(Part_early.values[:,:,2]+Part_late.values[:,:,2])).flatten()
    T_end = (0.5*(Part_early.values[:,:,3]+Part_late.values[:,:,3])).flatten()

    # Any NaNs are zeroed out
    S_1[np.isnan(S_1)] = 0
    S_2[np.isnan(S_2)] = 0
    T_1[np.isnan(T_1)] = 0
    T_2[np.isnan(T_2)] = 0
    C_1[np.isnan(C_1)] = 0
    C_2[np.isnan(C_2)] = 0

    ## Prior fluxes

    S_pre = SA_early.values.flatten()
    S_1 = SA_early.values.flatten()+dS_wfo.values.flatten()
    T_pre = CT_early.values.flatten()
    T_1 = CT_early.values.flatten()+dT_hfds.values.flatten()

    S_1[np.isnan(S_1)] = 0
    T_1[np.isnan(T_1)] = 0
    S_1[~np.isfinite(S_1)] = 0
    T_1[~np.isfinite(T_1)] = 0
    
    C_pre = C_early.values.flatten()         
    C_1 = C_early.values.flatten() + dC_Cflux.values.flatten()
    C_1[np.isnan(C_1)] = 0
    C_1[~np.isfinite(C_1)] = 0


    ## Here, we create the tracers and volumes matrices, which will be fed into the MTM function

    volumes = np.stack((Vol_1, Vol_2), axis=0)/volnorming # Shape: [2 x N]

    salinities = np.stack((S_1, S_2), axis=0)
    temps = np.stack((T_1, T_2), axis=0)
    carbon = np.stack((C_1, C_2), axis=0)


    tracers = np.stack((salinities, temps, carbon),axis=1) # Shape: [2 x M x N], where M = 2 for just T and S, and M>2 for T,S+other tracers


    print('Total number of bins =', int(Vol_1.shape[0]))
    N = int(Vol_1.shape[0])


    ### Define constraints

    # We must define whether a BSP bin is allowed to transport volume to another BSP bin. In the simplest case, all bins are allowed to transport to one another - but this yields nonphysical transport across vast distances and TS bounds.
    # To improve on this, two connectivity constraints are used: 
    # 1) Are the basins adjacent? This is defined via the connectivity array, below
    # 2) If YES, do the BSP bins have overlapping (or the same) TS boundaries?\
    # If yes, the bins are connected. If no, they are not.

    # Array defining the connection between the 9 basins;
    # 1 = connected, 0 = disconnected
    # connectivity_array = np.ones((Basins.size,Basins.size))

    connectivity_array = np.zeros((Basins.size,Basins.size))

    #connectivity_array[0,:] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    connectivity_array[0,:] = [1, 1, 0, 0, 0, 0, 0, 1, 0] # Open Bering Strait
    connectivity_array[1,:] = [1, 1, 1, 0, 0, 0, 0, 0, 0]
    connectivity_array[2,:] = [0, 1, 1, 1, 0, 0, 0, 0, 0]
    connectivity_array[3,:] = [0, 0, 1, 1, 0, 0, 0, 0, 1]
    connectivity_array[4,:] = [0, 0, 0, 0, 1, 0, 1, 0, 1]
    connectivity_array[5,:] = [0, 0, 0, 0, 0, 1, 1, 0, 1]
    connectivity_array[6,:] = [0, 0, 0, 0, 1, 1, 1, 1, 0]
    #connectivity_array[7,:] = [0, 0, 0, 0, 0, 0, 1, 1, 0]
    connectivity_array[7,:] = [1, 0, 0, 0, 0, 0, 1, 1, 0] # Open Bering Strait
    connectivity_array[8,:] = [0, 0, 0, 1, 1, 1, 0, 0, 1]

    d = {Basins[0]: connectivity_array.T[0,:],\
        Basins[1]: connectivity_array.T[1,:],\
        Basins[2]: connectivity_array.T[2,:],\
        Basins[3]: connectivity_array.T[3,:],\
        Basins[4]: connectivity_array.T[4,:],\
        Basins[5]: connectivity_array.T[5,:],\
        Basins[6]: connectivity_array.T[6,:],\
        Basins[7]: connectivity_array.T[7,:],\
        Basins[8]: connectivity_array.T[8,:]}
    
    table = pd.DataFrame(data=d, index=Basins)
    table

    # Array defining the transport between the 9 basins;
    # +/-1 = connected (North = +, East = +), 0 = no constraint

    ncons=3
    transport_array = np.zeros((Basins.size,Basins.size,ncons))

    # Constrain ITF
    transport_array[4,:,0] = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    transport_array[6,:,0] = [0, 0, 0, 0, -1, 0, 0, 0, 0]

    # Constrain Bering Strait
    transport_array[0,:,1] = [0, 0, 0, 0, 0, 0, 0, -1, 0]
    transport_array[7,:,2] = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    tablecons=0

    d = {Basins[0]: transport_array[:,:,tablecons].T[0,:],\
        Basins[1]: transport_array[:,:,tablecons].T[1,:],\
        Basins[2]: transport_array[:,:,tablecons].T[2,:],\
        Basins[3]: transport_array[:,:,tablecons].T[3,:],\
        Basins[4]: transport_array[:,:,tablecons].T[4,:],\
        Basins[5]: transport_array[:,:,tablecons].T[5,:],\
        Basins[6]: transport_array[:,:,tablecons].T[6,:],\
        Basins[7]: transport_array[:,:,tablecons].T[7,:],\
        Basins[8]: transport_array[:,:,tablecons].T[8,:]}
    
    table = pd.DataFrame(data=d, index=Basins)
    table

    # Define whether a bin is connected to every other bin
    # The two constraints used are: are the basins adjacent? 
    # If yes, are the bin indices the same? 
    # If yes, the bins are connected; if no, they are not connected. 

    # connected = np.ones((Vol_1.size, Vol_1.size)) ## For all connections case

    trans_big = np.zeros((Vol_1.size, Vol_1.size, ncons))
    connected = np.zeros((Vol_1.size, Vol_1.size))

    for i in tqdm(range(Vol_1.size)):
        for j in range(Vol_2.size):
            for k in range(ncons):
                trans_big[i,j,k] = transport_array[int(Basin_1_inds[i]), int(Basin_2_inds[j]),k]
            if connectivity_array[int(Basin_1_inds[i]), int(Basin_2_inds[j])]>0:
                connected[i,j] = 1 ## For NO DOCKING case
                ## UNCOMMENT BELOW IF YOU WANT DOCKING AGAIN
                #if Basin_names[i] == Basin_names[j]:
                #    connected[i,j] = connectivity_array[int(Basin_1_inds[i]), int(Basin_2_inds[j])]
                #elif S_start[i]==S_start[j] and T_start[i]==T_start[j]:
                #    connected[i,j] = connectivity_array[int(Basin_1_inds[i]), int(Basin_2_inds[j])]

    constraints = connected # Shape: An [N x N] matrix

    transport = trans_big  # Shape: An [N x N] matrix


    ### Defining weights

    S_scale=1/(volumes[1,:]*tracers[1,0,:]).std()
    T_scale=1/(volumes[1,:]*tracers[1,1,:]).std()
    C_scale=1/(volumes[1,:]*tracers[1,2,:]).std()

    ## We create a weight matrix
    # For sqrt(1/Area)
    A_2_modified = A_2.copy()
    #A_2_modified[A_2_modified==0] = 10**-15 # 10**-15 is the lowest I can make it for tree_depth=5 without it breaking
    #A_2_modified[A_2_modified==0] = 10**5 # 10**-15 is the lowest I can make it for tree_depth=5 without it breaking
    A_2_modified[A_2_modified==0] = np.nanmin(A_2[A_2>0])

    #print(np.nanmin(A_2[A_2>0]))

    #area_weight = np.sqrt(areanorming/A_2_modified)

    area_cons = A_2/A_2-1
    area_cons[np.isnan(area_cons)] = 1

    ## For log(Area)
    area_weight = np.log10(areanorming)/(np.log10(A_2))
    #area_weight[area_weight==0] = 100
    area_weight[A_2==0] = weight_multiplier*np.nanmax(area_weight)

    #weights = np.stack((ST_scale*area_weight,area_weight,np.ones((C_1.size))), axis=0) # Shape: An [M x N] matrix
    #weights = np.stack((S_scale*area_weight,T_scale*area_weight,np.ones((C_1.size))), axis=0) # Shape: An [M x N] matrix
    #hard_area = np.stack((area_cons,area_cons), axis=0)

    weights = np.stack((S_scale*area_weight,T_scale*area_weight,C_scale*area_weight), axis=0) # Shape: An [M x N] matrix
    hard_area = np.stack((area_cons,area_cons,area_cons), axis=0)

    # Weight by volume

    #vol_weight = np.sqrt(volnorming/Vol_2)
    #vol_weight[Vol_2==0] = 10**4
    #weights = np.stack((S_scale*vol_weight,T_scale*vol_weight,C_scale*vol_weight), axis=0) # Shape: An [M x N] matrix


    ### Run optimisation

    ## We run the optimiser to get the transports between water masses and the T,S mixed and T,S adjustment
    # threshold = 1*(range_yrs*yr2sec/volnorming) #xxx m/s

    yr2sec=3600*24*365.25

    trans_val1 = -12*(10**6*(dt/volnorming)) # ITF net transport

    trans_val3 = 18*(10**6*(dt/volnorming)) # Atlantic northward transport
    #trans_val3 = -29.2*(10**6*(dt/volnorming)) # Atlantic southward transport

    trans_val4 = 0*(10**6*(dt/volnorming)) # Bering Straight southward
    trans_val5 = 1.1*(10**6*(dt/volnorming)) # Bering Straight northward

    trans_val=np.hstack((trans_val1,trans_val4, trans_val5))
    #trans_val=trans_val1

    ## Function has form MTM.optimise(tracers=tracers, volumes=volumes, cons_matrix = constraints, 
    #                                 trans = [transport,trans_val], Asection = [connected_Asection,threshold], weights=weights, hard_area = hard_area, transmax = [transport_max,transmax_val])
    #     '''constraint = [G @ Q = V2 * C2 - G @ C1]
    #         cost = cp.sum_squares(Q*V1/A1)

    #         where C1 is the initial and C2 is the final tracer concentrations.'''

    #result = MTM_3.optimise(tracers=tracers, volumes=volumes, cons_matrix=constraints, weights=weights, hard_area = hard_area, trans = [transport,trans_val])
    result = MTM_3.optimise(tracers=tracers, volumes=volumes, cons_matrix=constraints, weights=weights, trans = [transport,trans_val])
    #result = MTM_3.optimise(tracers=tracers, volumes=volumes, cons_matrix=constraints, weights=weights)

    if result == None:
        return


    g_ij = result['g_ij'] ## An [N x N] matrix of transports between WMs
    Mixing = result['Mixing'] ## An [M x N] matrix of dtracer mixing for each WM
    Adjustment = result['Adjustment'] ## An [M x N] matrix of dtracer adjustment for each WM
    G = result['G']

    ## Break down the Mixing and Adjustment matrices into their constituent tracers
    dT_mixing = Mixing[1,:]
    dS_mixing = Mixing[0,:]

    dT_adj = Adjustment[1,:]
    dS_adj = Adjustment[0,:]

    ## Uncomment if M>2
    dC_mixing = Mixing[2,:]
    dC_adj = Adjustment[2,:]

    ## Plot budget closure for OTM solution

    # Temperature

    T = CT_early.values.flatten()+dT_hfds.values.flatten()
    T[np.isnan(T)]=0
    dT_mix=np.matmul(T,g_ij*volnorming)/V_late.values.flatten() - (CT_early.values.flatten()+dT_hfds.values.flatten())

    # Salinity

    S = SA_early.values.flatten()+dS_wfo.values.flatten()
    S[np.isnan(S)]=0
    dS_mix=np.matmul(S,g_ij*volnorming)/V_late.values.flatten() - (SA_early.values.flatten()+dS_wfo.values.flatten())

    # Carbon

    C = C_early.values.flatten()+dC_Cflux.values.flatten()
    C[np.isnan(C)]=0
    dC_mix=np.matmul(C,g_ij*volnorming)/V_late.values.flatten() - (C_early.values.flatten()+dC_Cflux.values.flatten())

    # Plot

    fig, axes = plt.subplots(1,3,figsize=(15,4))

    axes[0].plot(dT_hfds.values.flatten()+dT_mix+dT_adj.flatten())
    axes[0].plot(CT_late.values.flatten() - CT_early.values.flatten())
    axes[0].set_title('OTM temperature budget')

    axes[1].plot(dS_wfo.values.flatten()+dS_mix+dS_adj.flatten())
    axes[1].plot(SA_late.values.flatten() - SA_early.values.flatten())
    axes[1].set_title('OTM salt budget')

    axes[2].plot(dC_Cflux.values.flatten()+dC_mix+dC_adj.flatten())
    axes[2].plot(C_late.values.flatten() - C_early.values.flatten())
    axes[2].set_title('OTM carbon budget')


    ### Save outputs

    ## Save MTM outputs to netcdf

    import os
    if os.path.exists(file_path):
       os.remove(file_path)
       print('File deleted')

    da_dT_mixing = xr.DataArray(data = dT_mixing, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Temperature Mixing", units="\Delta K", variable_id="obs Tmix"))
    da_dS_mixing = xr.DataArray(data = dS_mixing, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Salinity Mixing", units="\Delta g/kg", variable_id="obs Smix"))
    da_dC_mixing = xr.DataArray(data = dC_mixing, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Carbon Mixing", units="\Delta mmol/m^3", variable_id="obs Cmix"))
    da_dT_adjustment = xr.DataArray(data = dT_adj, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Temperature Adjustment", units="\Delta K", variable_id="obs Tadj"))
    da_dS_adjustment = xr.DataArray(data = dS_adj, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Salinity Adjustment", units="\Delta g/kg", variable_id="obs Sadj"))
    da_dC_adjustment = xr.DataArray(data = dC_adj, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Carbon Adjustment", units="\Delta mmol/m^3", variable_id="obs Cadj"))
    da_dC_Cflux = xr.DataArray(data = dC_Cflux.values.flatten(), dims = ["WM_number"],
                           coords=dict(WM_number = np.arange(0,N)),
                        attrs=dict(description="Carbon flux", units="\Delta mmol/m^3", variable_id="obs Cflux"))
    da_gij = xr.DataArray(data = g_ij*volnorming, dims = ["WM_initial", "WM_final"],
                               coords=dict(WM_initial = np.arange(0,N), WM_final = np.arange(0,N)),
                            attrs=dict(description="Volume transport", units="m^3", variable_id="obs Gij"))
    da_Vol_early = xr.DataArray(data = Vol_1, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Early period volume", units="m^3", variable_id="Vol init"))
    da_Vol_late = xr.DataArray(data = Vol_2, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Late period volume", units="m^3", variable_id="Vol final"))
    da_A_early = xr.DataArray(data = A_1, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Early period area", units="m^2", variable_id="Area init"))
    da_A_late = xr.DataArray(data = A_2, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Late period area", units="m^2", variable_id="Area final"))
    da_C_early = xr.DataArray(data = C_1, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Early period carbon", units="mmol-C/m^3", variable_id="C early"))
    da_C_late = xr.DataArray(data = C_2, dims = ["WM_number"],
                               coords=dict(WM_number = np.arange(0,N)),
                            attrs=dict(description="Late period carbon", units="mmol-C/m^3", variable_id="C late"))
    da_connected = xr.DataArray(data = connected, dims = ["WM_initial", "WM_final"],
                                coords=dict(WM_initial = np.arange(0,N), WM_final = np.arange(0,N)),
                                attrs=dict(description="Connectivity matrix", units="None", variable_id="connectivity"))
    da_scale = xr.DataArray(data = np.array((T_scale,S_scale,C_scale)), dims = ["Tracer"])

    ## Create xarray DataSet that will hold all these DataArrays
    ds_BSP = xr.Dataset()
    ds_BSP['dT_mixing'] = da_dT_mixing
    ds_BSP['dS_mixing'] = da_dS_mixing
    ds_BSP['dC_mixing'] = da_dC_mixing
    ds_BSP['dT_adjustment'] = da_dT_adjustment
    ds_BSP['dS_adjustment'] = da_dS_adjustment
    ds_BSP['dC_adjustment'] = da_dC_adjustment
    ds_BSP['dC_Cflux'] = da_dC_Cflux
    ds_BSP['C_early'] = da_C_early
    ds_BSP['C_late'] = da_C_late
    ds_BSP['gij'] = da_gij
    ds_BSP['Vol_early'] = da_Vol_early
    ds_BSP['Vol_late'] = da_Vol_late
    ds_BSP['A_early'] = da_A_early
    ds_BSP['A_late'] = da_A_late
    ds_BSP['init_early'] = init_early
    ds_BSP['init_late'] = init_late
    ds_BSP['dyrs'] = dyrs
    ds_BSP['Basins'] = Basins
    ds_BSP['tree_depth'] = tree_depth
    ds_BSP['dt'] = dt
    ds_BSP['connectivity'] = da_connected
    ds_BSP['Scale_factors']=da_scale

    ds_BSP.to_netcdf(file_path, mode='w')
    print('File saved')

    return

