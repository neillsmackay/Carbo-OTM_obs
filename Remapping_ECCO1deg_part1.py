#!/usr/bin/env python
# coding: utf-8

# #### Remapping BSP into Eulerian coordinates for ECCO outputs interpolated onto 1 degree EN4 grid

import xarray as xr
import math
import numpy as np
import matplotlib.pyplot as plt
import gsw
import sys

# Load MTM result

yr_init=1992 # Applies to T and S and BSP data
start_year = 1992
end_year = 2018

BSP_path=('../BSP_obs/Ehmen_NN/ECCO/nitrate_Cstar_components/BSP_ECCO_NN_*.nc')

start_month = (start_year - yr_init)*12
end_month = (end_year - yr_init)*12


#MTM_result = xr.open_mfdataset('../Outputs/NN_ECCO_1995_2005_10_DIC.nc')

#month_init_early=(MTM_result.init_early-yr_init)*12
#month_init_late=(MTM_result.init_late-yr_init)*12
#Early_period = (np.array([month_init_early,month_init_early+MTM_result.dyrs*12]))
#Late_period = (np.array([month_init_late,month_init_late+MTM_result.dyrs*12]))


## LOAD T and S data from a gridded observations

chunks = {'time': 12}

filename = 'folders_obs.txt'
with open(filename) as f:
    mylist = f.read().splitlines() 
    
#ECCO_data = xr.open_mfdataset(mylist[12], decode_times=True, decode_cf=True).isel(time=slice(Early_period[0],Early_period[1])).chunk(chunks = chunks)
ECCO_data = xr.open_mfdataset(mylist[12], decode_times=True, decode_cf=True).isel(time=slice(start_month,end_month)).chunk(chunks = chunks)

# Get land mask

EN4_data = xr.open_mfdataset(mylist[20], decode_times=True, decode_cf=True).isel(time=0)

land_mask = xr.ones_like(EN4_data.temperature).rename(new_name_or_name_dict='land mask')
land_mask = (land_mask * ~np.isnan(EN4_data.temperature)).astype('bool')


## Remapping

BSP_data = xr.open_mfdataset(BSP_path)
#partitions=BSP_data.Partitions_hist.isel(Time=slice(Early_period[0],Early_period[1]), Basin=0)
partitions=BSP_data.Partitions_hist.isel(Time=slice(start_month,end_month), Basin=0)

ti =  int(sys.argv[1])

T = ECCO_data.THETA.isel(time = ti)*(~land_mask-1)/(~land_mask-1)
S = ECCO_data.SALT.isel(time = ti)*(~land_mask-1)/(~land_mask-1)
partitions = partitions.isel(Time = ti)



from tqdm import tqdm
da_fuzz = xr.zeros_like(S).expand_dims({'tree_depth':BSP_data.Depth.size}).assign_coords({'tree_depth':BSP_data.Depth.values})
        
for j in tqdm((range(BSP_data.Depth.size))):
    da_fuzz[j] = xr.where((S>partitions[j,0])&\
                            (T>partitions[j,2])&\
                            (S<=partitions[j,1])&\
                            (T<=partitions[j,3]),\
                            1, 0)


## Save da_fuzz to netcdf

chunks2={'tree_depth': 8}

outputpath = ('/home/users/nmackay/unicorns/MTM/Remapping_mask/ECCO1deg/Remapping_mask_ECCO1deg')

MTM_remapping=xr.Dataset()
MTM_remapping['da_fuzz'] = da_fuzz.chunk(chunks = chunks2).astype('float32')
MTM_remapping['partitions'] = partitions

MTM_remapping.to_netcdf(outputpath + '.' + str(ti) + '.nc')


# In[ ]:




