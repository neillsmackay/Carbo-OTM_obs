#!/usr/bin/env python
# coding: utf-8

# #### Remapping BSP into Eulerian coordinates for EN4 outputs

# In[2]:


import xarray as xr
import math
import numpy as np
import matplotlib.pyplot as plt
import gsw
import sys

# Load MTM result

yr_init=1990 # Applies to T and S and BSP data
start_year = 1990
end_year = 2018

BSP_path=('../BSP_obs/Ehmen_NN/ERA5/nitrate_Cstar_components/BSP_ERA5_NN_*.nc')

start_month = (start_year - yr_init)*12
end_month = (end_year - yr_init)*12

#MTM_result = xr.open_mfdataset('../Outputs/RFRv2_ERA5_1995_2005_10_transports.nc')

#tree_depth=int(math.log2(MTM_result.tree_depth))

#month_init_early=(MTM_result.init_early-yr_init)*12
#month_init_late=(MTM_result.init_late-yr_init)*12
#Early_period = (np.array([month_init_early,month_init_early+MTM_result.dyrs*12]))
#Late_period = (np.array([month_init_late,month_init_late+MTM_result.dyrs*12]))


## LOAD T and S data from a gridded observations

chunks = {'time': 12}

filename = 'folders_obs.txt'
with open(filename) as f:
    mylist = f.read().splitlines() 
    
#EN4_data = xr.open_mfdataset(mylist[0], decode_times=True, decode_cf=True).isel(time=slice(Early_period[0],Early_period[1])).chunk(chunks = chunks)
EN4_data = xr.open_mfdataset(mylist[20], decode_times=True, decode_cf=True).isel(time=slice(start_month,end_month)).chunk(chunks = chunks)

# Get land mask

land_mask = xr.ones_like(EN4_data.temperature.isel(time=0)).rename(new_name_or_name_dict='land mask')
land_mask = (land_mask * ~np.isnan(EN4_data.temperature.isel(time=0))).astype('bool')

# In[13]:


## Convert EN4 data from potential temperature and practical salinity to conservative temperature and absolute salinity

T0=273.15

EN4_P = gsw.p_from_z(-EN4_data.depth,EN4_data.lat)
EN4_SA = gsw.SA_from_SP(EN4_data.salinity,EN4_P,EN4_data.lon,EN4_data.lat)
EN4_CT = gsw.CT_from_pt(EN4_SA,EN4_data.temperature-T0)

T_all = EN4_CT*(~land_mask-1)/(~land_mask-1)
S_all = EN4_SA*(~land_mask-1)/(~land_mask-1)


# In[14]:


## Remapping

BSP_data = xr.open_mfdataset(BSP_path)
#partitions=BSP_data.Partitions_hist.isel(Time=slice(Early_period[0],Early_period[1]), Basin=0)
partitions=BSP_data.Partitions_hist.isel(Time=slice(start_month,end_month), Basin=0)

chunks2={'tree_depth': 16}

ti =  int(sys.argv[1])

T = T_all.isel(time = ti)
S = S_all.isel(time = ti)
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

outputpath = ('/home/users/nmackay/unicorns/MTM/Remapping_mask/EN4/Remapping_mask_EN4')

MTM_remapping=xr.Dataset()
MTM_remapping['da_fuzz'] = da_fuzz.chunk(chunks = chunks2).astype('float32')
MTM_remapping['partitions'] = partitions

MTM_remapping.to_netcdf(outputpath + '.' + str(ti) + '.nc')


