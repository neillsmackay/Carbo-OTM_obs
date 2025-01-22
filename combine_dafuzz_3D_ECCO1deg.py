#!/usr/bin/env python
# coding: utf-8

yr_init=1992 # Applies to T and S and BSP data
start_year = 1992
end_year = 2018

start_month = (start_year - yr_init)*12
end_month = (end_year - yr_init)*12

# Combine 3D fuzz files into single Eulerian mask

import xarray as xr


# Load fuzz files and calculate mask

chunks={'tree_depth': 8, 'time': 1}

path = ('/home/users/nmackay/unicorns/MTM/Remapping_mask/ECCO1deg/')

Remapping_mask = xr.open_mfdataset(path+'Remapping_mask_ECCO1deg.*.nc', chunks = chunks, data_vars = {'da_fuzz'}, combine = 'nested', concat_dim = 'time')

Remapping_mask = Remapping_mask.isel(time=slice(start_month,end_month))

Eulerian_mask = Remapping_mask.da_fuzz.sum('time')/Remapping_mask.da_fuzz.sum('time').sum('tree_depth')


# In[8]:


# Save to netcdf

MTM_remapping=xr.Dataset()
MTM_remapping['Eulerian_mask'] = Eulerian_mask.astype('float32')
MTM_remapping['partitions'] = Remapping_mask.partitions

MTM_remapping.to_netcdf(path + 'Remapping_mask_ECCO1deg_' + str(start_year) + '_' + str(end_year) + '.nc')


