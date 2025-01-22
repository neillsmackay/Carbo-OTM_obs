#!/usr/bin/env python
# coding: utf-8


yr_init=1990 # Applies to T and S and BSP data
start_year = 2002
end_year = 2010


start_month = (start_year - yr_init)*12
end_month = (end_year - yr_init)*12

# Combine 3D fuzz files into single Eulerian mask

import xarray as xr


# Load fuzz files and calculate mask

chunks={'tree_depth': 8, 'time': 1}

path = ('/home/users/nmackay/unicorns/MTM/Remapping_mask/EN4/')

Remapping_mask = xr.open_mfdataset(path+'Remapping_mask_EN4.*.nc', chunks = chunks, data_vars = {'da_fuzz'}, combine = 'nested', concat_dim = 'time')

Remapping_mask = Remapping_mask.isel(time=slice(start_month,end_month))

Eulerian_mask = Remapping_mask.da_fuzz.sum('time')/Remapping_mask.da_fuzz.sum('time').sum('tree_depth')


# Save to netcdf

MTM_remapping=xr.Dataset()
MTM_remapping['Eulerian_mask'] = Eulerian_mask.astype('float32')
MTM_remapping['partitions'] = Remapping_mask.partitions

MTM_remapping.to_netcdf(path + 'Remapping_mask_EN4_' + str(start_year) + '_' + str(end_year) + '.nc')


