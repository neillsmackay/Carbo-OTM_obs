#!/usr/bin/env python
# coding: utf-8

# ### Ensemble runs of OTM applied to observations

# In[1]:


from WM_Methods import run_OTM
import numpy as np
import os
import xarray as xr


# In[2]:


# Set up parameters for ensemble

#avg_periods = np.array([[1990,2000,6],[2000,2010,6]])
avg_periods = np.array([[2000,2010,6]])


# In[ ]:


# Run ensemble of averaging periods and interior carbon reconstruction ensemble members

forcing = 'ECCO'
cstardef='nitrate'
flux_priors = ('Cflux3a','Cflux3b','Cflux3c','Cflux3d','Cflux3e','Cflux3f')
weight_multiplier = 8

BSP_basedir = str('/home/users/nmackay/MTM/BSP_obs/Ehmen_NN/' + forcing + '/' + cstardef + '_Cstar_components/')

if forcing == 'ERA5' or forcing == 'JRA55':
    yr_init = 1990
elif forcing == 'ECCO':
    yr_init = 1992

print('BSP directory: ',BSP_basedir)
print('Initial year: ',yr_init)

for prior in np.arange(len(flux_priors)):
    
    for periods in np.arange(0,avg_periods.shape[0]):

        for member in np.arange(1,16):

            BSP_files = str(BSP_basedir + 'member_' + str(member) + '/BSP_' + forcing + '_NN*.nc')

            exec('prior_name=xr.open_mfdataset(BSP_files).' + flux_priors[prior] + '_sum_hist.attrs[\'description\']')

            runname = forcing + '_' + str(avg_periods[periods,0]) + '_' + str(avg_periods[periods,1]) + '_' + str(avg_periods[periods,2]) + '_' + prior_name.split()[0] + '_' + cstardef + '_outcropweight_' + str(weight_multiplier) + '_member' + str(member)

            print(runname)

            if os.path.isfile(BSP_basedir + 'member_' + str(member) + '/BSP_' + forcing + '_NN_0_11.nc'):

                run_OTM.OTM_obs_NN(runname,BSP_files,yr_init,avg_periods[periods,2],avg_periods[periods,0],avg_periods[periods,1],flux_priors[prior],weight_multiplier,use_DIC=False)


# In[ ]:




