#!/usr/bin/env python
# coding: utf-8

# #### Water-mass Methods Package
# ##### BSP script by Neill Mackay
# 
# This script runs the Binary Space Partitioning Code on a combination of observational datasets:
# 
# Interior T&S: EN4
# 
# Surface heat and freshwater fluxes: JRA55
# 
# Interior carbon: Tobias Ehmen neural network predictions

# In[1]:


## Import the BSP component of the WM_Methods package
from WM_Methods import BSP
## Other required packages for calculations and plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import itertools
import xarray as xr
import sys
from tqdm import tqdm
import dask as da
from scipy.interpolate import griddata
from scipy.interpolate import interpn
import gsw
import xesmf as xe

# In[7]:


t1 = "1990-01-01"
t2 = "2019-12-31"

chunks = {'time': 12}
chunks2 = {'time': 24}
chunks3 = {'time': 120}

## Load a text file that includes the list of strings pointing to the relevant data. 
## This ensures that pull requests don't continuously overwrite hardcoded file paths.

member=int(sys.argv[2]) # Set to 'ens' for ensemble mean
#member='ens' # Set to 'ens' for ensemble mean
cstardef='nitrate'
filename = 'folders_obs_member' + str(member) + '.txt'
if member=='ens':
    filename = 'folders_obs_JRA55.txt'

with open(filename) as f:
    mylist = f.read().splitlines()

print(mylist)

## Load the data using xarray. 

EN4_data = xr.open_mfdataset(mylist[20], decode_times=True, decode_cf=True).sel(time=slice(t1,t2)).chunk(chunks = chunks)
LS_data = xr.open_mfdataset(mylist[2], decode_times=True, decode_cf=True).sel(time=slice(t1,t2)).chunk(chunks = chunks3)
DIC_data = xr.open_mfdataset(mylist[4], decode_times=True, decode_cf=True).sortby('time').sel(time=slice(t1,t2)).sortby('longitude',ascending=True).chunk(chunks = chunks)
Cstar_data = xr.open_mfdataset(mylist[3], decode_times=True, decode_cf=True).sortby('time').sel(time=slice(t1,t2)).sortby('longitude',ascending=True).chunk(chunks = chunks)
AW_data = xr.open_mfdataset(mylist[10], decode_times=True, decode_cf=True).isel(time=slice(60,420)).chunk(chunks = chunks3) # 1990 onwards
DF_data = xr.open_mfdataset(mylist[16], decode_times=True, decode_cf=True).sel(time=slice(t1,t2)).chunk(chunks = chunks3).transpose()
TA_data = xr.open_mfdataset(mylist[17], decode_times=True, decode_cf=True).sortby('time').sel(time=slice(t1,t2)).sortby('longitude',ascending=True).chunk(chunks = chunks)
NO3_data = xr.open_mfdataset(mylist[18], decode_times=True, decode_cf=True).sortby('time').sel(time=slice(t1,t2)).sortby('longitude',ascending=True).chunk(chunks = chunks).assign_coords(time=DIC_data.time)
JRA55_data = xr.open_mfdataset(mylist[11], decode_times=True, decode_cf=True).sel(time=slice(t1,t2)).sortby('latitude',ascending=True).chunk(chunks = chunks2)
#SF_data = xr.open_mfdataset(mylist[19], decode_times=True, decode_cf=True).sel(time=slice(t1,t2),wind='CCMP2').chunk(chunks={'product':1, 'time':360, 'lat':180, 'lon':360}) # SeaFlux pCO2 product compilation
SF_data = xr.open_mfdataset(mylist[19], decode_times=True, decode_cf=True).sel(time=slice(t1,t2)).mean('wind').chunk(chunks={'product':1, 'time':360, 'lat':180, 'lon':360}) # SeaFlux pCO2 product compilation
O2_data = xr.open_mfdataset(mylist[21], decode_times=True, decode_cf=True).sortby('time').sel(time=slice(t1,t2)).sortby('longitude',ascending=True).chunk(chunks = chunks).assign_coords(time=DIC_data.time)
#PO4_data = xr.open_mfdataset(mylist[22], decode_times=True, decode_cf=True).sortby('time').sel(time=slice(t1,t2)).sortby('longitude',ascending=True).chunk(chunks = chunks)


# In[8]:


# Rearrange Watson data to -180:180 longitude

lon_AW = np.hstack((np.arange(0.5,180,1),np.arange(-179.5,0,1)))
AW_data = AW_data.assign_coords(lon = lon_AW).sortby('lon',ascending=True)

# Rearrange JRA55 data to -180:180 longitude

lon_JRA55 = JRA55_data.longitude.values
lon_JRA55[0] = -180
lon_JRA55 = np.where(lon_JRA55 > 180,lon_JRA55 - 360,lon_JRA55)
JRA55_data = JRA55_data.assign_coords(longitude = lon_JRA55).sortby('longitude', ascending=True)

## Load the basin and seafloor masks

EN4_mask = xr.open_mfdataset(mylist[5], decode_times=True, chunks = chunks)
lon_mask = np.hstack((np.arange(1,181),np.arange(-179,1)))
EN4_mask = EN4_mask.assign_coords(lon=lon_mask).sortby('lon',ascending=True)
                     
# Get land mask

land_mask = xr.ones_like(DIC_data.temperature.isel(time=0)).rename(new_name_or_name_dict='land mask')
land_mask = (land_mask * ~np.isnan(DIC_data.temperature.isel(time=0))).astype('bool')


# In[9]:


## Convert EN4 data in ML prediction file from potential temperature and practical salinity to conservative temperature and absolute salinity

EN4_P = gsw.p_from_z(-DIC_data.depth,DIC_data.latitude)
EN4_SA = gsw.SA_from_SP(DIC_data.salinity,EN4_P,DIC_data.longitude,DIC_data.latitude)
EN4_CT = gsw.CT_from_pt(EN4_SA,DIC_data.temperature)


# In[12]:


## Define the window of time over which BSP bins are calculated - this is important when parallelising this process
window = 12
ti = int(sys.argv[1])
#ti = 10


# In[13]:


## Interpolate surface flux data onto interior grid (now flux conserving). NOTE: LS_data is not nan-filled so interpolation produces gaps

yr2sec = 3600*24*365.25
LON, LAT = np.meshgrid(DIC_data.longitude,DIC_data.latitude)

# Interpolate

JRA55_hflux_interp = xr.zeros_like(DIC_data.temperature.isel(depth=0)).rename(new_name_or_name_dict='heat flux')
JRA55_fwflux_interp = xr.zeros_like(DIC_data.temperature.isel(depth=0)).rename(new_name_or_name_dict='freshwater flux')

LS_CO2flux_interp = xr.zeros_like(DIC_data.temperature.isel(depth=0)).rename(new_name_or_name_dict='CO2 flux')
AW_CO2flux_interp = xr.zeros_like(DIC_data.temperature.isel(depth=0)).rename(new_name_or_name_dict='CO2 flux')
SF_CO2flux_interp = xr.zeros_like(DIC_data.temperature.isel(depth=0)).rename(new_name_or_name_dict='CO2 flux').expand_dims({'product':SF_data.product},axis=0).assign_coords(product=SF_data.product)

ds_out = xr.Dataset()
ds_out['longitude']=DIC_data.longitude
ds_out['latitude']=DIC_data.latitude

SF_data_subset = SF_data.fillna(0).isel(time=slice(ti*window, (ti+1)*window))
regridder = xe.Regridder(SF_data_subset.fgco2,ds_out,"conservative", periodic=True)
SF_CO2flux_interp[:,ti*window:(ti+1)*window,:,:] = regridder(-SF_data_subset.fgco2/yr2sec*10**3, keep_attrs=True).values # mmol-C/m^2/s

JRA55_data_subset = JRA55_data.fillna(0).isel(time=slice(ti*window, (ti+1)*window))
regridder = xe.Regridder(JRA55_data_subset.shf,ds_out,"conservative", periodic=True)
JRA55_hflux_interp[ti*window:(ti+1)*window,:,:] = regridder(-JRA55_data_subset.lhf - JRA55_data_subset.shf + JRA55_data_subset.dswrf + JRA55_data_subset.dlwrf - JRA55_data_subset.uswrf - JRA55_data_subset.ulwrf, keep_attrs=True).values # W/m^2
JRA55_fwflux_interp[ti*window:(ti+1)*window,:,:] = regridder(JRA55_data_subset.tpratsfc - JRA55_data_subset.evpsfc + JRA55_data_subset.rofsfc, keep_attrs=True).values

LS_data_subset = LS_data.isel(time=slice(ti*window, (ti+1)*window))
regridder = xe.Regridder(LS_data_subset.fgco2_raw,ds_out,"conservative", periodic=True)
LS_CO2flux_interp[ti*window:(ti+1)*window,:,:] = regridder(-LS_data_subset.fgco2_raw/yr2sec*10**3, keep_attrs=True).values # # mmol-C/m^2/s

AW_data_subset = AW_data.fillna(0).isel(time=slice(ti*window, (ti+1)*window))
regridder = xe.Regridder(AW_data_subset.fgco2,ds_out,"conservative", periodic=True)
AW_CO2flux_interp[ti*window:(ti+1)*window,:,:] = regridder(AW_data_subset.fgco2*10**3, keep_attrs=True).values # mmol-C/m^2/s   
    
JRA55_hflux_interp = JRA55_hflux_interp.chunk(chunks = chunks3) # W/m^2
JRA55_fwflux_interp = JRA55_fwflux_interp.chunk(chunks = chunks3) # mm/day
LS_CO2flux_interp = LS_CO2flux_interp.chunk(chunks = chunks3)
                 
# Fill in gaps in Watson air-sea flux product

AW_CO2flux_interp = xr.where(LAT>=65, LS_CO2flux_interp, AW_CO2flux_interp)


# In[14]:


## Specify tracers

S0=35
rho0=1000 # Density of freshwater
R_CN = 6.6
R_CO = -0.69
R_NO = -0.09
R_CP = 106
R_NP = 16

#cstar = Cstar_data.CstarN_pred_mean
#dic = DIC_data.DIC_pred_mean

#cstar = Cstar_data.CstarN
#cstar = DIC_data.DIC_pred_mean - R_CN * NO3_data.nitrate_pred_mean - 0.5 * (TA_data.TAlk_pred_mean + NO3_data.nitrate_pred_mean)
#cstar = DIC_data.DIC_pred_mean - R_CO * O2_data.oxygen_pred_mean - 0.5 * (TA_data.TAlk_pred_mean + R_NO * O2_data.oxygen_pred_mean)
#dic = DIC_data.DIC_pred_mean

if member == 'ens':

    dic = DIC_data.DIC_pred_mean
    if cstardef=='nitrate':
        cstar = DIC_data.DIC_pred_mean - R_CN * NO3_data.nitrate_pred_mean - 0.5 * (TA_data.TAlk_pred_mean + NO3_data.nitrate_pred_mean)
    elif cstardef=='oxygen':
        cstar = DIC_data.DIC_pred_mean - R_CO * O2_data.oxygen_pred_mean - 0.5 * (TA_data.TAlk_pred_mean + R_NO * O2_data.oxygen_pred_mean)
    elif cstardef=='phosphate':
        cstar = DIC_data.DIC_pred_mean - R_CP * PO4_data.phosphate_pred_mean - 0.5 * (TA_data.TAlk_pred_mean + R_NP*PO4_data.phosphate_pred_mean)
    elif cstardef=='aou':
        O2sat = gsw.O2sol_SP_pt(DIC_data.salinity,DIC_data.temperature)
        AOU = O2sat - O2_data.oxygen_pred_mean
        #cstar = DIC_data.DIC_pred_mean + R_CO * AOU - 0.5 * (TA_data.TAlk_pred_mean - R_NO * AOU)
        cstar = DIC_data.DIC_pred_mean + R_CO * AOU - 0.5 * (TA_data.TAlk_pred_mean + NO3_data.nitrate_pred_mean)

else:
    dic = DIC_data.DIC
    if cstardef=='nitrate':
        cstar = DIC_data.DIC - R_CN * NO3_data.nitrate.values - 0.5 * (TA_data.TAlk + NO3_data.nitrate.values)
    elif cstardef=='oxygen':
        cstar = DIC_data.DIC - R_CO * O2_data.oxygen.values - 0.5 * (TA_data.TAlk + R_NO * O2_data.oxygen.values)
    elif cstardef=='phosphate':
        cstar = DIC_data.DIC - R_CP * PO4_data.phosphate - 0.5 * (TA_data.TAlk + R_NP*PO4)
    elif cstardef=='aou':
        O2sat = gsw.O2sol_SP_pt(DIC_data.salinity,DIC_data.temperature)
        AOU = O2sat - O2_data.oxygen.values
        cstar = DIC_data.DIC + R_CO * AOU - 0.5 * (TA_data.TAlk + NO3_data.nitrate.values)


# In[15]:


## Define grid area, volume and depth

area = xr.open_mfdataset(mylist[7]).areas*land_mask.isel(depth=0)
depth = DIC_data.depth.values

dArea_3D = area.expand_dims({'depth':depth},axis=1).assign_coords(depth=DIC_data.depth)
dArea_4D = area.expand_dims({'depth':depth},axis=1).assign_coords(depth=DIC_data.depth).expand_dims({'time':DIC_data.time},axis=0).assign_coords(time=DIC_data.time)
dSArea_3D = dArea_3D.copy(deep=True)

EN4_dz = (EN4_data.depth_bnds.isel(time=0)[:,1]-EN4_data.depth_bnds.isel(time=0)[:,0])

volume = dArea_3D*EN4_dz*land_mask
vol500 = volume.copy(deep=True)
depth_ind = np.argmin(depth<2000) # Set 'v500' depth

# Defining 9 basins
Basins = EN4_mask.Basins.values


# In[16]:


## Specify T,S
#EN4_T = DIC_data.temperature*(~land_mask-1)/(~land_mask-1)
#EN4_S = DIC_data.salinity*(~land_mask-1)/(~land_mask-1)
EN4_T = EN4_CT*(~land_mask-1)/(~land_mask-1)
EN4_S = EN4_SA*(~land_mask-1)/(~land_mask-1)


## Specify fluxes
JRA55_hf = JRA55_hflux_interp*area.values # W
JRA55_wfo = JRA55_fwflux_interp/(24*3600)/10**3*area.values*rho0 # kg/s
LS_Cflux = LS_CO2flux_interp*area # mmol-C/s
AW_Cflux = AW_CO2flux_interp*area # mmol-C/s
SF_Cflux = SF_CO2flux_interp*area # mmol-C/s


# In[17]:


## Turn fluxes into 3D files
JRA55_hf_3D = JRA55_hf.expand_dims({'depth':depth},axis=0).assign_coords(depth=EN4_data.depth).chunk("auto") # units: W
JRA55_wfo_3D = JRA55_wfo.expand_dims({'depth':depth},axis=0).assign_coords(depth=EN4_data.depth).chunk("auto") # units: kg/s
LS_Cflux_3D = LS_Cflux.expand_dims({'depth':depth},axis=0).assign_coords(depth=DIC_data.depth).chunk("auto") # units: mmol-C/s
AW_Cflux_3D = AW_Cflux.expand_dims({'depth':depth},axis=0).assign_coords(depth=DIC_data.depth).chunk("auto") # units: mmol-C/s
SF_Cflux_3D = SF_Cflux.expand_dims({'depth':depth},axis=0).assign_coords(depth=DIC_data.depth).chunk("auto") # units: mmol-C/s

## Set fluxes, area to be zero in the interior
## We also set our volume array that will be used for binning to be zero below a given depth

JRA55_hf_3D[1:,:,:,:] = 0
JRA55_wfo_3D[1:,:,:,:] = 0
LS_Cflux_3D[1:,:,:,:] = 0
AW_Cflux_3D[1:,:,:,:] = 0
SF_Cflux_3D[1:,:,:,:,:] = 0
dSArea_3D[:,1:,:] = 0
vol500[:,depth_ind:,:] = 0

## Flatten variables of interest into 2D arrays (time x flattened spatial dimensions)

volcello_flattened = (volume.stack(z=("latitude","longitude","depth"))).chunk("auto")
vol_500_flattened = (vol500.stack(z=("latitude","longitude","depth"))).chunk("auto")
bigthetao_flattened = (EN4_T.stack(z=("latitude","longitude","depth"))).chunk("auto")
so_flattened = (EN4_S.stack(z=("latitude","longitude","depth"))).chunk("auto")
cstar_flattened = (cstar.stack(z=("latitude","longitude","depth"))).chunk("auto")
dic_flattened = (dic.stack(z=("latitude","longitude","depth"))).chunk("auto")
areacello_flattened = (dSArea_3D.stack(z=("latitude","longitude","depth"))).chunk("auto")
mask_flattened = (EN4_mask.mask_EN4.stack(z=("lat","lon","depth"))).chunk("auto")
hfds_flattened = JRA55_hf_3D.stack(z=("latitude","longitude","depth")).chunk("auto")
wfo_flattened = JRA55_wfo_3D.stack(z=("latitude","longitude","depth")).chunk("auto")
Cflux1_flattened = LS_Cflux_3D.stack(z=("latitude","longitude","depth")).chunk("auto")
Cflux2_flattened = AW_Cflux_3D.stack(z=("latitude","longitude","depth")).chunk("auto")
CfluxSF_flattened = SF_Cflux_3D.stack(z=("latitude","longitude","depth")).chunk("auto")

## Shorten their names (not necessary but makes for easier code readability)
BA = mask_flattened
V = volcello_flattened
V500 = vol_500_flattened
S = ((so_flattened))
T = ((bigthetao_flattened))
C1 = dic_flattened
C2 = cstar_flattened
A = (areacello_flattened)
HF = hfds_flattened
WFO = wfo_flattened
CF1 = Cflux1_flattened # Watson
CF2 = Cflux2_flattened # Landschutzer
# Fay et al pCO2 product harmonisation
CF3a = CfluxSF_flattened[0,:,:]
CF3b = CfluxSF_flattened[1,:,:]
CF3c = CfluxSF_flattened[2,:,:]
CF3d = CfluxSF_flattened[3,:,:]
CF3e = CfluxSF_flattened[4,:,:]
CF3f = CfluxSF_flattened[5,:,:]


# In[18]:


## Define the number of BSP bins to output, where number = 2**tree_depth
tree_depth = 6
nbasins = len(Basins)
# Note: if 2**depth approaches the sample size, the code will not work as an equal volume constraint will become impossible!

## Create empty arrays that will be filled by the BSP-ised bins
## Array sizes are (Basin, time, BSP depth) other than for the bin corners which are
## (Basin, time, BSP depth, 4)

partitions_hist = np.zeros((nbasins, window,2**tree_depth, 4))
T_mean_hist = np.zeros((nbasins, window,2**tree_depth))
S_mean_hist = np.zeros((nbasins, window,2**tree_depth))
C1_mean_hist = np.zeros((nbasins, window,2**tree_depth))
C2_mean_hist = np.zeros((nbasins, window,2**tree_depth))
V_sum_hist = np.zeros((nbasins, window,2**tree_depth))
V500_sum_hist = np.zeros((nbasins, window,2**tree_depth))
A_sum_hist = np.zeros((nbasins, window,2**tree_depth))
hfds_sum_hist = np.zeros((nbasins, window,2**tree_depth))
wfo_sum_hist = np.zeros((nbasins, window,2**tree_depth))
Cflux1_sum_hist = np.zeros((nbasins, window,2**tree_depth))
Cflux2_sum_hist = np.zeros((nbasins, window,2**tree_depth))
Cflux3a_sum_hist = np.zeros((nbasins, window,2**tree_depth))
Cflux3b_sum_hist = np.zeros((nbasins, window,2**tree_depth))
Cflux3c_sum_hist = np.zeros((nbasins, window,2**tree_depth))
Cflux3d_sum_hist = np.zeros((nbasins, window,2**tree_depth))
Cflux3e_sum_hist = np.zeros((nbasins, window,2**tree_depth))
Cflux3f_sum_hist = np.zeros((nbasins, window,2**tree_depth))


# In[ ]:


## Run a loop over times and basins

'''
1) calc: This function calculates the BSP bins for any 2D distribution. We input the x,y, and v parameters, as well as the 
tree depth, first axis to split orthogonal to, and any diagnostics we want to output. 
We are able to output summed variables for each bin, and meaned variables for each bin.
The weight over which the mean is calculated can also be different to the distribution weight, v.

2) split: The `calc` function outputs a large nested list, which needs to be split into the constituent diagnostics of interest. 
Due to the recursive nature of the `calc` function, this splitting must be accomplished in a second function, `split`.
The output of the `split` function is a dictionary with BSP box boundaries, summed variables and meaned variables. 

3) draw: The `draw` function allows us to visualise the BSP boundaries on top of the original distribution. 
'''

time_array = np.zeros(window)

for i in tqdm(range(ti*window, (ti+1)*window)):
    time_array[int(i-ti*window)] = i
    for j in tqdm(range(Basins.size)):

        # Get a single timestep as numpy, not dask
        ## The x and y axes
        x = S[i,:].values
        y = T[i,:].values
        ## Any tracers to find the weighted mean of
        z1 = C1[i,:].values
        z2 = C2[i,:].values
        ## The 2D distribution to calculate bins on
        v = V500.values
        
        ## Summed variables to output for each bin
        c = V.values*BA[j,:].values
        a = A.values*BA[j,:].values
        u = HF[i,:].values*BA[j,:].values
        w = WFO[i,:].values*BA[j,:].values      
        q1 = CF1[i,:].values*BA[j,:].values
        q2 = CF2[i,:].values*BA[j,:].values
        q3a = CF3a[i,:].values*BA[j,:].values
        q3b = CF3b[i,:].values*BA[j,:].values
        q3c = CF3c[i,:].values*BA[j,:].values
        q3d = CF3d[i,:].values*BA[j,:].values
        q3e = CF3e[i,:].values*BA[j,:].values
        q3f = CF3f[i,:].values*BA[j,:].values
                   
        
        # Clean out NAN values
        idx = np.isfinite(y)
        x = x[idx]
        y = y[idx]
        z1 = z1[idx]
        z2 = z2[idx]
        v = v[idx]      
        c = c[idx]
        a = a[idx]
        u = u[idx]
        w = w[idx]
        q1 = q1[idx]
        q2 = q2[idx]
        q3a = q3a[idx]
        q3b = q3b[idx]
        q3c = q3c[idx]
        q3d = q3d[idx]
        q3e = q3e[idx]
        q3f = q3f[idx]


        ## Calculate the BSP bins
        BSP_out = BSP.calc(x,y,v, depth=tree_depth, axis=1, mean=[x,y,z1,z2],sum=[v,c,a,u,w,q1,q2,q3a,q3b,q3c,q3d,q3e,q3f],weight=c)
        # Split the output into constituent diagnostics
        vals = BSP.split(BSP_out, depth=tree_depth)
        
        ## Draw the BSP bins onto original grid
        
        if i == ti*window and j==0:

            fig = plt.figure(figsize=(10,4))
            ax = fig.add_subplot(1,2,1)
            BSP.draw(x,y,np.log10(v),vals['bounding_box'],'grey', depth=tree_depth)
            cbar = plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            cbar.set_label('Distribution weight')
            
            ax = fig.add_subplot(1,2,2)
            BSP.draw(vals['meaned_vals'][:,0],vals['meaned_vals'][:,1],vals['summed_vals'][:,0],vals['bounding_box'],'red', depth=tree_depth, cmap=plt.cm.viridis)
            cbar = plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            cbar.set_label('Summed weight')
        
            plt.show()
            plt.savefig('BSP_bins.png', bbox_inches='tight', dpi=300)
            
        ## Allocate the BSP outputs into arrays
            
        partitions_hist[j,int(i-ti*window),:,:] = vals['bounding_box']
        S_mean_hist[j,int(i-ti*window),:] = vals['meaned_vals'][:,0]
        T_mean_hist[j,int(i-ti*window),:] = vals['meaned_vals'][:,1]
        C1_mean_hist[j,int(i-ti*window),:] = vals['meaned_vals'][:,2]
        C2_mean_hist[j,int(i-ti*window),:] = vals['meaned_vals'][:,3]
        V500_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,0]
        V_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,1]
        A_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,2]
        hfds_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,3]
        wfo_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,4]
        Cflux1_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,5]
        Cflux2_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,6]
        Cflux3a_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,7]
        Cflux3b_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,8]
        Cflux3c_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,9]
        Cflux3d_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,10]
        Cflux3e_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,11]
        Cflux3f_sum_hist[j,int(i-ti*window),:] = vals['summed_vals'][:,12]
        



# In[1]:


## We redefine each outputted numpy array as an xarray DataArray with the goal of saving it as a netCDF file

da_partitions_hist = xr.DataArray(data = partitions_hist, dims = ["Basin","Time", "Depth", "Coords"], 
                           coords=dict(Basin = Basins, Time = time_array, Depth= np.arange(2**tree_depth), Coords = np.arange(4)),
                        attrs=dict(description="[x0,y0,xmax,ymax] bounds of BSP framework", variable_id="Partitions"))

da_S_mean_hist = xr.DataArray(data = S_mean_hist, dims = ["Basin", "Time", "Depth"], 
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Mean Salinity", units="g/kg", variable_id="S"))
da_T_mean_hist = xr.DataArray(data = T_mean_hist, dims = ["Basin", "Time","Depth"], 
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Mean Temperature", units="K", variable_id="T"))
da_C1_mean_hist = xr.DataArray(data = C1_mean_hist, dims = ["Basin", "Time","Depth"], 
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Mean C_star", units="mmol-C/m3", variable_id="C_star"))
da_C2_mean_hist = xr.DataArray(data = C2_mean_hist, dims = ["Basin", "Time","Depth"], 
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Mean DIC", units="mmol-C/m3", variable_id="DIC"))

da_V_sum_hist = xr.DataArray(data = V_sum_hist, dims = ["Basin", "Time","Depth"], 
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Total Volume", units="m^3", variable_id="Basin V_sum"))
da_A_sum_hist = xr.DataArray(data = A_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Total Area", units="m^2", variable_id="Basin A_sum"))

da_hfds_sum_hist = xr.DataArray(data = hfds_sum_hist, dims = ["Basin", "Time","Depth"], 
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Heat Flux", units="W", variable_id="hfds"))
da_wfo_sum_hist = xr.DataArray(data = wfo_sum_hist, dims = ["Basin", "Time","Depth"], 
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Freshwater Flux", units="kg/s", variable_id="wfo"))

da_Cflux1_sum_hist = xr.DataArray(data = Cflux1_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="LS Carbon Flux", units="mmol-C/s", variable_id="Cflux1"))
da_Cflux2_sum_hist = xr.DataArray(data = Cflux2_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="AW Carbon Flux", units="mmol-C/s", variable_id="Cflux2"))

da_Cflux3a_sum_hist = xr.DataArray(data = Cflux3a_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="JENA Carbon Flux", units="mmol-C/s", variable_id="Cflux3a"))
da_Cflux3b_sum_hist = xr.DataArray(data = Cflux3b_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="SOMFFN Carbon Flux", units="mmol-C/s", variable_id="Cflux3b"))
da_Cflux3c_sum_hist = xr.DataArray(data = Cflux3c_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="CMEMS Carbon Flux", units="mmol-C/s", variable_id="Cflux3c"))
da_Cflux3d_sum_hist = xr.DataArray(data = Cflux3d_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="CSIR Carbon Flux", units="mmol-C/s", variable_id="Cflux3d"))
da_Cflux3e_sum_hist = xr.DataArray(data = Cflux3e_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="JMA Carbon Flux", units="mmol-C/s", variable_id="Cflux3e"))
da_Cflux3f_sum_hist = xr.DataArray(data = Cflux3f_sum_hist, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="NIES Carbon Flux", units="mmol-C/s", variable_id="Cflux3f"))


## Input all xarray DataArrays into a DataSet

ds_BSP = xr.Dataset()
ds_BSP['Partitions_hist'] = da_partitions_hist
ds_BSP['T_mean_hist'] = da_T_mean_hist
ds_BSP['S_mean_hist'] = da_S_mean_hist
ds_BSP['DIC_mean_hist'] = da_C1_mean_hist
ds_BSP['Cstar_mean_hist'] = da_C2_mean_hist
ds_BSP['V_sum_hist'] = da_V_sum_hist
ds_BSP['A_sum_hist'] = da_A_sum_hist
ds_BSP['hfds_sum_hist'] = da_hfds_sum_hist
ds_BSP['wfo_sum_hist'] = da_wfo_sum_hist
ds_BSP['Cflux1_sum_hist'] = da_Cflux1_sum_hist
ds_BSP['Cflux2_sum_hist'] = da_Cflux2_sum_hist
ds_BSP['Cflux3a_sum_hist'] = da_Cflux3a_sum_hist
ds_BSP['Cflux3b_sum_hist'] = da_Cflux3b_sum_hist
ds_BSP['Cflux3c_sum_hist'] = da_Cflux3c_sum_hist
ds_BSP['Cflux3d_sum_hist'] = da_Cflux3d_sum_hist
ds_BSP['Cflux3e_sum_hist'] = da_Cflux3e_sum_hist
ds_BSP['Cflux3f_sum_hist'] = da_Cflux3f_sum_hist

if member=='ens':
    ds_BSP.to_netcdf(mylist[8] + 'JRA55/' + cstardef + '_Cstar_components/BSP_JRA55_NN_%i_%i.nc' %(ti*window, (ti+1)*window-1))
else:
    ds_BSP.to_netcdf(mylist[8] + 'JRA55/' + cstardef + '_Cstar_components/member_' + str(member) + '/BSP_JRA55_NN_%i_%i.nc' %(ti*window, (ti+1)*window-1))


