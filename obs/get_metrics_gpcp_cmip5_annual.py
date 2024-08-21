import time as clocktime
import glob
import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle

# principal component analysis
from eofs.xarray import Eof
from utils import get_slope
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eof', type=int, default=1)
    args = vars(parser.parse_args())
    return args

args = get_args()
n_mode = args['eof']

variable = 'pr'

if variable == 'tos':
    cmip_var = 'tos'
    eof_start = 1950
    start_year = 1950
    end_year = 2019
elif variable=='monmaxpr':
    cmip_var = 'pr'
    eof_start = 1979
    start_year = 1980
    end_year = 2020
elif variable=='pr':
    cmip_var = 'pr'
    eof_start = 1979
    start_year = 1983
    end_year = 2020

print(variable, ' ', start_year, ' ', end_year, ' ', eof_start)

with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022_annual_cmip5/'+variable+'-record-stand-False-unforced-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver = record['solver']
pc_list = record['pc']

with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022_annual_cmip5/'+variable+'-record-stand-True-unforced-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_stand = record['solver']
pc_list_stand = record['pc']



if variable=='tos':
    mask = xr.open_dataset('../maskland.nc')
    missing_xa = xr.where(np.isnan(mask.tos.isel(time=0)), np.nan, 1)
else:
    mask = xr.open_dataset('../nomask.nc')
    missing_xa = xr.where(np.isnan(mask.tas.isel(time=0)), np.nan, 1)

gpcp = []
for i in range(1983, 2021):
    data = xc.open_dataset('/p/lustre3/shiduan/GPCP/regrid/'+str(i)+'.nc')
    gpcp.append(data)
gpcp = xr.concat(gpcp, dim='time')
print(gpcp.time)
gpcp = gpcp.bounds.add_missing_bounds(axes=['T'])
gpcp = gpcp.temporal.group_average('__xarray_dataarray_variable__', freq='year')
gpcp = gpcp["__xarray_dataarray_variable__"].transpose('time', 'lon', 'lat')
gpcp = gpcp.fillna(0)

gpcp_anomaly = gpcp-gpcp.mean(dim='time')
gpcp_stand = gpcp_anomaly/gpcp_anomaly.std(dim='time')

# piControl:
path = '/p/lustre3/shiduan/ForceSMIP/CMIP5/pr/piControl/'
files = glob.glob(path+"*.nc")

picontrol = []
picontrol_std = []
for file in files:
    ds = xc.open_dataset(file)
    ds = ds.temporal.group_average('pr', freq='year')
    ds = ds['pr']*86400
    ds = ds-ds.mean(dim='time')
    if np.sum(np.isnan(ds))==0:
        picontrol.append(ds)
        ds_std = ds/ds.std(dim='time')
        picontrol_std.append(ds_std)
    else:
        print(file, ' NAN')
print(len(picontrol), ' ', len(picontrol_std))

def calculate_metrics(solver_list, obs, unforced_list, pc_series, n_mode=1):
    pc1 = pc_series.isel(mode=n_mode-1)
    m, b = np.polyfit(np.arange(pc1.shape[0]), pc1, deg=1)
    if m<0:
        pc1 = -pc1
        reverse=True
        print('reverse')
    else:
        reverse=False
    pc1 = pc1.sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
    timescales = np.arange(5, (end_year+1-start_year))
    pc_max = pc1.max().data
    pc_min = pc1.min().data
    print(pc_max, ' ', pc_min)
    # normalize cmip ensemble pcs. 
    
    # normalize pc series. 
    ds_in = obs
    pseudo_pc = solver_list[0].projectField(ds_in-ds_in.mean(dim='time')).isel(mode=n_mode-1)
    pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min)
    pseudo_pc = pseudo_pc*2-1
    if reverse:
        pseudo_pc = -pseudo_pc
    print(pseudo_pc.shape, pseudo_pc.max().data, ' ', pseudo_pc.min().data)
    # get noise and signal
    noise_pcs = []
    for unforced in unforced_list:
        ds_in = unforced*missing_xa
        ds_in = ds_in.transpose('time', 'lon', 'lat')
        psd = solver_list[0].projectField(ds_in-ds_in.mean(dim='time'))
        psd = psd.isel(mode=n_mode-1)
        psd = (psd-pc_max)/(pc_max-pc_min)
        psd = psd*2-1
        if reverse:
            psd = -psd
        noise_pcs.append(psd)
        # print('psd: ', psd)
    print(len(noise_pcs), ' ', len(unforced_list))

    # get noise strength
    timescales = np.arange(5, (end_year-start_year+1))
    # initialize noise time series dictionary
    noise = {}
    # loop over timescales
    for nyears in timescales:
        # initialize list of noise trends
        it_noise = []
        # loop over models
        for ts in noise_pcs:
            # time = np.array([t.year for t in ts.time.values])
            time = np.arange(len(ts))
            # get the number of non-overlapping time windows
            nsamples = int(np.floor(len(ts) / nyears))
            # loop over the time windows (trend time periods)
            for ns in range(nsamples):
                # get time interval indices
                sample_inds = np.arange(ns*nyears, ns*nyears+nyears)
                # subset time series
                ts_sub = ts.isel(time=sample_inds)
                # compute trend
                m, b = np.polyfit(time[sample_inds], ts_sub, 1)
                # add trend to list
                it_noise.append(m)
        # add list to noise dictionary
        noise[nyears] = it_noise
    # get signal and obs strength
    signal = {}
    obs_signal = {}
    obs_se = []
    # model 
    # pc1 = pc_series.isel(mode=n_mode-1).sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
    pc1_norm = (pc1-pc_min)/(pc_max-pc_min)
    pc1_norm = pc1_norm*2-1
    # loop over each timescale to compute the PC trend
    for nyears in timescales:
        # get indices for time scale
        sample_inds = np.arange(0, nyears)
        # compute the trend
        time = np.arange(len(pc1_norm))
        m, b = np.polyfit(time[sample_inds], pc1_norm.isel(time=sample_inds), 1)
        # store the trend (signal)
        signal[nyears] = m
        
    # observation
    pc1 = pseudo_pc
    obs_time = np.arange(len(pc1))
    for nyears in timescales:
        # get indices for time scale
        sample_inds = np.arange(0, nyears)
        # compute the trend
        # m, b = np.polyfit(obs_time[sample_inds], pc1.isel(time=sample_inds), 1)
        m, e = get_slope(time[sample_inds], pc1.isel(time=sample_inds))
        # store the trend (signal)
        obs_signal[nyears] = m
        obs_se.append(e)
    sn = []
    s_list = []
    n_list = []
    s_obs_list = []
    sn_obs = []
    for ts in timescales:
        # compute s/n ratio from pre-computed
        # signal/noise values
        s = signal[ts]
        n = np.std(noise[ts])
        sn.append(s/n)
        s_list.append(s)
        n_list.append(n)
        s = obs_signal[ts]
        s_obs_list.append(s)
        sn_obs.append(s/n)
    results = {
        'sn': sn, 'sn_obs': sn_obs, 
        'signal':signal, 'noise':noise, 
        's_list':s_list, 'n_list':n_list, 
        'obs_pc':pseudo_pc,
        's_obs_list': s_obs_list, 
        'pc':pc1_norm,
        'pc_max': pc_max, 'pc_min': pc_min,
        'obs_se': obs_se, 
    }
    return results


results_stand = calculate_metrics(obs=gpcp_stand,
    solver_list=solver_stand, 
    unforced_list=picontrol_std, pc_series=pc_list_stand[0], n_mode=n_mode)


results_raw= calculate_metrics(obs=gpcp_anomaly,
    solver_list=solver, 
    unforced_list=picontrol, pc_series=pc_list[0], n_mode=n_mode)

if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022_annual_cmip5/GPCP/'):
    os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022_annual_cmip5/GPCP/')
p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022_annual_cmip5/GPCP/'


path = p+variable+'-metrics-stand-False-month-False-unforced-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_raw, pfile)

path = p+variable+'-metrics-stand-True-month-False-unforced-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)

print(results_raw['sn_obs'])

print(results_stand['sn_obs'])
