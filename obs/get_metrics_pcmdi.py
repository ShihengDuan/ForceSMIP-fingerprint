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

variable = 'tos'

if variable == 'tos':
    cmip_var = 'tos'
    eof_start = 1950
    start_year = 1950
    end_year = 2021
elif variable=='monmaxpr':
    cmip_var = 'pr'
    eof_start = 1979
    start_year = 1980
    end_year = 2018
elif variable=='pr':
    cmip_var = 'pr'
    eof_start = 1979
    start_year = 1983
    end_year = 2018
if variable=='tos':
    mask = xr.open_dataset('../maskland.nc')
    missing_xa = xr.where(np.isnan(mask.tos.isel(time=0)), np.nan, 1)
else:
    mask = xr.open_dataset('../nomask.nc')
    missing_xa = xr.where(np.isnan(mask.tas.isel(time=0)), np.nan, 1)

def calculate_metrics(solver_list, obs, unforced_list, pc_series, month=False):
    pc1 = pc_series.isel(mode=0).sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
    pc_max = pc1.max().data
    pc_min = pc1.min().data
    print(pc_max, ' ', pc_min)
    if month:
        pseudo_pc_month = []
        for month in range(1, 13):
            # normalize 
            solver = solver_list[month-1]
            ds_in = obs.sel(time=obs.time.dt.month==month)*missing_xa
            pseudo_pc = solver.projectField(ds_in-ds_in.mean(dim='time')).isel(mode=0)
            pseudo_pc_month.append(pseudo_pc)
        pseudo_pc = xr.concat(pseudo_pc_month, dim='time')
        pseudo_pc = pseudo_pc.sortby('time')
        pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min) # 0 to 1 
        pseudo_pc = pseudo_pc*2-1 # -1 to 1 
    else:
        # normalize pc series. 
        ds_in = obs*missing_xa
        pseudo_pc = solver_list[0].projectField(ds_in-ds_in.mean(dim='time')).isel(mode=0)
        pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min)
        pseudo_pc = pseudo_pc*2-1
    print(pseudo_pc.shape, pseudo_pc.max().data, ' ', pseudo_pc.min().data)
    # get noise and signal
    noise_pcs = []
    for unforced in unforced_list:
        unforced = unforced[cmip_var].sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
        member = unforced.shape[0]
        for m in range(member):
            ds_in = unforced.isel(member=m)*missing_xa
            ds_in = ds_in.transpose('time', 'lon', 'lat')
            if month:
                noise_month = []
                for month in range(1, 13):
                    solver = solver_list[month-1]
                    ds_in_month = ds_in.sel(time=ds_in.time.dt.month==month)
                    # print(month, ' ', np.sum(np.isnan(ds_in)).data)
                    psd = solver.projectField(ds_in_month-ds_in_month.mean(dim='time'))
                    noise_month.append(psd)
                noise_month = xr.concat(noise_month, dim='time')
                noise_month = noise_month.sortby('time')
                noise_month = noise_month.isel(mode=0)
                noise_month = (noise_month-pc_min)/(pc_max-pc_min)
                noise_month = noise_month*2-1
                noise_pcs.append(noise_month)
            else:
                psd = solver_list[0].projectField(ds_in-ds_in.mean(dim='time'))
                psd = psd.isel(mode=0)
                psd = (psd-pc_max)/(pc_max-pc_min)
                psd = psd*2-1
                noise_pcs.append(psd)
    # get noise strength
    timescales = np.arange(12, (end_year-start_year+1)*12)
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
    pc1 = pc_series.isel(mode=0).sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
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


print(variable, ' ', start_year, ' ', end_year, ' ', eof_start)

if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/PCMDI/'):
    os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/PCMDI/')
p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/PCMDI/'

pcmdi = xr.open_dataset('/p/lustre3/shiduan/sst/PCMDI/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-8_gn_195001-202112_2p5x2p5_masked.nc')
pcmdi = pcmdi["tos"].transpose('time', 'lon', 'lat')
pcmdi = pcmdi.sel(time=slice('1950-01-01', '2022-01-01'))
print(pcmdi.shape, ' ', pcmdi.time.data[-1])
pcmdi = pcmdi.fillna(0)
pcmdi = pcmdi*missing_xa

with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-True-joint-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_unforced = record['solver']
unforced_list_month_unforced = record['unforced_list']
pc_month_unforced = record['pc']
unforced_std = record['unforced_std'] # for obs normalization

pcmdi_anomaly = pcmdi.groupby(pcmdi.time.dt.month)-pcmdi.groupby(pcmdi.time.dt.month).mean(dim='time')
pcmdi_unforced = pcmdi_anomaly.groupby(pcmdi_anomaly.time.dt.month)/unforced_std['tos']
pcmdi_stand = pcmdi_anomaly.groupby(pcmdi_anomaly.time.dt.month)/pcmdi_anomaly.groupby(pcmdi_anomaly.time.dt.month).std(dim='time')
pcmdi_stand = xr.where(np.isfinite(pcmdi_stand), pcmdi_stand, 0)
pcmdi_stand = pcmdi_stand.fillna(0)
pcmdi_stand = pcmdi_stand * missing_xa
# Month Unforced
pc_all_unforced = xr.concat(pc_month_unforced, dim='time')
pc_all_unforced = pc_all_unforced.sortby('time')
results_month_unforced = calculate_metrics(obs=pcmdi_unforced,
    solver_list=solver_list_month_unforced, 
    unforced_list=unforced_list_month_unforced, pc_series=pc_all_unforced, month=True)
path = p+variable+'-metrics-stand-True-month-True-unforced-True-joint-False'
with open(path, 'wb') as pfile:
    pickle.dump(results_month_unforced, pfile)

# Month standard
with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-False-joint-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_stand = record['solver']
unforced_list_month_stand = record['unforced_list']
pc_month_stand = record['pc']
pc_all_stand = xr.concat(pc_month_stand, dim='time')
pc_all_stand = pc_all_stand.sortby('time')

results_month_stand = calculate_metrics(obs=pcmdi_stand,
    solver_list=solver_list_month_stand, 
    unforced_list=unforced_list_month_stand, pc_series=pc_all_stand, month=True)
path = p+variable+'-metrics-stand-True-month-True-unforced-False-joint-False'
with open(path, 'wb') as pfile:
    pickle.dump(results_month_stand, pfile)

# Month
with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-True-unforced-False-joint-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month = record['solver']
unforced_list_month = record['unforced_list']
pc_month = record['pc']
pc_all = xr.concat(pc_month, dim='time')
pc_all = pc_all.sortby('time')
results_month = calculate_metrics(obs=pcmdi_anomaly,
    solver_list=solver_list_month, unforced_list=unforced_list_month, pc_series=pc_all, month=True)
path = p+variable+'-metrics-stand-False-month-True-unforced-False-joint-False'
with open(path, 'wb') as pfile:
    pickle.dump(results_month, pfile)

# All
with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-False-unforced-False-joint-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver = record['solver']
unforced_list = record['unforced_list']
pc_list = record['pc']
results_raw= calculate_metrics(obs=pcmdi_anomaly,
    solver_list=solver, 
    unforced_list=unforced_list, pc_series=pc_list[0], month=False)
path = p+variable+'-metrics-stand-False-month-False-unforced-False-joint-False'
with open(path, 'wb') as pfile:
    pickle.dump(results_raw, pfile)

# Standard
with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-False-joint-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_stand = record['solver']
unforced_list_stand = record['unforced_list']
pc_list_stand = record['pc']
results_stand = calculate_metrics(obs=pcmdi_stand,
    solver_list=solver_stand, 
    unforced_list=unforced_list_stand, pc_series=pc_list_stand[0], month=False)
path = p+variable+'-metrics-stand-True-month-False-unforced-False-joint-False'
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)


# Unforced
with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-True-joint-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_unforced = record['solver']
unforced_list_unforced = record['unforced_list']
pc_unforced = record['pc']
results_unforced = calculate_metrics(obs=pcmdi_unforced,
    solver_list=solver_list_unforced, 
    unforced_list=unforced_list_unforced, pc_series=pc_unforced[0], month=False)
path = p+variable+'-metrics-stand-True-month-False-unforced-True-joint-False'
with open(path, 'wb') as pfile:
    pickle.dump(results_unforced, pfile)
