import time as clocktime
import glob
import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle
import sys

# principal component analysis
from eofs.xarray import Eof
from utils import get_slope
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pacific', type=int, default=0)
    parser.add_argument('--atlantic', type=int, default=0)
    parser.add_argument('--late', type=int, default=0)
    args = vars(parser.parse_args())
    return args

args = get_args()
pacific = args['pacific']
pacific = pacific>0
atlantic = args['atlantic']
atlantic = atlantic>0
if pacific and atlantic:
    sys.exit("NO Such combination")
late = args['late']
late = late>0
variable = 'tos'

if variable == 'tos':
    cmip_var = 'tos'
    eof_start = 1950
    start_year = 1950
    end_year = 2021
    if late:
        eof_start = 1979
        start_year = 1979
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
                    solver = solver_list_month[month-1]
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

if pacific:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-True-joint-False-Pacific'
elif atlantic:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-True-joint-False-Atlantic'
else:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-True-joint-False'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_unforced = record['solver']
unforced_list_month_unforced = record['unforced_list']
pc_month_unforced = record['pc']
unforced_std = record['unforced_std'] # for obs normalization
del record

if pacific:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-True-joint-False-Pacific'
elif atlantic:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-True-joint-False-Atlantic'
else:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-True-joint-False'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_unforced = record['solver']
unforced_list_unforced = record['unforced_list']
pc_unforced = record['pc']
del record

if pacific:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-True-unforced-False-joint-False-Pacific'
elif atlantic:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-True-unforced-False-joint-False-Atlantic'
else:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-True-unforced-False-joint-False'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month = record['solver']
unforced_list_month = record['unforced_list']
pc_month = record['pc']
del record

if pacific:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-False-joint-False-Pacific'
elif atlantic:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-False-joint-False-Atlantic'
else:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-False-joint-False'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_stand = record['solver']
unforced_list_month_stand = record['unforced_list']
pc_month_stand = record['pc']
del record

if pacific:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-False-unforced-False-joint-False-Pacific'
elif atlantic:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-False-unforced-False-joint-False-Atlantic'
else:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-False-unforced-False-joint-False'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver = record['solver']
unforced_list = record['unforced_list']
pc_list = record['pc']
del record

if pacific:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-False-joint-False-Pacific'
elif atlantic:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-False-joint-False-Atlantic'
else:
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-False-joint-False'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_stand = record['solver']
unforced_list_stand = record['unforced_list']
pc_list_stand = record['pc']
del record

pc_all = xr.concat(pc_month, dim='time')
pc_all = pc_all.sortby('time')
pc_all_stand = xr.concat(pc_month_stand, dim='time')
pc_all_stand = pc_all_stand.sortby('time')
pc_all_unforced = xr.concat(pc_month_unforced, dim='time')
pc_all_unforced = pc_all_unforced.sortby('time')

if variable=='tos':
    mask = xr.open_dataset('../maskland.nc')
    missing_xa = xr.where(np.isnan(mask.tos.isel(time=0)), np.nan, 1)
    if pacific:
        missing_xa = missing_xa.sel(lon=slice(150, 280))
    elif atlantic:
        missing_data_maskx = xr.open_dataset('../maskland.nc')
        missing_data_maskx = xc.swap_lon_axis(missing_data_maskx, to=(-180, 180))
        missing_data_maskx = missing_data_maskx.sel(lon=slice(-80, 80))
        missing_data = np.where(np.isnan(missing_data_maskx.tos.squeeze().transpose('lon', 'lat')), np.nan, 1)
        missing_xa = xr.where(np.isnan(missing_data_maskx.tos.isel(time=0)), np.nan, 1)

else:
    mask = xr.open_dataset('../nomask.nc')
    missing_xa = xr.where(np.isnan(mask.tas.isel(time=0)), np.nan, 1)

cobe = xc.open_dataset('/p/lustre3/shiduan/sst/COBE/sst_COBE_195001-202212_2p5x2p5.nc')
if pacific:
    cobe = cobe.sel(lon=slice(150, 280))
elif atlantic:
    cobe = xc.swap_lon_axis(cobe, to=(-180, 180))
    cobe = cobe.sel(lon=slice(-80, 80))
cobe = cobe["sst"].transpose('time', 'lon', 'lat')
cobe = cobe.sel(time=slice('1950-01-01', '2021-12-31'))
if late:
    cobe = cobe.sel(time=slice('1979-01-01', '2021-12-31'))
print(cobe.shape, ' ', cobe.time.data[-1])
cobe = cobe.fillna(0)
cobe = cobe*missing_xa
cobe_anomaly = cobe.groupby(cobe.time.dt.month)-cobe.groupby(cobe.time.dt.month).mean(dim='time')
cobe_unforced = cobe_anomaly.groupby(cobe_anomaly.time.dt.month)/unforced_std['tos']
cobe_stand = cobe_anomaly.groupby(cobe_anomaly.time.dt.month)/cobe_anomaly.groupby(cobe_anomaly.time.dt.month).std(dim='time')
cobe_stand = xr.where(np.isfinite(cobe_stand), cobe_stand, 0)
cobe_stand = cobe_stand.fillna(0)
cobe_stand = cobe_stand * missing_xa

results_month = calculate_metrics(obs=cobe_anomaly,
    solver_list=solver_list_month, unforced_list=unforced_list_month, pc_series=pc_all, month=True)

results_month_stand = calculate_metrics(obs=cobe_stand,
    solver_list=solver_list_month_stand, 
    unforced_list=unforced_list_month_stand, pc_series=pc_all_stand, month=True)

results_month_unforced = calculate_metrics(obs=cobe_unforced,
    solver_list=solver_list_month_unforced, 
    unforced_list=unforced_list_month_unforced, pc_series=pc_all_unforced, month=True)

results_stand = calculate_metrics(obs=cobe_stand,
    solver_list=solver_stand, 
    unforced_list=unforced_list_stand, pc_series=pc_list_stand[0], month=False)

results_unforced = calculate_metrics(obs=cobe_unforced,
    solver_list=solver_list_unforced, 
    unforced_list=unforced_list_unforced, pc_series=pc_unforced[0], month=False)

results_raw= calculate_metrics(obs=cobe_anomaly,
    solver_list=solver, 
    unforced_list=unforced_list, pc_series=pc_list[0], month=False)

if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/COBE/'):
    os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/COBE/')
p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/COBE/'

if pacific:
    path = p+variable+'-metrics-stand-False-month-True-unforced-False-joint-False-Pacific'
elif atlantic:
    path = p+variable+'-metrics-stand-False-month-True-unforced-False-joint-False-Atlantic'
else:
    path = p+variable+'-metrics-stand-False-month-True-unforced-False-joint-False'
if late:
    path = path+'-late'
with open(path, 'wb') as pfile:
    pickle.dump(results_month, pfile)

if pacific:
    path = p+variable+'-metrics-stand-True-month-True-unforced-False-joint-False-Pacific'
elif atlantic:
    path = p+variable+'-metrics-stand-True-month-True-unforced-False-joint-False-Atlantic'
else:
    path = p+variable+'-metrics-stand-True-month-True-unforced-False-joint-False'
if late:
    path = path+'-late'
with open(path, 'wb') as pfile:
    pickle.dump(results_month_stand, pfile)

if pacific:
    path = p+variable+'-metrics-stand-True-month-True-unforced-True-joint-False-Pacific'
elif atlantic:
    path = p+variable+'-metrics-stand-True-month-True-unforced-True-joint-False-Atlantic'
else:
    path = p+variable+'-metrics-stand-True-month-True-unforced-True-joint-False'
if late:
    path = path+'-late'
with open(path, 'wb') as pfile:
    pickle.dump(results_month_unforced, pfile)

if pacific:
    path = p+variable+'-metrics-stand-False-month-False-unforced-False-joint-False-Pacific'
elif atlantic:
    path = p+variable+'-metrics-stand-False-month-False-unforced-False-joint-False-Atlantic'
else:
    path = p+variable+'-metrics-stand-False-month-False-unforced-False-joint-False'
if late:
    path = path+'-late'
with open(path, 'wb') as pfile:
    pickle.dump(results_raw, pfile)

if pacific:
    path = p+variable+'-metrics-stand-True-month-False-unforced-False-joint-False-Pacific'
elif atlantic:
    path = p+variable+'-metrics-stand-True-month-False-unforced-False-joint-False-Atlantic'
else:
    path = p+variable+'-metrics-stand-True-month-False-unforced-False-joint-False'
if late:
    path = path+'-late'
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)

if pacific:
    path = p+variable+'-metrics-stand-True-month-False-unforced-True-joint-False-Pacific'
elif atlantic:
    path = p+variable+'-metrics-stand-True-month-False-unforced-True-joint-False-Atlantic'
else:
    path = p+variable+'-metrics-stand-True-month-False-unforced-True-joint-False'
if late:
    path = path+'-late'
with open(path, 'wb') as pfile:
    pickle.dump(results_unforced, pfile)
