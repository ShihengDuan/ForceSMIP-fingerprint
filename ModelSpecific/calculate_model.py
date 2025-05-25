import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle

import argparse

def calculate_metrics_forcesmip(solver_list, unforced_list, pc_series, missing_xa, month=False, n_mode=1, start_year=1983, end_year=2020, cmip_var='pr'):
    pc1 = pc_series.isel(mode=n_mode-1)
    if month:
        reversed_month = []
        reorder_pc1 = []
        for month in range(1, 13):
            pc1_month = pc1.sel(time=pc1.time.dt.month==month)
            # print(month, ' ', pc1_month)
            m, b = np.polyfit(np.arange(pc1_month.shape[0]), pc1_month, deg=1)
            if m<0:
                pc1_month = -pc1_month
                reversed_month.append(month)
            reorder_pc1.append(pc1_month)
        pc1 = xr.concat(reorder_pc1, dim='time')
        print('reversed_month: ', reversed_month)
        pc1 = pc1.sortby('time')
    else:
        m, b = np.polyfit(np.arange(pc1.shape[0]), pc1, deg=1)
        if m<0:
            reverse = True
            pc1 = -pc1
            print('Reverse')
        else:
            reverse = False
    pc1 = pc1.sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
    timescales = np.arange(12, (end_year+1-start_year)*12)
    pc_max = pc1.max().data
    pc_min = pc1.min().data
    print(pc_max, ' ', pc_min)
    
    # get noise and signal
    noise_pcs = []
    for unforced in unforced_list:
        unforced = unforced[cmip_var].sel(time=slice(str(start_year)+'-01-01', str(2022)+'-12-31'))
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
                    if month in reversed_month:
                        psd = -psd
                    noise_month.append(psd)
                noise_month = xr.concat(noise_month, dim='time')
                noise_month = noise_month.sortby('time')
                noise_month = noise_month.isel(mode=n_mode-1).sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
                noise_month = (noise_month-pc_min)/(pc_max-pc_min)
                noise_month = noise_month*2-1
                noise_pcs.append(noise_month)
            else:
                psd = solver_list[0].projectField(ds_in-ds_in.mean(dim='time'))
                psd = psd.isel(mode=n_mode-1).sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
                if reverse:
                    psd = -psd
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
        
    sn = []
    s_list = []
    n_list = []
    for ts in timescales:
        # compute s/n ratio from pre-computed
        # signal/noise values
        s = signal[ts]
        n = np.std(noise[ts])
        sn.append(s/n)
        s_list.append(s)
        n_list.append(n)
    results = {
        'sn': sn, 
        'signal':signal, 'noise':noise, 
        's_list':s_list, 'n_list':n_list,  
        'pc':pc1_norm,
        'pc_max': pc_max, 'pc_min': pc_min,
    }
    return results

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eof', type=int, default=1)
    parser.add_argument('--model', type=str, default='CanESM5')
    parser.add_argument('--regen_mask', type=int, default=0)
    args = vars(parser.parse_args())
    return args

args = get_args()
n_mode = args['eof']
regen_mask = args['regen_mask']>0
variable = 'pr'
model = args['model']
print('model: ', model)

eof_start = 1979
start_year = 1983
end_year = 2020

mask = xr.open_dataset('../nomask.nc')
missing_xa = xr.where(np.isnan(mask.tas.isel(time=0)), np.nan, 1)
with open(f'/g/g92/shiduan/lustre2/ForceSMIP/EOF/modes_all/1979_2022/model_{model}/pr-record-stand-False-month-False-unforced-False-joint-False', 'rb') as pfile:
    model_record = pickle.load(pfile)
with open(f'/g/g92/shiduan/lustre2/ForceSMIP/EOF/modes_all/1979_2022/model_{model}/pr-record-stand-False-month-True-unforced-False-joint-False', 'rb') as pfile:
    model_month_record = pickle.load(pfile)
with open(f'/g/g92/shiduan/lustre2/ForceSMIP/EOF/modes_all/1979_2022/model_{model}/pr-record-stand-True-month-False-unforced-False-joint-False', 'rb') as pfile:
    model_stand_record = pickle.load(pfile)
with open(f'/g/g92/shiduan/lustre2/ForceSMIP/EOF/modes_all/1979_2022/model_{model}/pr-record-stand-True-month-True-unforced-False-joint-False', 'rb') as pfile:
    model_stand_month_record = pickle.load(pfile)

# AllMonth, Anomaly
print('AllMonth, Anomaly')
results = calculate_metrics_forcesmip(model_record['solver'], model_record['unforced_list'], model_record['pc'][0], 
                                      missing_xa, month=False, n_mode=n_mode, start_year=start_year, end_year=end_year, cmip_var=variable)
path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/model_'+str(model)+'/'+variable + \
    '-CMIP-metrics-stand-False-month-False-unforced-False-joint-False'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results, pfile)

# AllMonth, StdAnomaly
print('AllMonth, StdAnomaly')
results = calculate_metrics_forcesmip(model_stand_record['solver'], model_stand_record['unforced_list'], model_stand_record['pc'][0], 
                                      missing_xa, month=False, n_mode=n_mode, start_year=start_year, end_year=end_year, cmip_var=variable)
path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/model_'+str(model)+'/'+variable + \
    '-CMIP-metrics-stand-True-month-False-unforced-False-joint-False'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results, pfile)

# MbyM, Anomaly
print('MonthbyMonth, Anomaly')
pc_all = xr.concat(model_month_record['pc'], dim='time')
pc_all = pc_all.sortby('time')
results = calculate_metrics_forcesmip(model_month_record['solver'], model_month_record['unforced_list'], pc_all, 
                                      missing_xa, month=False, n_mode=n_mode, start_year=start_year, end_year=end_year, cmip_var=variable)
path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/model_'+str(model)+'/'+variable + \
    '-CMIP-metrics-stand-False-month-True-unforced-False-joint-False'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results, pfile)

# MbyM, StdAnomaly
print('MonthbyMonth, StdAnomaly')
pc_all = xr.concat(model_stand_month_record['pc'], dim='time')
pc_all = pc_all.sortby('time')
results = calculate_metrics_forcesmip(model_stand_month_record['solver'], model_stand_month_record['unforced_list'], pc_all, 
                                      missing_xa, month=False, n_mode=n_mode, start_year=start_year, end_year=end_year, cmip_var=variable)
path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/model_'+str(model)+'/'+variable + \
    '-CMIP-metrics-stand-True-month-True-unforced-False-joint-False'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results, pfile)
