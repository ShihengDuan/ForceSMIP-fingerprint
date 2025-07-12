import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle
import pandas as pd

from utils import get_slope
import argparse

def calculate_metrics_forcesmip_calendar_month(solver_list, obs, unforced_list, pc_series, missing_xa, month=False, 
                                n_mode=1, start_year=1983, end_year=2020, 
                                cmip_var='pr', calendar_month=1):
    pc1 = pc_series.isel(mode=n_mode-1)
    if month:
        reversed_month = []
        reorder_pc1 = []
        # for m in range(1, 13):
        pc1_month = pc1.sel(time=pc1.time.dt.month==calendar_month)
        m, b = np.polyfit(np.arange(pc1_month.shape[0]), pc1_month, deg=1)
        if m<0:
            pc1_month = -pc1_month
            reversed_month.append(calendar_month)
        reorder_pc1.append(pc1_month)
        pc1 = pc1_month
        print('reversed_month: ', reversed_month)
        # pc1 = pc1.sortby('time')
    else:
        m, b = np.polyfit(np.arange(pc1.shape[0]), pc1, deg=1)
        if m<0:
            reverse = True
            pc1 = -pc1
            print('Reverse')
        else:
            reverse = False
        pc1 = pc1.sel(time=pc1.time.dt.month==calendar_month)
    pc1 = pc1.sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
    
    timescales = np.arange(12, (end_year+1-start_year)*12)
    pc_max = pc1.max().data
    pc_min = pc1.min().data
    print(pc_max, ' ', pc_min)
    # normalize cmip ensemble pcs. 
    if month:
        pseudo_pc_month = []
        # for m in range(1, 13):
        # normalize 
        solver = solver_list[calendar_month-1]
        ds_in = obs.sel(time=obs.time.dt.month==calendar_month)
        pseudo_pc = solver.projectField(ds_in-ds_in.mean(dim='time')).isel(mode=n_mode-1)
        if m in reversed_month: # flip sign based on solver pc. 
            pseudo_pc = -pseudo_pc
        # pseudo_pc_month.append(pseudo_pc)
        # pseudo_pc = xr.concat(pseudo_pc_month, dim='time')
        # pseudo_pc = pseudo_pc.sortby('time')
        pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min) # 0 to 1 
        pseudo_pc = pseudo_pc*2-1 # -1 to 1 
    else:
        # normalize pc series. 
        ds_in = obs.sel(time=obs.time.dt.month==calendar_month)
        pseudo_pc = solver_list[0].projectField(ds_in-ds_in.mean(dim='time')).isel(mode=n_mode-1)
        if reverse:
            pseudo_pc = -pseudo_pc
        pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min)
        pseudo_pc = pseudo_pc*2-1
    pseudo_pc = pseudo_pc.sel(time=pseudo_pc.time.dt.month==calendar_month)
    print(pseudo_pc.shape, pseudo_pc.max().data, ' ', pseudo_pc.min().data)
    # get noise and signal
    noise_pcs = []
    for unforced in unforced_list:
        unforced = unforced[cmip_var].sel(time=slice(str(start_year)+'-01-01', str(2022)+'-12-31'))
        unforced = unforced
        member = unforced.shape[0]
        for m in range(member):
            ds_in = unforced.isel(member=m)*missing_xa
            ds_in = ds_in.transpose('time', 'lon', 'lat')
            if month:
                noise_month = []
                # for mm in range(1, 13):
                solver = solver_list[calendar_month-1]
                ds_in_month = ds_in.sel(time=ds_in.time.dt.month==calendar_month)
                # print(month, ' ', np.sum(np.isnan(ds_in)).data)
                psd = solver.projectField(ds_in_month-ds_in_month.mean(dim='time'))
                if calendar_month in reversed_month:
                    psd = -psd
                noise_month.append(psd)

                # noise_month = xr.concat(noise_month, dim='time')
                # noise_month = noise_month.sortby('time')
                noise_month = psd
                noise_month = noise_month.isel(mode=n_mode-1).sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
                noise_month = (noise_month-pc_min)/(pc_max-pc_min)
                noise_month = noise_month*2-1
                noise_month = noise_month.sel(time=noise_month.time.dt.month==calendar_month)
                noise_pcs.append(noise_month)
            else:
                psd = solver_list[0].projectField(ds_in-ds_in.mean(dim='time'))
                psd = psd.isel(mode=n_mode-1).sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
                if reverse:
                    psd = -psd
                psd = (psd-pc_max)/(pc_max-pc_min)
                psd = psd*2-1
                psd = psd.sel(time=psd.time.dt.month==calendar_month)
                noise_pcs.append(psd)
    # get noise strength
    timescales = np.arange(5, (end_year-start_year+1)) # annual timestep
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eof', type=int, default=1)
    parser.add_argument('--regen_mask', type=int, default=0)
    parser.add_argument('--month', type=int, default=1)
    args = vars(parser.parse_args())
    return args

args = get_args()
n_mode = args['eof']
regen_mask = args['regen_mask']>0
calendar_month = args['month']
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
    if regen_mask:
        end_year = 2016 # land only pr. 

print(variable, ' ', start_year, ' ', end_year, ' ', eof_start)
path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-True-unforced-False-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pd.read_pickle(pfile)
solver_list_month = record['solver']
unforced_list_month = record['unforced_list']
pc_month = record['pc']
all_pcs_month = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-False-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pd.read_pickle(pfile)
solver_list_month_stand = record['solver']
unforced_list_month_stand = record['unforced_list']
pc_month_stand = record['pc']
all_pcs_month_stand = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-False-unforced-False-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pd.read_pickle(pfile)
solver = record['solver']
unforced_list = record['unforced_list']
pc_list = record['pc']
all_pcs = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-False-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pd.read_pickle(pfile)
solver_stand = record['solver']
unforced_list_stand = record['unforced_list']
pc_list_stand = record['pc']
all_pcs_stand = record['all_pcs']

pc_all = xr.concat(pc_month, dim='time')
pc_all = pc_all.sortby('time')

pc_all_stand = xr.concat(pc_month_stand, dim='time')
pc_all_stand = pc_all_stand.sortby('time')

if variable=='tos':
    mask = xr.open_dataset('../maskland.nc')
    missing_xa = xr.where(np.isnan(mask.tos.isel(time=0)), np.nan, 1)
else:
    mask = xr.open_dataset('../nomask.nc')
    missing_xa = xr.where(np.isnan(mask.tas.isel(time=0)), np.nan, 1)
if regen_mask:
    maskfile = "/p/lustre1/shiduan/REGEN/REGEN_mask_forcesmip.nc"
    missing_data_maskx = xr.open_dataset(maskfile)
    missing_data = np.where(np.isnan(missing_data_maskx.p.transpose('lon', 'lat')), np.nan, 1)
    missing_xa = xr.where(np.isnan(missing_data_maskx.p), np.nan, 1)
# load GPCP
gpcp = []
for i in range(1983, 2021):
    data = xc.open_dataset('/p/lustre3/shiduan/GPCP/regrid/'+str(i)+'.nc')
    gpcp.append(data)
gpcp = xr.concat(gpcp, dim='time')
print(gpcp.time)
gpcp = gpcp["__xarray_dataarray_variable__"].transpose('time', 'lon', 'lat')
gpcp = gpcp.fillna(0)
gpcp = gpcp*missing_xa
if regen_mask:
    gpcp = gpcp.sel(time=slice('1983-01-01', '2016-12-31'))
gpcp_anomaly = gpcp.groupby(gpcp.time.dt.month)-gpcp.groupby(gpcp.time.dt.month).mean(dim='time')
gpcp_stand = gpcp_anomaly.groupby(gpcp_anomaly.time.dt.month)/gpcp_anomaly.groupby(gpcp_anomaly.time.dt.month).std(dim='time')



results_month = calculate_metrics_forcesmip_calendar_month(obs=gpcp_anomaly,
    solver_list=solver_list_month, unforced_list=unforced_list_month, pc_series=pc_all, month=True, n_mode=n_mode, missing_xa=missing_xa,
    start_year=start_year, end_year=end_year, calendar_month=calendar_month)

results_month_stand = calculate_metrics_forcesmip_calendar_month(obs=gpcp_stand,
    solver_list=solver_list_month_stand, 
    unforced_list=unforced_list_month_stand, pc_series=pc_all_stand, month=True, n_mode=n_mode, missing_xa=missing_xa,
    start_year=start_year, end_year=end_year, calendar_month=calendar_month)



results_stand = calculate_metrics_forcesmip_calendar_month(obs=gpcp_stand,
    solver_list=solver_stand, 
    unforced_list=unforced_list_stand, pc_series=pc_list_stand[0], month=False, n_mode=n_mode, missing_xa=missing_xa,
    start_year=start_year, end_year=end_year, calendar_month=calendar_month)


results_raw= calculate_metrics_forcesmip_calendar_month(obs=gpcp_anomaly,
    solver_list=solver, 
    unforced_list=unforced_list, pc_series=pc_list[0], month=False, n_mode=n_mode, missing_xa=missing_xa,
    start_year=start_year, end_year=end_year, calendar_month=calendar_month)

if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/GPCP/'):
    os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/GPCP/')
p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/GPCP/'

path = p+variable+'-metrics-stand-False-month-True-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
path = path+'-calendar_month-'+str(calendar_month)
with open(path, 'wb') as pfile:
    pickle.dump(results_month, pfile)

path = p+variable+'-metrics-stand-True-month-True-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
path = path+'-calendar_month-'+str(calendar_month)
with open(path, 'wb') as pfile:
    pickle.dump(results_month_stand, pfile)


path = p+variable+'-metrics-stand-False-month-False-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
path = path+'-calendar_month-'+str(calendar_month)
with open(path, 'wb') as pfile:
    pickle.dump(results_raw, pfile)

path = p+variable+'-metrics-stand-True-month-False-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
path = path+'-calendar_month-'+str(calendar_month)
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)

# MSWEP
# load MSWEP
mswep = xc.open_dataset('/p/lustre3/shiduan/MSWEP/MSWEP-V280-Past-v20231102-monpr-forcesmip.nc')
mswep = mswep['__xarray_dataarray_variable__'].transpose('time', 'lon', 'lat')
mswep = mswep.fillna(0)
mswep = mswep*missing_xa
mswep = mswep.sel(time=slice('1983-01-01', '2021-01-01'))
if regen_mask:
    mswep = mswep.sel(time=slice('1983-01-01', '2016-12-31'))
print(mswep.shape, ' ', mswep.time.data[-1])
mswep_anomaly = mswep.groupby(mswep.time.dt.month)-mswep.groupby(mswep.time.dt.month).mean(dim='time')
mswep_stand = mswep_anomaly.groupby(mswep_anomaly.time.dt.month)/mswep_anomaly.groupby(mswep_anomaly.time.dt.month).std(dim='time')


results_month = calculate_metrics_forcesmip_calendar_month(obs=mswep_anomaly,
    solver_list=solver_list_month, unforced_list=unforced_list_month, pc_series=pc_all, month=True, n_mode=n_mode, missing_xa=missing_xa,
    start_year=start_year, end_year=end_year, calendar_month=calendar_month)

results_month_stand = calculate_metrics_forcesmip_calendar_month(obs=mswep_stand,
    solver_list=solver_list_month_stand, 
    unforced_list=unforced_list_month_stand, pc_series=pc_all_stand, month=True, n_mode=n_mode, missing_xa=missing_xa,
    start_year=start_year, end_year=end_year, calendar_month=calendar_month)

results_stand = calculate_metrics_forcesmip_calendar_month(obs=mswep_stand,
    solver_list=solver_stand, 
    unforced_list=unforced_list_stand, pc_series=pc_list_stand[0], month=False, n_mode=n_mode, missing_xa=missing_xa,
    start_year=start_year, end_year=end_year, calendar_month=calendar_month)

results_raw= calculate_metrics_forcesmip_calendar_month(obs=mswep_anomaly,
    solver_list=solver, 
    unforced_list=unforced_list, pc_series=pc_list[0], month=False, n_mode=n_mode, missing_xa=missing_xa,
    start_year=start_year, end_year=end_year, calendar_month=calendar_month)

if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/MSWEP/'):
    os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/MSWEP/')
p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/MSWEP/'

path = p+variable+'-metrics-stand-False-month-True-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
path = path+'-calendar_month-'+str(calendar_month)
with open(path, 'wb') as pfile:
    pickle.dump(results_month, pfile)

path = p+variable+'-metrics-stand-True-month-True-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
path = path+'-calendar_month-'+str(calendar_month)
with open(path, 'wb') as pfile:
    pickle.dump(results_month_stand, pfile)


path = p+variable+'-metrics-stand-False-month-False-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
path = path+'-calendar_month-'+str(calendar_month)
with open(path, 'wb') as pfile:
    pickle.dump(results_raw, pfile)

path = p+variable+'-metrics-stand-True-month-False-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
path = path+'-calendar_month-'+str(calendar_month)
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)
