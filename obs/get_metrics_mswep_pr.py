import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle
import pandas as pd

from utils import calculate_metrics_forcesmip
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eof', type=int, default=1)
    parser.add_argument('--regen_mask', type=int, default=0)
    args = vars(parser.parse_args())
    return args

args = get_args()
n_mode = args['eof']
regen_mask = args['regen_mask']>0
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

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-True-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pd.read_pickle(pfile)
solver_list_month_unforced = record['solver']
unforced_list_month_unforced = record['unforced_list']
pc_month_unforced = record['pc']
all_pcs_month_unforced = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-True-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pd.read_pickle(pfile)
solver_list_unforced = record['solver']
unforced_list_unforced = record['unforced_list']
pc_unforced = record['pc']
all_pcs_unforced = record['all_pcs']
unforced_std = record['unforced_std'] # for obs normalization


pc_all = xr.concat(pc_month, dim='time')
pc_all = pc_all.sortby('time')

pc_all_stand = xr.concat(pc_month_stand, dim='time')
pc_all_stand = pc_all_stand.sortby('time')

pc_all_unforced = xr.concat(pc_month_unforced, dim='time')
pc_all_unforced = pc_all_unforced.sortby('time')

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

mswep = xc.open_dataset('/p/lustre3/shiduan/MSWEP/MSWEP-V280-Past-v20231102-monpr-forcesmip.nc')
mswep = mswep['__xarray_dataarray_variable__'].transpose('time', 'lon', 'lat')
mswep = mswep.fillna(0)
mswep = mswep*missing_xa
mswep = mswep.sel(time=slice('1983-01-01', '2021-01-01'))
if regen_mask:
    mswep = mswep.sel(time=slice('1983-01-01', '2016-12-31'))
print(mswep.shape, ' ', mswep.time.data[-1])
mswep_anomaly = mswep.groupby(mswep.time.dt.month)-mswep.groupby(mswep.time.dt.month).mean(dim='time')
mswep_unforced = mswep_anomaly.groupby(mswep_anomaly.time.dt.month)/unforced_std['pr']
mswep_stand = mswep_anomaly.groupby(mswep_anomaly.time.dt.month)/mswep_anomaly.groupby(mswep_anomaly.time.dt.month).std(dim='time')


results_month = calculate_metrics_forcesmip(n_mode=n_mode, obs=mswep_anomaly,
    solver_list=solver_list_month, unforced_list=unforced_list_month, pc_series=pc_all, month=True, missing_xa=missing_xa, 
    start_year=start_year, end_year=end_year)

results_month_stand = calculate_metrics_forcesmip(n_mode=n_mode, obs=mswep_stand,
    solver_list=solver_list_month_stand, 
    unforced_list=unforced_list_month_stand, pc_series=pc_all_stand, month=True, missing_xa=missing_xa, 
    start_year=start_year, end_year=end_year)

results_month_unforced = calculate_metrics_forcesmip(n_mode=n_mode, obs=mswep_unforced,
    solver_list=solver_list_month_unforced, 
    unforced_list=unforced_list_month_unforced, pc_series=pc_all_unforced, month=True, missing_xa=missing_xa, 
    start_year=start_year, end_year=end_year)

results_stand = calculate_metrics_forcesmip(n_mode=n_mode, obs=mswep_stand,
    solver_list=solver_stand, 
    unforced_list=unforced_list_stand, pc_series=pc_list_stand[0], month=False, missing_xa=missing_xa, 
    start_year=start_year, end_year=end_year)

results_unforced = calculate_metrics_forcesmip(n_mode=n_mode, obs=mswep_unforced,
    solver_list=solver_list_unforced, 
    unforced_list=unforced_list_unforced, pc_series=pc_unforced[0], month=False, missing_xa=missing_xa, 
    start_year=start_year, end_year=end_year)

results_raw= calculate_metrics_forcesmip(n_mode=n_mode, obs=mswep_anomaly,
    solver_list=solver, 
    unforced_list=unforced_list, pc_series=pc_list[0], month=False, missing_xa=missing_xa, 
    start_year=start_year, end_year=end_year)


if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/MSWEP/'):
    os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/MSWEP/')
p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/MSWEP/'


path = p+variable+'-metrics-stand-False-month-True-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_month, pfile)

path = p+variable+'-metrics-stand-True-month-True-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_month_stand, pfile)

path = p+variable+'-metrics-stand-True-month-True-unforced-True-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_month_unforced, pfile)

path = p+variable+'-metrics-stand-False-month-False-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_raw, pfile)

path = p+variable+'-metrics-stand-True-month-False-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)

path = p+variable+'-metrics-stand-True-month-False-unforced-True-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_unforced, pfile)
