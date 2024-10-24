import time as clocktime
import glob
import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle

# principal component analysis
from eofs.xarray import Eof
from utils import get_slope, get_picontrol_cmip5, calculate_metrics_cmip5
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eof', type=int, default=1)
    parser.add_argument('--eof_end', type=int, default=2016)
    parser.add_argument('--less', type=int, default=0)
    args = vars(parser.parse_args())
    return args


args = get_args()
n_mode = args['eof']
eof_end = args['eof_end']
variable = 'pr'
less = args['less'] > 0

if variable == 'tos':
    cmip_var = 'tos'
    eof_start = 1950
    start_year = 1950
    end_year = 2019
elif variable == 'monmaxpr':
    cmip_var = 'pr'
    eof_start = 1979
    start_year = 1980
    end_year = 2020
elif variable == 'pr':
    cmip_var = 'pr'
    eof_start = 1979
    start_year = 1983
    end_year = 2020

print(variable, ' ', start_year, ' ', end_year, ' ', eof_start)
root_path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_'+str(eof_end)+'_cmip5/'
if less:
    root_path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
        str(eof_start)+'_'+str(eof_end)+'_cmip5_less/'
with open(root_path+variable+'-record-stand-False-month-True-unforced-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month = record['solver']
pc_month = record['pc']

with open(root_path+variable+'-record-stand-True-month-True-unforced-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_stand = record['solver']
pc_month_stand = record['pc']

with open(root_path+variable+'-record-stand-False-month-False-unforced-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver = record['solver']
pc_list = record['pc']


with open(root_path+variable+'-record-stand-True-month-False-unforced-False', 'rb') as pfile:
    record = pickle.load(pfile)
solver_stand = record['solver']
pc_list_stand = record['pc']


pc_all = xr.concat(pc_month, dim='time')
pc_all = pc_all.sortby('time')
pc_all_stand = xr.concat(pc_month_stand, dim='time')
pc_all_stand = pc_all_stand.sortby('time')

if variable == 'tos':
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

gpcp = gpcp["__xarray_dataarray_variable__"].transpose('time', 'lon', 'lat')
gpcp = gpcp.fillna(0)
if eof_end < 2020:
    gpcp = gpcp.sel(time=slice('1983-01-01', str(eof_end)+'-12-31'))
print(gpcp.time)
gpcp_anomaly = gpcp.groupby(gpcp.time.dt.month) - \
    gpcp.groupby(gpcp.time.dt.month).mean(dim='time')
gpcp_stand = gpcp_anomaly.groupby(gpcp_anomaly.time.dt.month) / \
    gpcp_anomaly.groupby(gpcp_anomaly.time.dt.month).std(dim='time')

picontrol, picontrol_std = get_picontrol_cmip5()
end_year = np.min([eof_end, 2020])
print('end_year: ', end_year)
results_month_stand = calculate_metrics_cmip5(obs=gpcp_stand, missing_xa=missing_xa,
                                              solver_list=solver_list_month_stand,
                                              unforced_list=picontrol_std, pc_series=pc_all_stand, month=True, n_mode=n_mode,
                                              start_year=start_year, end_year=end_year)
# month stand
center = np.array(results_month_stand['s_obs_list'])
center = center/np.array(results_month_stand['n_list'])
center = xr.DataArray(center, dims=['time'], coords={
                      'time': np.arange(len(center))})
center = center.reindex(time=list(reversed(center.time)))
indices = (center < 2)
first_indices = (indices.argmax(dim='time'))
whole_period = center.time.shape[0]
back = whole_period-first_indices
if back < whole_period:
    print(back.data, ' back')
else:
    print('None')

results_month = calculate_metrics_cmip5(obs=gpcp_anomaly, missing_xa=missing_xa,
                                        solver_list=solver_list_month, unforced_list=picontrol, pc_series=pc_all, month=True, n_mode=n_mode,
                                        start_year=start_year, end_year=end_year)
center = np.array(results_month['s_obs_list'])
center = center/np.array(results_month['n_list'])
center = xr.DataArray(center, dims=['time'], coords={
                      'time': np.arange(len(center))})
center = center.reindex(time=list(reversed(center.time)))
indices = (center < 2)
first_indices = (indices.argmax(dim='time'))
whole_period = center.time.shape[0]
back = whole_period-first_indices
if back < whole_period:
    print(back.data, ' back')
else:
    print('None')

results_stand = calculate_metrics_cmip5(obs=gpcp_stand, missing_xa=missing_xa,
                                        solver_list=solver_stand,
                                        unforced_list=picontrol_std, pc_series=pc_list_stand[0], month=False, n_mode=n_mode,
                                        start_year=start_year, end_year=end_year)


results_raw = calculate_metrics_cmip5(obs=gpcp_anomaly, missing_xa=missing_xa,
                                      solver_list=solver,
                                      unforced_list=picontrol, pc_series=pc_list[0], month=False, n_mode=n_mode,
                                      start_year=start_year, end_year=end_year)

if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_'+str(eof_end)+'_cmip5/GPCP/'):
    os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' +
                str(eof_start)+'_'+str(eof_end)+'_cmip5/GPCP/')
if less:
    if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_'+str(eof_end)+'_cmip5_less/GPCP/'):
        os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' +
                    str(eof_start)+'_'+str(eof_end)+'_cmip5_less/GPCP/')
p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_'+str(eof_end)+'_cmip5/GPCP/'
if less:
    p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
        str(eof_start)+'_'+str(eof_end)+'_cmip5_less/GPCP/'
path = p+variable+'-metrics-stand-False-month-True-unforced-False'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_month, pfile)

path = p+variable+'-metrics-stand-True-month-True-unforced-False'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_month_stand, pfile)


path = p+variable+'-metrics-stand-False-month-False-unforced-False'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_raw, pfile)

path = p+variable+'-metrics-stand-True-month-False-unforced-False'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)
