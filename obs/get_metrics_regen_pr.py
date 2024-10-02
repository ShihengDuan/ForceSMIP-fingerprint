import time as clocktime
import glob
import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle

# principal component analysis
from eofs.xarray import Eof
from utils import calculate_metrics_forcesmip

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
    end_year = 2016
elif variable=='pr':
    cmip_var = 'pr'
    eof_start = 1979
    start_year = 1983
    end_year = 2016

print(variable, ' ', start_year, ' ', end_year, ' ', eof_start)


with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-True-joint-False-REGEN-mask', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_unforced = record['solver']
unforced_list_month_unforced = record['unforced_list']
pc_month_unforced = record['pc']
unforced_std = record['unforced_std'] # for obs normalization
pc_all_unforced = xr.concat(pc_month_unforced, dim='time')
pc_all_unforced = pc_all_unforced.sortby('time')

maskfile = "/p/lustre1/shiduan/REGEN/REGEN_mask_forcesmip.nc"
mask = xr.open_dataset(maskfile)
missing_xa = xr.where(np.isnan(mask.p), np.nan, 1)

regen = xc.open_dataset('/p/lustre1/shiduan/REGEN/monpr_forcesmip.nc')
regen = regen['p'].transpose('time', 'lon', 'lat')
regen = regen.sel(time=slice('1983-01-01', '2016-12-31'))
print(regen.shape, ' ', regen.time.data[-1])
regen_anomaly = regen.groupby(regen.time.dt.month)-regen.groupby(regen.time.dt.month).mean(dim='time')
regen_unforced = regen_anomaly.groupby(regen_anomaly.time.dt.month)/unforced_std['pr']
regen_stand = regen_anomaly.groupby(regen_anomaly.time.dt.month)/regen_anomaly.groupby(regen_anomaly.time.dt.month).std(dim='time')
regen_anomaly = regen_anomaly.fillna(0)
regen_anomaly = regen_anomaly*missing_xa
regen_stand = regen_stand.fillna(0)
regen_stand = regen_stand*missing_xa
regen_unforced = regen_unforced.fillna(0)
regen_unforced = regen_unforced*missing_xa

results_month_unforced = calculate_metrics_forcesmip(obs=regen_unforced,
    solver_list=solver_list_month_unforced, 
    unforced_list=unforced_list_month_unforced, pc_series=pc_all_unforced, month=True, 
    missing_xa=missing_xa, start_year=1983, end_year=2016)

with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-True-unforced-False-joint-False-REGEN-mask', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month = record['solver']
unforced_list_month = record['unforced_list']
pc_month = record['pc']
pc_all = xr.concat(pc_month, dim='time')
pc_all = pc_all.sortby('time')
results_month = calculate_metrics_forcesmip(obs=regen_anomaly,
    solver_list=solver_list_month, unforced_list=unforced_list_month, pc_series=pc_all, month=True,
    missing_xa=missing_xa, start_year=1983, end_year=2016)

with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-True-unforced-False-joint-False-REGEN-mask', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_stand = record['solver']
unforced_list_month_stand = record['unforced_list']
pc_month_stand = record['pc']

pc_all_stand = xr.concat(pc_month_stand, dim='time')
pc_all_stand = pc_all_stand.sortby('time')
results_month_stand = calculate_metrics_forcesmip(obs=regen_stand,
    solver_list=solver_list_month_stand, 
    unforced_list=unforced_list_month_stand, pc_series=pc_all_stand, month=True,
    missing_xa=missing_xa, start_year=1983, end_year=2016)

with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-False-month-False-unforced-False-joint-False-REGEN-mask', 'rb') as pfile:
    record = pickle.load(pfile)
solver = record['solver']
unforced_list = record['unforced_list']
pc_list = record['pc']

results_raw= calculate_metrics_forcesmip(obs=regen_anomaly,
    solver_list=solver, 
    unforced_list=unforced_list, pc_series=pc_list[0], month=False,
    missing_xa=missing_xa, start_year=1983, end_year=2016)


with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-False-joint-False-REGEN-mask', 'rb') as pfile:
    record = pickle.load(pfile)
solver_stand = record['solver']
unforced_list_stand = record['unforced_list']
pc_list_stand = record['pc']

results_stand = calculate_metrics_forcesmip(obs=regen_stand,
    solver_list=solver_stand, 
    unforced_list=unforced_list_stand, pc_series=pc_list_stand[0], month=False,
    missing_xa=missing_xa, start_year=1983, end_year=2016)


with open('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/'+variable+'-record-stand-True-month-False-unforced-True-joint-False-REGEN-mask', 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_unforced = record['solver']
unforced_list_unforced = record['unforced_list']
pc_unforced = record['pc']

results_unforced = calculate_metrics_forcesmip(obs=regen_unforced,
    solver_list=solver_list_unforced, 
    unforced_list=unforced_list_unforced, pc_series=pc_unforced[0], month=False,
    missing_xa=missing_xa, start_year=1983, end_year=2016)



if not os.path.exists('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/REGEN/'):
    os.makedirs('/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/REGEN/')
p = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(eof_start)+'_2022/REGEN/'


path = p+variable+'-metrics-stand-False-month-True-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_month, pfile)

path = p+variable+'-metrics-stand-True-month-True-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_month_stand, pfile)

path = p+variable+'-metrics-stand-True-month-True-unforced-True-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_month_unforced, pfile)

path = p+variable+'-metrics-stand-False-month-False-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_raw, pfile)

path = p+variable+'-metrics-stand-True-month-False-unforced-False-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)

path = p+variable+'-metrics-stand-True-month-False-unforced-True-joint-False'
if n_mode>1:
    path = path+'-n_mode-'+str(n_mode)
with open(path, 'wb') as pfile:
    pickle.dump(results_unforced, pfile)
