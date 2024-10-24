import xarray as xr
import numpy as np
import pickle

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', type=str,
                        choices=['monmaxpr', 'pr', 'tos'], default='pr')
    parser.add_argument('--late', type=int, default=0)
    parser.add_argument('--eof', type=int, default=1)
    # REGEN land mask.
    parser.add_argument('--regen_mask', type=int, default=0)
    args = vars(parser.parse_args())
    return args


args = get_args()
variable = args['variable']
late = args['late']
n_mode = args['eof']
regen_mask = args['regen_mask'] > 0
# start_year = args['start_year']
# end_year = args['end_year']
if variable == 'tos':
    cmip_var = 'tos'
    eof_start = 1950
    start_year = 1950
    end_year = 2021
    if late:
        start_year = 1979
        eof_start = 1979
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
    if regen_mask:
        end_year = 2016  # land only pr.

print(variable, ' ', start_year, ' ', end_year, ' ', eof_start)

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-record-stand-False-month-True-unforced-False-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month = record['solver']
unforced_list_month = record['unforced_list']
pc_month = record['pc']
all_pcs_month = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-record-stand-True-month-True-unforced-False-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_stand = record['solver']
unforced_list_month_stand = record['unforced_list']
pc_month_stand = record['pc']
all_pcs_month_stand = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-record-stand-False-month-False-unforced-False-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver = record['solver']
unforced_list = record['unforced_list']
pc_list = record['pc']
all_pcs = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-record-stand-True-month-False-unforced-False-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_stand = record['solver']
unforced_list_stand = record['unforced_list']
pc_list_stand = record['pc']
all_pcs_stand = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-record-stand-True-month-True-unforced-True-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_month_unforced = record['solver']
unforced_list_month_unforced = record['unforced_list']
pc_month_unforced = record['pc']
all_pcs_month_unforced = record['all_pcs']

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-record-stand-True-month-False-unforced-True-joint-False'
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'rb') as pfile:
    record = pickle.load(pfile)
solver_list_unforced = record['solver']
unforced_list_unforced = record['unforced_list']
pc_unforced = record['pc']
all_pcs_unforced = record['all_pcs']


pc_all = xr.concat(pc_month, dim='time')
pc_all = pc_all.sortby('time')

pc_all_stand = xr.concat(pc_month_stand, dim='time')
pc_all_stand = pc_all_stand.sortby('time')

pc_all_unforced = xr.concat(pc_month_unforced, dim='time')
pc_all_unforced = pc_all_unforced.sortby('time')


if variable == 'tos':
    mask = xr.open_dataset('../maskland.nc')
    missing_xa = xr.where(np.isnan(mask.tos.isel(time=0)), np.nan, 1)
else:
    mask = xr.open_dataset('../nomask.nc')
    missing_xa = xr.where(np.isnan(mask.tas.isel(time=0)), np.nan, 1)
if regen_mask:
    maskfile = "/p/lustre1/shiduan/REGEN/REGEN_mask_forcesmip.nc"
    missing_data_maskx = xr.open_dataset(maskfile)
    missing_data = np.where(
        np.isnan(missing_data_maskx.p.transpose('lon', 'lat')), np.nan, 1)
    missing_xa = xr.where(np.isnan(missing_data_maskx.p), np.nan, 1)


def calculate_metrics_cmip(solver_list, cmip_pcs, unforced_list, pc_series, month=False, n_mode=1):
    # cmip_pcs: cmip6 pseudo pcs
    # pc_series: solver pc.
    pc1 = pc_series.isel(mode=n_mode-1)
    if month:
        reversed_month = []
        reorder_pc1 = []
        for month in range(1, 13):
            # reversed based on solver pc.
            pc1_month = pc1.sel(time=pc1.time.dt.month == month)
            m, b = np.polyfit(np.arange(pc1_month.shape[0]), pc1_month, deg=1)
            if m < 0:
                pc1_month = -pc1_month
                reversed_month.append(month)
            reorder_pc1.append(pc1_month)
        pc1 = xr.concat(reorder_pc1, dim='time')
        print('reversed_month: ', reversed_month)
        pc1 = pc1.sortby('time')
    else:
        m, b = np.polyfit(np.arange(pc1.shape[0]), pc1, deg=1)
        if m < 0:
            reverse = True
            pc1 = -pc1
            print('Reverse')
        else:
            reverse = False
    pc1 = pc1.sel(time=slice(str(start_year) +
                  '-01-01', str(end_year+1)+'-01-01'))
    timescales = np.arange(12, (end_year+1-start_year)*12)
    pc_max = pc1.max().data
    pc_min = pc1.min().data
    print(pc_max, ' ', pc_min)
    # normalize cmip ensemble pcs.
    cmip_psedupcs = []
    time = np.arange(len(pc1))
    if not month:
        for model_pc in cmip_pcs:
            model_pc = model_pc.sel(time=slice(
                str(start_year)+'-01-01', str(end_year+1)+'-01-01')).isel(mode=n_mode-1)
            if reverse:
                model_pc = -model_pc
            model_pc = (model_pc-pc_min)/(pc_max-pc_min)
            model_pc = model_pc*2-1  # -1 to 1
            cmip_psedupcs.append(model_pc)
    if month:
        for model_pc in cmip_pcs:
            reorder_pc = []
            for month in range(1, 13):
                model_pc_month = model_pc.sel(time=slice(
                    str(start_year)+'-01-01', str(end_year+1)+'-01-01')).isel(mode=n_mode-1)
                model_pc_month = model_pc_month.sel(
                    time=model_pc_month.time.dt.month == month)
                if month in reversed_month:  # flip sign based on solver pc.
                    model_pc_month = -model_pc_month
                reorder_pc.append(model_pc_month)
            reorder_pc = xr.concat(reorder_pc, dim='time')
            reorder_pc = reorder_pc.sortby('time')
            reorder_pc = (reorder_pc-pc_min)/(pc_max-pc_min)
            reorder_pc = reorder_pc*2-1  # -1 to 1
            cmip_psedupcs.append(reorder_pc)

    noise_pcs = []
    for unforced in unforced_list:
        unforced = unforced[cmip_var].sel(time=slice(
            str(eof_start)+'-01-01', str(2022)+'-12-31'))  # select the period as pseudo-PC
        member = unforced.shape[0]
        for m in range(member):
            ds_in = unforced.isel(member=m)*missing_xa
            ds_in = ds_in.transpose('time', 'lon', 'lat')
            if month:
                noise_month = []
                for month in range(1, 13):
                    solver = solver_list[month-1]
                    ds_in_month = ds_in.sel(time=ds_in.time.dt.month == month)
                    # print(month, ' ', np.sum(np.isnan(ds_in)).data)
                    psd = solver.projectField(
                        ds_in_month-ds_in_month.mean(dim='time'))
                    if month in reversed_month:
                        psd = -psd
                    noise_month.append(psd)
                noise_month = xr.concat(noise_month, dim='time')
                noise_month = noise_month.sortby('time')
                noise_month = noise_month.isel(
                    mode=n_mode-1).sel(time=slice(str(start_year)+'-01-01', str(end_year)+'-12-31'))
                noise_month = (noise_month-pc_min)/(pc_max-pc_min)
                noise_month = noise_month*2-1
                noise_pcs.append(noise_month)
            else:
                psd = solver_list[0].projectField(ds_in-ds_in.mean(dim='time'))
                # slice the target period, as done for CMIP6.
                psd = psd.isel(
                    mode=n_mode-1).sel(time=slice(str(start_year)+'-01-01', str(end_year)+'-12-31'))
                if reverse:
                    psd = -psd
                psd = (psd-pc_max)/(pc_max-pc_min)
                psd = psd*2-1
                noise_pcs.append(psd)
    # get noise strength

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

    # model
    # pc1 = pc_series.isel(mode=n_mode-1).sel(time=slice(str(start_year)+'-01-01', str(end_year+1)+'-01-01'))
    pc1_norm = (pc1-pc_min)/(pc_max-pc_min)
    pc1_norm = pc1_norm*2-1
    # loop over each timescale to compute the PC trend
    sn = []
    s_list = []
    n_list = []
    for nyears in timescales:
        # get indices for time scale
        sample_inds = np.arange(0, nyears)
        # compute the trend
        time = np.arange(len(pc1_norm))
        m, b = np.polyfit(time[sample_inds],
                          pc1_norm.isel(time=sample_inds), 1)
        # store the trend (signal)
        signal[nyears] = m
        n = np.std(noise[nyears])
        sn.append(m/n)
        s_list.append(m)
        n_list.append(n)

    # get noise and signal
    model_results = {}
    for i, model_pcs in enumerate(cmip_psedupcs):
        model_signal = []
        model_sn = []
        for nyears in timescales:
            sample_inds = np.arange(0, nyears)
            # print(time[sample_inds].shape, model_pcs.isel(time=sample_inds).transpose('time', 'member'))
            m, b = np.polyfit(time[sample_inds], model_pcs.isel(
                time=sample_inds).transpose('time', 'member'), 1)
            # model_signal[nyears] = m # ensemble members
            model_signal.append(np.expand_dims(m, axis=0))  # 1, members
            n = np.std(noise[nyears])
            # model_sn[nyears] = m/n
            model_sn.append(np.expand_dims(m/n, axis=0))
        model_signal = np.concatenate(model_signal, axis=0)  # time, ensemble
        model_sn = np.concatenate(model_sn, axis=0)
        model_signal = xr.DataArray(model_signal, dims=['time', 'member'],
                                    coords={'time': timescales, 'member': np.arange(model_signal.shape[1])})
        model_sn = xr.DataArray(model_sn, dims=['time', 'member'],
                                coords={'time': timescales, 'member': np.arange(model_sn.shape[1])})
        model_results[i] = {'signal': model_signal, 'sn': model_sn}

    results = {
        'sn': sn, 'signal': signal, 'noise': noise, 'n_list': n_list,
        'model_results': model_results,
        'pc': pc1_norm, 'cmip_pcs': cmip_psedupcs,
    }
    return results


results_month = calculate_metrics_cmip(solver_list=solver_list_month, cmip_pcs=all_pcs_month,
                                       unforced_list=unforced_list_month, pc_series=pc_all, month=True, n_mode=n_mode)

results_month_stand = calculate_metrics_cmip(solver_list=solver_list_month_stand, cmip_pcs=all_pcs_month_stand,
                                             unforced_list=unforced_list_month_stand, pc_series=pc_all_stand, month=True, n_mode=n_mode)

results_month_unforced = calculate_metrics_cmip(solver_list=solver_list_month_unforced, cmip_pcs=all_pcs_month_unforced,
                                                unforced_list=unforced_list_month_unforced, pc_series=pc_all_unforced, month=True, n_mode=n_mode)

results_anomaly = calculate_metrics_cmip(solver_list=solver, cmip_pcs=all_pcs,
                                         unforced_list=unforced_list, pc_series=pc_list[0], month=False, n_mode=n_mode)

results_stand = calculate_metrics_cmip(solver_list=solver_stand, cmip_pcs=all_pcs_stand,
                                       unforced_list=unforced_list_stand, pc_series=pc_list_stand[0], month=False, n_mode=n_mode)

results_unforced = calculate_metrics_cmip(solver_list=solver_list_unforced, cmip_pcs=all_pcs_unforced,
                                          unforced_list=unforced_list_unforced, pc_series=pc_unforced[0], month=False, n_mode=n_mode)

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-CMIP-metrics-stand-False-month-True-unforced-False-joint-False'
if late:
    path = path+'-late'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_month, pfile)

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-CMIP-metrics-stand-True-month-True-unforced-False-joint-False'
if late:
    path = path+'-late'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_month_stand, pfile)

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-CMIP-metrics-stand-True-month-True-unforced-True-joint-False'
if late:
    path = path+'-late'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_month_unforced, pfile)

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-CMIP-metrics-stand-False-month-False-unforced-False-joint-False'
if late:
    path = path+'-late'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_anomaly, pfile)

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-CMIP-metrics-stand-True-month-False-unforced-False-joint-False'
if late:
    path = path+'-late'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_stand, pfile)

path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/' + \
    str(eof_start)+'_2022/'+variable + \
    '-CMIP-metrics-stand-True-month-False-unforced-True-joint-False'
if late:
    path = path+'-late'
if n_mode > 1:
    path = path+'-n_mode-'+str(n_mode)
if regen_mask:
    path = path+'-REGEN-mask'
with open(path, 'wb') as pfile:
    pickle.dump(results_unforced, pfile)
