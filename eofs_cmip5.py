import time as clocktime
import glob
import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle
import sys

# principal component analysis
from eofs.multivariate.standard import MultivariateEof
from eofs.xarray import Eof

# define a lambda function to perform natural sort
import re
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", s)]
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', type=str, default='pr')
    parser.add_argument('--stand', type=int, default=0)
    parser.add_argument('--unforced', type=int, default=0)
    parser.add_argument('--month', type=int, default=0)
    parser.add_argument('--joint', type=int, default=0)
    parser.add_argument('--pc_start', type=int, default=1983)
    parser.add_argument('--pc_end', type=int, default=2020)
    parser.add_argument('--start_year', type=int, default=1979)
    parser.add_argument('--eof_end', type=int, default=2022)
    parser.add_argument('--less', type=int, default=0)
    args = vars(parser.parse_args())
    return args

cmipTable = {
    "pr": "Amon",
    "psl": "Amon",
    "tas": "Amon",
    "zmta": "Amon",
    "tos": "Omon",
    "siconc": "OImon",
    "monmaxpr": "Aday",
    "monmaxtasmax": "Aday",
    "monmintasmin": "Aday",
}
cmipVar = {
    "pr": "pr",
    "psl": "psl",
    "tas": "tas",
    "zmta": "ta",
    "tos": "tos",
    "siconc": "siconc",
    "monmaxpr": "pr",
    "monmaxtasmax": "tasmax",
    "monmintasmin": "tasmin",
}
nlat = 72
nlon = 144


if __name__ == '__main__':
    args = get_args()
    variable = args['variable']
    stand = args['stand']
    stand = stand>0
    month = args['month']
    month_bool = month>0
    eof_end = args['eof_end']
    unforced = args['unforced']
    unforced = unforced>0
    print('Month: ', month_bool, ' stand: ', stand,  ' unforced: ', unforced)
    if unforced and not stand:
        sys.exit("NO Such combination")
    less = args['less']>0
    pc_start = args['pc_start']
    pc_end = args['pc_end']
    print('CMIP pseudo-pc start and end: ', pc_start, pc_end)

    root_dir = "/p/lustre3/shiduan/ForceSMIP/CMIP5"
    ncvar = variable  # variable to be used: pr, psl, tas, zmta, tos, siconc, monmaxpr, monmaxtasmax, monmintasmin
    vid = cmipVar[ncvar]  # the variable id in the netcdf file differs – this maps to the standard CMIP variable name
    start_year = args['start_year']
    reference_period = (str(start_year)+"-01-01", str(eof_end+1)+"-01-01") # climatological period (for anomaly calculations)
    print(ncvar)
    print(vid)
    tv_time_period = (str(start_year)+"-01-01", str(eof_end+1)+"-01-01")
    # get training models
    files = glob.glob('/p/lustre3/shiduan/ForceSMIP/CMIP5/*')
    models = [p.split('/')[-1] for p in files]
    models.remove('pr')
    models = sorted(models)
    print(models, ' ', len(models))

    model_mean_list = []
    un_forced_list = []
    all_models = []
    un_forced_std_list = [] # just for std calculation
    # loop over training models
    ds_model = None
    for imodel, model in enumerate(models):
        print(model)
        stime = clocktime.time()
        # get model files
        mpath = '/p/lustre3/shiduan/ForceSMIP/CMIP5/' + model
        mfiles = glob.glob(mpath + '/*')
        count = 0
        # initialize model ensemble xarray dataset
        ds_model = None
        if less: # less ensemble members, but keep all_models
            mfiles = mfiles[:1]
        for im, file in enumerate(mfiles):
            # print member progress
            print('.', end='')
            
            ds = xc.open_dataset(file)
            ds = ds.sel(time=slice(tv_time_period[0], tv_time_period[1]))
            try:
                ds = ds.bounds.add_missing_bounds(axes=['T'])
                ds = ds.squeeze()
            
                if len(ds['time'])>12*(eof_end-start_year):
                    ds['__xarray_dataarray_variable__'] = ds['__xarray_dataarray_variable__']*86400
                    # calculate departures (relative to user-specified reference time period)
                    ds = ds.temporal.departures('__xarray_dataarray_variable__', freq='month', 
                                                reference_period=reference_period)
                    if np.sum(np.isnan(ds['__xarray_dataarray_variable__']))>0:
                        print(file)
                    else:
                        if stand and not unforced: # stand with month std and do not use unforced component. 
                            ds = ds.groupby(ds.time.dt.month)/ds.groupby(ds.time.dt.month).std(dim='time')
                            ds = ds.where(ds.apply(np.isfinite)).fillna(0.0)
                        count += 1
                        
                        if 'ref_time' not in locals():
                            ref_time = ds.time
                        for i, t in enumerate(ds.time.values):
                            m = t.month; y = t.year
                            rt = ref_time.values[i]; rm = rt.month; ry = rt.year
                            if ((ry != y) | (rm != m)):
                                raise ValueError("model time and reference time do not match")
                        ds["time"] = ref_time.copy()
                        ds = ds.bounds.add_missing_bounds(axes=['T'])
                        if ds_model is None:
                            ds_model = ds
                        else:
                            ds_model = xr.concat((ds_model, ds), dim='member')
                else:
                    print('Time period is not long enough')
            except:
                print('exception')      
            
        if count>1:
            ds_model_mean = ds_model.mean(dim='member', skipna=False)
        else:
            ds_model_mean = ds_model
        if count>0:
            ds_model_mean = ds_model_mean.load()
            ds_model_mean = ds_model_mean.bounds.add_missing_bounds(axes=['T'])
            model_mean_list.append(ds_model_mean)
            all_models.append(ds_model)
            print(ds_model['__xarray_dataarray_variable__'].shape, ' ds_model_shape')
        del ds_model, ds_model_mean #, ds_model_anomaly
        # print time elapse for model
        etime = clocktime.time()
        print()
        print("Time elapsed: " + str(etime - stime) + " seconds "+ str(count))
        print()
    ds_multi_model = xr.concat(model_mean_list, dim='model', 
                               # data_vars=['__xarray_dataarray_variable__', 'lon_bnds', 'lat_bnds']
                               )

    if ncvar == "tas" or ncvar == "pr" or ncvar == "psl" or ncvar == "monmaxpr" or ncvar == "monmaxtasmax" or ncvar == "monmintasmin":
        maskfile = "nomask.nc"
        missing_data_maskx = xr.open_dataset(maskfile)
        missing_data = np.where(np.isnan(missing_data_maskx.tas.squeeze().transpose('lon', 'lat')), np.nan, 1)
        missing_xa = xr.where(np.isnan(missing_data_maskx.tas.isel(time=0)), np.nan, 1)
    else:
        maskfile = "maskland.nc"
        missing_data_maskx = xr.open_dataset(maskfile)
        missing_data = np.where(np.isnan(missing_data_maskx.tos.squeeze().transpose('lon', 'lat')), np.nan, 1)
        missing_xa = xr.where(np.isnan(missing_data_maskx.tos.isel(time=0)), np.nan, 1)
    del maskfile
    #missing_data.shape

    ds_multi_model_mean = ds_multi_model.mean(dim='model', skipna=False)
    ds_multi_model_mean = ds_multi_model_mean.bounds.add_missing_bounds()
    print(ds_multi_model_mean)
    ds_multi_model_mean[ncvar] = ds_multi_model_mean['__xarray_dataarray_variable__'].transpose('time', 'lon', 'lat')
    
    masked = ds_multi_model_mean[ncvar] * np.tile(np.expand_dims(missing_data, axis=0), (ds_multi_model_mean[ncvar].shape[0], 1, 1))
    ds = xc.open_dataset('/p/lustre3/shiduan/ForceSMIP/Training/Amon/pr/CanESM5/pr_mon_CanESM5_historical_ssp585_r9i1p2f1.188001-202212.nc')
    lat_weights = ds.spatial.get_weights(axis=['Y'])
    solvers = []
    eofs_record = []
    pc_record = []
    variance = []
    
    if month_bool: # do month by month EOF
        for month in range(1, 13):
            ds_in = masked.sel(time=masked.time.dt.month==month)
            solver = Eof(ds_in, weights=lat_weights)
            solvers.append(solver)
            eofs_record.append(solver.eofs(neofs=5))
            pc_record.append(solver.pcs(npcs=5, pcscaling=0))
            variance.append(solver.varianceFraction())
        all_pcs = []
        for i, model in enumerate(all_models):
            if len(model['__xarray_dataarray_variable__'].shape)>3: # time, member, lat, lon.
                n_members = model['__xarray_dataarray_variable__'].shape[0]
            else:
                model = model.expand_dims(dim="member")
                n_members = 1
            model_pcs = []
            for im in range(n_members):
                ds_in = model['__xarray_dataarray_variable__'].isel(member=im).sel(
                    time=slice(str(start_year)+'-01-01', str(eof_end+1)+'-01-01'))
                ds_in = ds_in.transpose('time', 'lon', 'lat')
                ds_in = ds_in * np.tile(np.expand_dims(missing_data, axis=0), (ds_in.shape[0], 1, 1))
                month_pcs = []
                for month in range(1, 13):
                    ds_in_month = ds_in.sel(time=ds_in.time.dt.month==month)
                    ds_in_month = ds_in_month-ds_in_month.mean(dim='time')
                    solver = solvers[month-1]
                    month_pc = solver.projectField(ds_in_month, neofs=5).sel(
                        time=slice(str(pc_start)+'-01-01', str(pc_end+1)+'-01-01'))
                    month_pcs.append(month_pc)
                month_pcs = xr.concat(month_pcs, dim='time')
                month_pcs = month_pcs.sortby('time')
                model_pcs.append(month_pcs)
            model_pcs = xr.concat(model_pcs, dim='member')
            all_pcs.append(model_pcs)
    else: # do all eofs
        solver = Eof(masked, weights=lat_weights) 
        pcs = solver.pcs(npcs=5, pcscaling=0)
        pc_all = {}
        pc_all['PC'] = pcs
        pc_record.append(pcs)
        eofs = solver.eofs(neofs=5)
        variance.append(solver.varianceFraction())
        eofs_record.append(eofs)
        solvers.append(solver)
        all_pcs = []
        for i, model in enumerate(all_models):
            if len(model['__xarray_dataarray_variable__'].shape)>3: # time, member, lat, lon.
                n_members = model['__xarray_dataarray_variable__'].shape[0]
            else:
                model = model.expand_dims(dim="member")
                n_members = 1
            model_pcs = []
            for im in range(n_members):
                ds_in = model['__xarray_dataarray_variable__'].isel(member=im).sel(
                    time=slice(str(start_year)+'-01-01', str(eof_end+1)+'-01-01'))
                ds_in = ds_in.transpose('time', 'lon', 'lat')
                ds_in = ds_in * np.tile(np.expand_dims(missing_data, axis=0), (ds_in.shape[0], 1, 1))
                ds_in = ds_in - ds_in.mean(dim='time')
                pse_pc = solver.projectField(ds_in, neofs=5).sel(time=slice(str(pc_start)+'-01-01', str(pc_end+1)+'-01-01'))
                model_pcs.append(pse_pc)
            model_pcs = xr.concat(model_pcs, dim='member')
            all_pcs.append(model_pcs)
    
    record = {'solver': solvers, 
              'pc': pc_record,  'all_pcs':all_pcs}
    
    '''if unforced:
        record['unforced_std'] = un_forced_std # this is the unforced anomaly std. used to normalize obs.  '''
    
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(start_year)+'_'+str(eof_end)+'_cmip5/'
    if less:
        path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(start_year)+'_'+str(eof_end)+'_cmip5_less/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+variable+'-solver-stand-'+str(stand)+'-month-'+str(month_bool)+'-unforced-'+str(unforced), 'wb') as pfile:
        pickle.dump(solvers, pfile)
    with open(path+variable+'-record-stand-'+str(stand)+'-month-'+str(month_bool)+'-unforced-'+str(unforced), 'wb') as pfile:
        pickle.dump(record, pfile)
    