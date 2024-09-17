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
    parser.add_argument('--pc_start', type=int, default=1983)
    parser.add_argument('--pc_end', type=int, default=2018)
    parser.add_argument('--start_year', type=int, default=1979)
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
    unforced = args['unforced']
    unforced = unforced>0
    print(' stand: ', stand, ' unforced: ', unforced)
    if unforced and not stand:
        sys.exit("NO Such combination")
    
    pc_start = args['pc_start']
    pc_end = args['pc_end']
    print('CMIP pseudo-pc start and end: ', pc_start, pc_end)

    models = ['CESM2', 'MIROC6', 'MPI-ESM1-2-LR', 'MIROC-ES2L', 'CanESM5']
    root_dir = "/p/lustre3/shiduan/ForceSMIP"  # path to forcesmip data (NCAR)
    ncvar = variable  # variable to be used: pr, psl, tas, zmta, tos, siconc, monmaxpr, monmaxtasmax, monmintasmin
    vid = cmipVar[ncvar]  # the variable id in the netcdf file differs – this maps to the standard CMIP variable name
    start_year = args['start_year']
    reference_period = (str(start_year)+"-01-01", "2022-12-31") # climatological period (for anomaly calculations)
    print(ncvar)
    print(vid)
    print(models)
    # choose evaluation data
    eval_tier = "Tier1"  # Tier1, Tier2, or Tier3
    tv_time_period = (str(start_year)+"-01-01", "2023-01-01")
    # get training models
    files = glob.glob(root_dir + '/Training/' + cmipTable[ncvar] + '/' + ncvar + '/*')
    models = [p.split('/')[-1] for p in files]
    models = sorted(models)
    if not os.path.exists('data/'):
        os.mkdir('data')

    model_mean_list = []
    un_forced_list = []
    all_models = []
    un_forced_std_list = [] # just for std calculation
    # loop over training models
    for imodel, model in enumerate(models):
        print(model)
        # start timer
        stime = clocktime.time()
        # get model files
        mpath = root_dir + '/Training/' + cmipTable[ncvar] + '/' + ncvar + '/' + model
        mfiles = glob.glob(mpath + '/*')
        # parse file names to get list of model members
        # CESM2 has a non-CMIP naming convention
        if model == "CESM2":
            members = [p.split("ssp370_")[-1].split(".1880")[0] for p in mfiles]
        else:
            members = [p.split("_")[-1].split(".")[0] for p in mfiles]
        members.sort(key=natsort)
        # print progress
        print(str(imodel + 1) + " / " + str(len(models)) + ": " + model + " (" + str(len(members)) + " members)")
        # initialize model ensemble xarray dataset
        ds_model = None
        for im, member in enumerate(members):
            # print member progress
            print('.', end='')
            # get member filename
            fn = glob.glob(mpath + "/*_" + member + ".*.nc")
            # make sure filename is unique
            if len(fn) != 1:
                raise ValueError("Unexpected number of model members")
            else:
                fn = fn[0]
            # load data
            ds = xc.open_dataset(fn)
            ds = ds.bounds.add_missing_bounds(axes=['T'])
            # remove singletons / lon
            ds = ds.squeeze()
            #ds = ds.drop_vars('lon')
            # subset data to user-specified time period
            ds = ds.sel(time=slice(tv_time_period[0], tv_time_period[1]))
            # calculate departures (relative to user-specified reference time period)
            ds = ds.temporal.departures(vid, freq='month', reference_period=reference_period)
            if stand and not unforced: # stand with month std and do not use unforced component. 
                ds = ds.groupby(ds.time.dt.month)/ds.groupby(ds.time.dt.month).std(dim='time')
            if 'file_qf' in ds.variables:
                ds = ds.drop('file_qf')
            if 'ref_time' not in locals():
                ref_time = ds.time
            for i, t in enumerate(ds.time.values):
                m = t.month; y = t.year
                rt = ref_time.values[i]; rm = rt.month; ry = rt.year
                if ((ry != y) | (rm != m)):
                    raise ValueError("model time and reference time do not match")
            ds["time"] = ref_time.copy()
            # add model realization to model ensemble dataset
            if ds_model is None:
                ds_model = ds
            else:
                ds_model = xr.concat((ds_model, ds), dim='member')
        # after looping over members, compute model ensemble mean time series
        ds_model_mean = ds_model.mean(dim='member', skipna=False) # get forced component 
        if not unforced: # Standardized or Anomaly. 
            un_forced = ds_model-ds_model_mean # unforced component. It is standardized. 
            un_forced_list.append(un_forced) # save in the unforced component list
        '''if unforced: # use unforced to standardize. 
            un_forced = ds_model-ds_model_mean # unforced component. It is NOT standardized. 
            un_forced_std_list.append(un_forced)
            std = un_forced.groupby(un_forced.time.dt.month).std(dim=['time', 'member'])
            ds_model = ds_model.groupby(ds_model.time.dt.month)/std # standardize with unforced std. 
            ds_model_mean = ds_model.mean(dim='member', skipna=False) # standardize with unforced std. 
            un_forced = ds_model-ds_model_mean # unforced component. It is standardized. 
            un_forced_list.append(un_forced)
            # ds_model_mean = ds_model.mean(dim='member', skipna=False) # get normalized ensemble mean. '''
        ds_model = ds_model.load() # member, time, lat, lon
        ds_model_mean = ds_model_mean.load()
        all_models.append(ds_model) 
        model_mean_list.append(ds_model_mean)
        data_path = '/p/lustre2/shiduan/ForceSMIP/data/start-'+str(start_year)+'/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        del ds_model, ds_model_mean
        etime = clocktime.time()
        print()
        print("Time elapsed: " + str(etime - stime) + " seconds")
        print()
    ds_multi_model = xr.concat(model_mean_list, dim='model')
    if unforced:
        un_forced_concat = xr.concat(un_forced_std_list, dim='member')
        un_forced_std = un_forced_concat.groupby(un_forced_concat.time.dt.month).std(dim=['time', 'member'])

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
    index_array = xr.DataArray([0, 1, 2, 3, 4], dims="model") # change the index to pick up models
    # index_array = xr.DataArray([0], dims="model") # change the index to pick up models
    ds_multi_model_mean5 = ds_multi_model.isel(model=index_array).mean(dim='model', skipna=False)
    ds_multi_model_mean5 = ds_multi_model_mean5.bounds.add_missing_bounds()
    print(ds_multi_model_mean5)
    ds_multi_model_mean5[ncvar] = ds_multi_model_mean5[vid].transpose('time', 'lon', 'lat')
    
    masked = ds_multi_model_mean5[ncvar] * np.tile(np.expand_dims(missing_data, axis=0), (ds_multi_model_mean5[ncvar].shape[0], 1, 1))
    ds = xc.open_dataset('/p/lustre3/shiduan/ForceSMIP/Training/Amon/pr/CanESM5/pr_mon_CanESM5_historical_ssp585_r9i1p2f1.188001-202212.nc')
    lat_weights = ds.spatial.get_weights(axis=['Y'])
    solvers = []
    eofs_record = []
    pc_record = []
    variance = []
    
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
        n_members = model[vid].shape[0]
        model_pcs = []
        for im in range(n_members):
            ds_in = model[vid].isel(member=im).sel(time=slice(str(start_year)+'-01-01', '2023-01-01'))
            ds_in = ds_in.transpose('time', 'lon', 'lat')
            ds_in = ds_in * np.tile(np.expand_dims(missing_data, axis=0), (ds_in.shape[0], 1, 1))
            ds_in = ds_in - ds_in.mean(dim='time')
            pse_pc = solver.projectField(ds_in, neofs=5).sel(time=slice(str(pc_start)+'-01-01', str(pc_end+1)+'-01-01'))
            model_pcs.append(pse_pc)
        model_pcs = xr.concat(model_pcs, dim='member')
        all_pcs.append(model_pcs)
    
    record = {'solver': solvers, 
              'pc': pc_record, 'unforced_list':un_forced_list, 'all_pcs':all_pcs}
    
    if unforced:
        record['unforced_std'] = un_forced_std # this is the unforced anomaly std. used to normalize obs.  
        
    path = '/p/lustre2/shiduan/ForceSMIP/EOF/modes_all/'+str(start_year)+'_2022_annual/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+variable+'-solver-stand-'+str(stand)+'-unforced-'+str(unforced), 'wb') as pfile:
        pickle.dump(solvers, pfile)
    with open(path+variable+'-record-stand-'+str(stand)+'-unforced-'+str(unforced), 'wb') as pfile:
        pickle.dump(record, pfile)
    if unforced:
        un_forced_std.to_netcdf(path+variable+'_unforced_std.nc')
    
