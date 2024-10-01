import numpy as np
from sklearn.linear_model import LinearRegression
import xcdat as xc
import glob
import xarray as xr

def get_slope(x, y):
    X_with_intercept = np.empty(shape=(x.shape[0], 2))
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1] = x
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_hat = model.predict(x.reshape(-1, 1))
    residuals = y - y_hat
    rss = np.sum(np.square(residuals))
    xss = np.sum(np.square(x-np.mean(x)))
    n = len(x)
    se = np.sqrt(rss/((n-2)*xss))
    slope = model.coef_[0]
    return slope, se

def get_picontrol_cmip5():
    path = '/p/lustre3/shiduan/ForceSMIP/CMIP5/pr/piControl/'
    files = glob.glob(path+"*.nc")
    picontrol = []
    picontrol_std = []
    for file in files:
        ds = xc.open_dataset(file)
        ds = ds['pr']*86400
        ds = ds.groupby(ds.time.dt.month)-ds.groupby(ds.time.dt.month).mean(dim='time')
        if np.sum(np.isnan(ds))==0:
            picontrol.append(ds)
            ds_std = ds.groupby(ds.time.dt.month)/ds.groupby(ds.time.dt.month).std(dim='time')
            ds_std = ds_std.where(np.isfinite(ds_std)).fillna(0.0)
            picontrol_std.append(ds_std)
        else:
            print(file, ' NAN')
    print(len(picontrol), ' ', len(picontrol_std))
    return picontrol, picontrol_std

def calculate_metrics_cmip5(solver_list, obs, unforced_list, pc_series, missing_xa, 
                            month=False, n_mode=1, start_year=1983, end_year=2020):
    pc1 = pc_series.isel(mode=n_mode-1)
    if month:
        reversed_month = []
        reorder_pc1 = []
        for month in range(1, 13):
            pc1_month = pc1.sel(time=pc1.time.dt.month==month)
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
    # normalize cmip ensemble pcs. 
    if month:
        pseudo_pc_month = []
        for month in range(1, 13):
            # normalize 
            solver = solver_list[month-1]
            ds_in = obs.sel(time=obs.time.dt.month==month)
            pseudo_pc = solver.projectField(ds_in-ds_in.mean(dim='time')).isel(mode=n_mode-1)
            if month in reversed_month: # flip sign based on solver pc. 
                pseudo_pc = -pseudo_pc
            pseudo_pc_month.append(pseudo_pc)
        pseudo_pc = xr.concat(pseudo_pc_month, dim='time')
        pseudo_pc = pseudo_pc.sortby('time')
        pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min) # 0 to 1 
        pseudo_pc = pseudo_pc*2-1 # -1 to 1 
    else:
        # normalize pc series. 
        ds_in = obs
        pseudo_pc = solver_list[0].projectField(ds_in-ds_in.mean(dim='time')).isel(mode=n_mode-1)
        if reverse:
            pseudo_pc = -pseudo_pc
        pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min)
        pseudo_pc = pseudo_pc*2-1
    print(pseudo_pc.shape, pseudo_pc.max().data, ' ', pseudo_pc.min().data)
    # get noise and signal
    noise_pcs = []
    for unforced in unforced_list:
        ds_in = unforced*missing_xa
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
            noise_month = noise_month.isel(mode=n_mode-1)
            noise_month = (noise_month-pc_min)/(pc_max-pc_min)
            noise_month = noise_month*2-1
            noise_pcs.append(noise_month)
        else:
            psd = solver_list[0].projectField(ds_in-ds_in.mean(dim='time'))
            psd = psd.isel(mode=n_mode-1)
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

def calculate_metrics_forcesmip(solver_list, obs, unforced_list, pc_series, missing_xa, month=False, n_mode=1, start_year=1983, end_year=2020, cmip_var='pr'):
    pc1 = pc_series.isel(mode=n_mode-1)
    if month:
        reversed_month = []
        reorder_pc1 = []
        for month in range(1, 13):
            pc1_month = pc1.sel(time=pc1.time.dt.month==month)
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
    # normalize cmip ensemble pcs. 
    if month:
        pseudo_pc_month = []
        for month in range(1, 13):
            # normalize 
            solver = solver_list[month-1]
            ds_in = obs.sel(time=obs.time.dt.month==month)
            pseudo_pc = solver.projectField(ds_in-ds_in.mean(dim='time')).isel(mode=n_mode-1)
            if month in reversed_month: # flip sign based on solver pc. 
                pseudo_pc = -pseudo_pc
            pseudo_pc_month.append(pseudo_pc)
        pseudo_pc = xr.concat(pseudo_pc_month, dim='time')
        pseudo_pc = pseudo_pc.sortby('time')
        pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min) # 0 to 1 
        pseudo_pc = pseudo_pc*2-1 # -1 to 1 
    else:
        # normalize pc series. 
        ds_in = obs
        pseudo_pc = solver_list[0].projectField(ds_in-ds_in.mean(dim='time')).isel(mode=n_mode-1)
        if reverse:
            pseudo_pc = -pseudo_pc
        pseudo_pc = (pseudo_pc-pc_min)/(pc_max-pc_min)
        pseudo_pc = pseudo_pc*2-1
    print(pseudo_pc.shape, pseudo_pc.max().data, ' ', pseudo_pc.min().data)
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